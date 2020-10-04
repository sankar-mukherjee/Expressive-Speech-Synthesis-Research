import argparse
import traceback

import librosa
import tensorflow as tf
import numpy as np
from tqdm import trange

from utils.config_manager import ConfigManager
from preprocessing.data_handling import load_files, Dataset, DataPrepper
from utils.decorators import ignore_exception, time_it
from utils.scheduling import piecewise_linear_schedule, reduction_schedule
from utils.logging import SummaryManager
from utils.audio import Audio
from model.models import train_models_step, MINE
from model.layers import SelfAttentionBlocks

np.random.seed(42)
tf.random.set_seed(42)

# dynamically allocate GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPUs')
    except Exception:
        traceback.print_exc()


@ignore_exception
@time_it
def validate(tts_model, mine_model, val_dataset, summary_manager):
    val_loss = {'tts_loss': 0., 'mi_loss': 0., 'loss': 0.}
    norm = 0.
    for val_mel, val_text, val_stop in val_dataset.all_batches():
        ouput = tts_model.val_step(inp=val_text,
                                   tar=val_mel,
                                   stop_prob=val_stop)
        # MINE
        text_enc_output_joint, text_enc_output_marginal = mine_model.input_reshape(ouput['text_enc_output'])
        mi_loss = mine_model.call(text_enc_output_joint, text_enc_output_marginal, ouput['gst_output'])
        ouput.update({'mi_loss': mi_loss})
        ouput.update({'loss': ouput['tts_loss'] + 0.1 * tf.maximum(0, ouput['mi_loss'])})

        norm += 1
        val_loss['mi_loss'] += ouput['mi_loss']
        val_loss['tts_loss'] += ouput['tts_loss']
        val_loss['loss'] += ouput['loss']

    val_loss['mi_loss'] /= norm
    val_loss['tts_loss'] /= norm
    val_loss['loss'] /= norm
    summary_manager.display_loss(ouput, tag='Validation', plot_all=True)
    summary_manager.display_attention_heads(ouput, tag='ValidationAttentionHeads')
    summary_manager.display_mel(mel=ouput['mel_linear'][0], tag=f'Validation/linear_mel_out')
    summary_manager.display_mel(mel=ouput['final_output'][0], tag=f'Validation/predicted_mel')
    residual = abs(ouput['mel_linear'] - ouput['final_output'])
    summary_manager.display_mel(mel=residual[0], tag=f'Validation/conv-linear_residual')
    summary_manager.display_mel(mel=val_mel[0], tag=f'Validation/target_mel')
    return val_loss['loss'], val_loss['mi_loss']


# consuming CLI, creating paths and directories, load data

parser = argparse.ArgumentParser()
parser.add_argument('--config', dest='config', type=str)
parser.add_argument('--reset_dir', dest='clear_dir', action='store_true',
                    help="deletes everything under this config's folder.")
parser.add_argument('--reset_logs', dest='clear_logs', action='store_true',
                    help="deletes logs under this config's folder.")
parser.add_argument('--reset_weights', dest='clear_weights', action='store_true',
                    help="deletes weights under this config's folder.")
parser.add_argument('--session_name', dest='session_name', default=None)
args = parser.parse_args()

# model name
config_manager = ConfigManager(config_path='config/wavernn', model_kind='autoregressive', session_name='gst_mine')


config = config_manager.config
config_manager.create_remove_dirs(clear_dir=args.clear_dir,
                                  clear_logs=args.clear_logs,
                                  clear_weights=args.clear_weights)
config_manager.dump_config()
config_manager.print_config()

train_samples, _ = load_files(metafile=str(config_manager.train_datadir / 'train_metafile.txt'),
                              meldir=str(config_manager.train_datadir / 'mels'),
                              num_samples=config['n_samples'])  # (phonemes, mel)
val_samples, _ = load_files(metafile=str(config_manager.train_datadir / 'test_metafile.txt'),
                            meldir=str(config_manager.train_datadir / 'mels'),
                            num_samples=config['n_samples'])  # (phonemes, text, mel)

# get model, prepare data for model, create datasets

# load pretrained model
# this model architecture is the same as TransformerTTS TextEncoder
pretrained_text_encoder_model = SelfAttentionBlocks(model_dim=config['encoder_model_dimension'],
                                                    dropout_rate=config['encoder_model_dimension'],
                                                    num_heads=config['encoder_num_heads'],
                                                    feed_forward_dimension=config['encoder_feed_forward_dimension'],
                                                    maximum_position_encoding=config['encoder_max_position_encoding'],
                                                    dense_blocks=config['encoder_dense_blocks'],
                                                    conv_filters=config['encoder_attention_conv_filters'],
                                                    kernel_size=config['encoder_attention_conv_kernel'],
                                                    conv_activation='relu',
                                                    name='TextEncoder')
ckpt = tf.train.Checkpoint(net=pretrained_text_encoder_model)
manager = tf.train.CheckpointManager(ckpt, config['pretrained_text_encoder_model'] + 'autoregressive_weights',
                                     max_to_keep=None)
ckpt.restore(manager.latest_checkpoint)
pretrained_text_encoder_model.trainable = False

# load TTS model assign textEncoder with pretrained model
tts_model = config_manager.get_model()
config_manager.compile_model(tts_model)
# tts_model.text_encoder = pretrained_text_encoder_model

# Initialized MINE model
mine_model = MINE(dense_hidden_units=config['mine_dense_hidden_units'])
mine_model.compile(
    optimizer=tf.keras.optimizers.Adam(np.array(config['learning_rate_schedule'])[0, 1].astype(np.float32),
                                       beta_1=0.9,
                                       beta_2=0.98,
                                       epsilon=1e-9))
mi_loss_holder = ({'mi_loss': tf.cast(0.0, tf.float32)})

# load DATA
data_prep = DataPrepper(config=config,
                        tokenizer=tts_model.text_pipeline.tokenizer)

test_list = [data_prep(s) for s in val_samples]
train_dataset = Dataset(samples=train_samples,
                        preprocessor=data_prep,
                        batch_size=config['batch_size'],
                        mel_channels=config['mel_channels'],
                        shuffle=True)
val_dataset = Dataset(samples=val_samples,
                      preprocessor=data_prep,
                      batch_size=config['batch_size'],
                      mel_channels=config['mel_channels'],
                      shuffle=False)

# get audio config ready to predict unknown sentence
audio = Audio(config)

# create logger and checkpointer and restore latest model
summary_manager = SummaryManager(model=tts_model, log_dir=config_manager.log_dir, config=config)
checkpoint = tf.train.Checkpoint(step=tf.Variable(1),
                                 optimizer=tts_model.optimizer,
                                 net=tts_model)
manager = tf.train.CheckpointManager(checkpoint, str(config_manager.weights_dir),
                                     max_to_keep=config['keep_n_weights'],
                                     keep_checkpoint_every_n_hours=config['keep_checkpoint_every_n_hours'])
checkpoint.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print(f'\nresuming training from step {tts_model.step} ({manager.latest_checkpoint})')
else:
    print(f'\nstarting training from scratch')

if config['debug'] is True:
    print('\nWARNING: DEBUG is set to True. Training in eager mode.')

# main event
print('\nTRAINING')
losses = []
_ = train_dataset.next_batch()
t = trange(tts_model.step, config['max_steps'], leave=True)
for _ in t:
    t.set_description(f'step {tts_model.step}')

    # get data
    mel, phonemes, stop = train_dataset.next_batch()

    # TTS initialization
    decoder_prenet_dropout = piecewise_linear_schedule(tts_model.step, config['decoder_prenet_dropout_schedule'])
    learning_rate = piecewise_linear_schedule(tts_model.step, config['learning_rate_schedule'])
    reduction_factor = reduction_schedule(tts_model.step, config['reduction_factor_schedule'])
    drop_n_heads = tf.cast(reduction_schedule(tts_model.step, config['head_drop_schedule']), tf.int32)
    tts_model.set_constants(decoder_prenet_dropout=decoder_prenet_dropout,
                            learning_rate=learning_rate,
                            reduction_factor=reduction_factor,
                            drop_n_heads=drop_n_heads)
    # train both models
    output = train_models_step(phonemes, mel, stop, tts_model, mine_model, mi_loss_holder)

    # logging
    losses.append(float(output['loss']))
    t.display(f'reduction factor {reduction_factor}', pos=10)
    t.display(f'step loss: {losses[-1]}', pos=1)
    for pos, n_steps in enumerate(config['n_steps_avg_losses']):
        if len(losses) > n_steps:
            t.display(f'{n_steps}-steps average loss: {sum(losses[-n_steps:]) / n_steps}', pos=pos + 2)

    summary_manager.display_loss(output, tag='Train')
    summary_manager.display_scalar(tag='Meta/decoder_prenet_dropout', scalar_value=tts_model.decoder_prenet.rate)
    summary_manager.display_scalar(tag='Meta/learning_rate', scalar_value=tts_model.optimizer.lr)
    summary_manager.display_scalar(tag='Meta/reduction_factor', scalar_value=tts_model.r)
    summary_manager.display_scalar(tag='Meta/drop_n_heads', scalar_value=tts_model.drop_n_heads)
    if tts_model.step % config['train_images_plotting_frequency'] == 0:
        summary_manager.display_attention_heads(output, tag='TrainAttentionHeads')
        summary_manager.display_mel(mel=output['mel_linear'][0], tag=f'Train/linear_mel_out')
        summary_manager.display_mel(mel=output['final_output'][0], tag=f'Train/predicted_mel')
        residual = abs(output['mel_linear'] - output['final_output'])
        summary_manager.display_mel(mel=residual[0], tag=f'Train/conv-linear_residual')
        summary_manager.display_mel(mel=mel[0], tag=f'Train/target_mel')

    if tts_model.step % config['weights_save_frequency'] == 0:
        save_path = manager.save()
        t.display(f'checkpoint at step {tts_model.step}: {save_path}', pos=len(config['n_steps_avg_losses']) + 2)

    if tts_model.step % config['validation_frequency'] == 0:
        val_loss, time_taken = validate(tts_model=tts_model,
                                        mine_model=mine_model,
                                        val_dataset=val_dataset,
                                        summary_manager=summary_manager)
        t.display(
            f'validation loss at step {tts_model.step}: {val_loss[0]} and mi_loss: {val_loss[1]} (took {time_taken}s)',
            pos=len(config['n_steps_avg_losses']) + 3)

    if tts_model.step % config['prediction_frequency'] == 0 and (tts_model.step >= config['prediction_start_step']):
        for j in range(config['n_predictions']):
            mel, phonemes, stop = test_list[j]
            t.display(f'Predicting {j}', pos=len(config['n_steps_avg_losses']) + 4)
            pred = tts_model.predict(phonemes, mel, encode=False, verbose=False)
            pred_mel = pred['mel']
            target_mel = mel
            summary_manager.display_attention_heads(outputs=pred, tag=f'TestAttentionHeads/sample {j}')
            summary_manager.display_mel(mel=pred_mel, tag=f'Test/sample {j}/predicted_mel')
            summary_manager.display_mel(mel=target_mel, tag=f'Test/sample {j}/target_mel')
            if tts_model.step > config['audio_start_step']:
                summary_manager.display_audio(tag=f'Target/sample {j}', mel=target_mel)
                summary_manager.display_audio(tag=f'Prediction/sample {j}', mel=pred_mel)

                # predict unknown sentence
                ref_type = {'sarcasm', 'commanding'}
                ref_path = '../../database/ref_audio/'

                f = open(ref_path + 'test_sentence', "r")
                test_sentence = list(f)
                f.close()

                for rt in ref_type:
                    y, _ = librosa.load(ref_path + rt + '.wav', sr=config['sampling_rate'])
                    ref_mel = np.transpose(audio.mel_spectrogram(y))
                    u_pred = tts_model.predict(test_sentence[0].rstrip(), ref_mel)
                    summary_manager.display_mel(mel=u_pred['mel'], tag=rt + ' Predicted_mel_sample ' + str(j))
                    summary_manager.display_attention_heads(outputs=u_pred,
                                                            tag=rt + ' AttentionHeads_sample ' + str(j))
                    summary_manager.display_audio(tag=rt + ' sample ' + str(j), mel=u_pred['mel'])

print('Done.')
