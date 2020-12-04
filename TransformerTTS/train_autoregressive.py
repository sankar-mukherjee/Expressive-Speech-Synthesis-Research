import argparse
import itertools
import traceback

import librosa
import tensorflow as tf
import numpy as np
from tqdm import trange

from utils.config_manager import ConfigManager
from preprocessing.data_handling import load_files, Dataset, DataPrepper
from utils.scheduling import piecewise_linear_schedule, reduction_schedule
from utils.logging import SummaryManager
from utils.audio import Audio
from model.traning_steps import train_models_step
from model.MINE import MINE, CLUB
from model.load_pretrained_models import load_pretrained_models

np.random.seed(42)
tf.random.set_seed(42)

########################################################################################################################
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

########################################################################################################################
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

########################################################################################################################
# model name
config_manager = ConfigManager(config_path='config/wavernn', model_kind='autoregressive',
                               session_name='test_mine')

config = config_manager.config
config_manager.create_remove_dirs(clear_dir=args.clear_dir,
                                  clear_logs=args.clear_logs,
                                  clear_weights=args.clear_weights)
config_manager.dump_config()
config_manager.print_config()

########################################################################################################################
# get model, prepare data for model, create datasets

# load TTS model
tts_model = config_manager.get_model()
config_manager.compile_model(tts_model)

########################################################################################################################
# Initialized MINE model
mine_model = []
if config['use_mine']:
    no_mine_net = config['mine_pair_types']
    if not config['use_speaker_style_tts']:
        no_mine_net = [config['mine_pair_types'][0]]
    mine_model = []
    for i in no_mine_net:
        if config['mine_type'] == 'CLUB':
            t = CLUB(dense_hidden_units=config['mine_dense_hidden_units'], pair_type=i)
            mine_model.append(t)
        elif config['mine_type'] == 'MINE':
            t = MINE(conv_filters=config['mine_conv_filters'],
                     conv_kernel=config['mine_conv_kernel'],
                     dense_hidden_units=config['mine_dense_hidden_units'],
                     beta_values=config['mine_beta_values'],
                     divergence_type=config['divergence_type'],
                     pair_type=i)
            mine_model.append(t)
        elif config['mine_type'] == 'MINE_CLUB':
            t = CLUB(dense_hidden_units=config['mine_dense_hidden_units'], pair_type=i)
            mine_model.append(t)
            t = MINE(conv_filters=config['mine_conv_filters'],
                     conv_kernel=config['mine_conv_kernel'],
                     dense_hidden_units=config['mine_dense_hidden_units'],
                     beta_values=config['mine_beta_values'],
                     divergence_type=config['divergence_type'],
                     pair_type=i)
            mine_model.append(t)
        else:
            print('mine_type is not right')

    for i in mine_model:
        i.compile(
            optimizer=tf.keras.optimizers.Adam(np.array(config['learning_rate_mine_schedule'])[0, 1].astype(np.float32),
                                               beta_1=0.9,
                                               beta_2=0.98,
                                               epsilon=1e-9))
mi_holder = ({'use_mine': config['use_mine'],
              'mine_sep_call': config['mine_sep_call'],
              'mi_loss': tf.cast(0.0, tf.float32),
              'exp_terms': tf.zeros([len(config['mine_beta_values']), 2], tf.float32),
              'weight_factor': tf.cast(config['mine_weight_factor'], tf.float32),
              'smoothing_factor': tf.cast(config['mine_smoothing_factor'], tf.float32)})

########################################################################################################################
# checkpoint and restore latest model

# restore TTS model
checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=tts_model.optimizer, net=tts_model)
tts_manager = tf.train.CheckpointManager(checkpoint, str(config_manager.tts_weights_dir), max_to_keep=None)
checkpoint.restore(tts_manager.latest_checkpoint)

# restore MINE model
mine_manager = {}
if config['use_mine']:
    for i, m in enumerate(mine_model):
        checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=m.optimizer, net=m)
        mine_manager[i] = tf.train.CheckpointManager(checkpoint, str(config_manager.mine_weights_dir[i]),
                                                     max_to_keep=None)
        checkpoint.restore(mine_manager[i].latest_checkpoint)

if tts_manager.latest_checkpoint:
    print(f'\nresuming training from step {tts_model.step} ({tts_manager.latest_checkpoint})')
    if config['use_mine']:
        for i, m in enumerate(mine_manager):
            print(f'\nresuming training from step {mine_model[i].step} ({mine_manager[i].latest_checkpoint})')
else:
    print(f'\nstarting training from scratch')
    ####################################################################################################################
    # load pretrained model
    if config['use_pretrained']:
        if config['use_pretrained_type'] == 'all':
            pretrain_weights_path = config['pretrained_all_weights']
            pretrain_config_path = config['pretrained_all_config']
            print(f'\nload pre-trained all weights: ', config['pretrained_all_weights'])
        else:
            pretrain_weights_path = config['pretrained_text_encoder_weights']
            pretrain_config_path = config['pretrained_text_encoder_config']
            print(f'\nload pre-trained text-encoder weights: ', config['pretrained_text_encoder_weights'])

        # load pretrained model
        tts_model_pretrained = load_pretrained_models(pretrain_config_path, pretrain_weights_path,
                                                      config['use_pretrained_type'])
        # dry run
        _ = tts_model.call(inputs=tf.random.uniform(shape=[1, 50], dtype=tf.float32),
                           targets=tf.random.uniform(shape=[1, 50, 80], dtype=tf.float32),
                           spk_embed=tf.random.uniform(shape=[1, 1, 256], dtype=tf.float32),
                           train_text_encoder=True,
                           train_style_encoder=True,
                           train_decoder=True)

        if config['use_pretrained_type'] == 'all':
            # assign whole network with pretrained model
            for i, var in enumerate(tts_model.trainable_variables):
                text_enc_var = tts_model_pretrained.trainable_variables[i]
                tts_model.trainable_variables[i].assign(text_enc_var)
        else:
            # assign textEncoder (layer 1) with pretrained model
            for i, var in enumerate(tts_model.layers[1].trainable_variables):
                text_enc_var = tts_model_pretrained.layers[1].trainable_variables[i]
                tts_model.layers[1].trainable_variables[i].assign(text_enc_var)

        # freeze textEncoder weights
        tts_model.text_encoder.trainable = False

########################################################################################################################
# load DATA
if config['use_speaker_style_tts']:
    spk_embed_dir = str(config_manager.train_datadir / 'spk_embeds')
else:
    spk_embed_dir = None

train_samples, _ = load_files(metafile=str(config_manager.train_datadir / 'train_metafile.txt'),
                              mel_dir=str(config_manager.train_datadir / 'mels'),
                              spk_embed_dir=spk_embed_dir,
                              num_samples=config['n_samples'])  # (phonemes, text, mel, speaker)
val_samples, _ = load_files(metafile=str(config_manager.train_datadir / 'test_metafile.txt'),
                            mel_dir=str(config_manager.train_datadir / 'mels'),
                            spk_embed_dir=spk_embed_dir,
                            num_samples=1)  # (phonemes, text, mel, speaker)

data_prep = DataPrepper(config=config, tokenizer=tts_model.text_pipeline.tokenizer)

test_list = [data_prep(s) for s in val_samples]
tts_train_dataset = Dataset(samples=train_samples,
                            preprocessor=data_prep,
                            batch_size=config['tts_batch_size'],
                            mel_channels=config['mel_channels'],
                            shuffle=True)
mine_train_dataset = Dataset(samples=train_samples,
                             preprocessor=data_prep,
                             batch_size=np.array(config['mine_batch_size_schedule'])[0, 1],  # take the first batch size
                             mel_channels=config['mel_channels'],
                             shuffle=True)

# get audio config ready to predict unknown sentence
audio = Audio(config)

########################################################################################################################
# create logger
summary_manager = SummaryManager(model=tts_model, log_dir=config_manager.log_dir, config=config)

if config['debug'] is True:
    print('\nWARNING: DEBUG is set to True. Training in eager mode.')

########################################################################################################################
# main event
print('\nTRAINING')
losses = []
_ = tts_train_dataset.next_batch()
_ = mine_train_dataset.next_batch()
t = trange(tts_model.step, config['max_steps'], leave=True)
for _ in t:
    t.set_description(f'step {tts_model.step}')

    # get batches of data
    tts_mel, tts_phonemes, tts_stop, tts_spk = tts_train_dataset.next_batch()
    tts_spk = tf.expand_dims(tts_spk, 1)
    # check for dummy value for speaker embedding
    if tf.shape(tts_spk)[2] == 1:
        tts_spk = None

    # change mine batch size
    if config['use_mine'] and config['mine_sep_call']:
        if tts_model.step == np.array(config['mine_batch_size_schedule'])[1, 0]:
            mine_train_dataset.change_batches(np.array(config['mine_batch_size_schedule'])[1, 1])
        mine_mel, mine_phonemes, mine_stop, mine_spk = mine_train_dataset.next_batch()
        mine_spk = tf.expand_dims(mine_spk, 1)
        # check for dummy value for speaker embedding
        if tf.shape(mine_spk)[2] == 1:
            mine_spk = None
    else:
        mine_mel, mine_phonemes, mine_stop, mine_spk = 0, 0, 0, 0

    # TTS initialization
    decoder_prenet_dropout = piecewise_linear_schedule(tts_model.step, config['decoder_prenet_dropout_schedule'])
    learning_rate = piecewise_linear_schedule(tts_model.step, config['learning_rate_tts_schedule'])
    reduction_factor = reduction_schedule(tts_model.step, config['reduction_factor_schedule'])
    drop_n_heads = tf.cast(reduction_schedule(tts_model.step, config['head_drop_schedule']), tf.int32)
    tts_model.set_constants(decoder_prenet_dropout=decoder_prenet_dropout,
                            learning_rate=learning_rate,
                            reduction_factor=reduction_factor,
                            drop_n_heads=drop_n_heads)
    # train both models
    output = train_models_step(tts_mel, tts_phonemes, tts_stop, tts_spk,
                               mine_mel, mine_phonemes, mine_stop, mine_spk,
                               tts_model, mine_model, mi_holder,
                               config['train_text_encoder'], config['train_style_encoder'], config['train_decoder'],
                               config['use_style_loss'])

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
        summary_manager.display_mel(mel=tts_mel[0], tag=f'Train/target_mel')

    if tts_model.step % config['weights_save_frequency'] == 0:
        save_path = tts_manager.save()
        t.display(f'checkpoint at step {tts_model.step}: {save_path}', pos=len(config['n_steps_avg_losses']) + 2)
        if config['use_mine']:
            for i, m in enumerate(mine_manager):
                save_path = mine_manager[i].save()
                t.display(f'checkpoint at step {mine_model[i].step}: {save_path}',
                          pos=len(config['n_steps_avg_losses']) + 2)

    if tts_model.step % config['prediction_frequency'] == 0 and (tts_model.step >= config['prediction_start_step']):
        for j in range(config['n_predictions']):
            mel, phonemes, stop, spk_embed = test_list[j]
            # check for dummy value for speaker embedding
            if spk_embed[0] == np.inf:
                spk_embed = None
            else:
                spk_embed = tf.expand_dims(spk_embed, 0)
            t.display(f'Predicting {j}', pos=len(config['n_steps_avg_losses']) + 4)
            pred = tts_model.predict(phonemes, mel, spk_embed, encode=False, verbose=False)
            pred_mel = pred['mel']
            target_mel = mel
            summary_manager.display_attention_heads(outputs=pred, tag=f'TestAttentionHeads/sample {j}')
            summary_manager.display_mel(mel=pred_mel, tag=f'Test/sample {j}/predicted_mel')
            summary_manager.display_mel(mel=target_mel, tag=f'Test/sample {j}/target_mel')
            if tts_model.step > config['audio_start_step']:
                summary_manager.display_audio(tag=f'Target/sample {j}', mel=target_mel)
                summary_manager.display_audio(tag=f'Prediction/sample {j}', mel=pred_mel)

                # predict unknown sentence
                f = open(config['base_directory'] + config['text_path'], "r")
                test_sentence = list(f)
                f.close()

                if config['use_speaker_style_tts']:
                    syn_types = [list(zip(x, config['speaker_types']))
                                 for x in itertools.permutations(config['style_types'], len(config['speaker_types']))]
                    syn_types = [item for sublist in syn_types for item in sublist]
                else:
                    syn_types = [config['style_types']]

                for rt in syn_types:
                    y, _ = librosa.load(config['base_directory'] + config['style_audio_path'] + rt[0] + '.wav',
                                        sr=config['sampling_rate'])
                    ref_mel = np.transpose(audio.mel_spectrogram(y))

                    if config['use_speaker_style_tts']:
                        spk_embed = np.load(config['base_directory'] + config['speaker_audio_path'] +
                                            'speaker_embedding/' + rt[1] + '.npy')
                        spk_embed = tf.expand_dims(spk_embed, 0)
                        file_name = '-'.join(rt)
                    else:
                        spk_embed = None
                        file_name = rt[0]

                    u_pred = tts_model.predict(test_sentence[0].rstrip(), ref_mel, spk_embed)
                    summary_manager.display_mel(mel=u_pred['mel'], tag=file_name + ' Predicted_mel_sample ' + str(j))
                    summary_manager.display_attention_heads(outputs=u_pred,
                                                            tag=file_name + ' AttentionHeads_sample ' + str(j))
                    summary_manager.display_audio(tag=file_name + ' sample ' + str(j), mel=u_pred['mel'])

print('Done.')
