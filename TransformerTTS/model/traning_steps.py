import tensorflow as tf

from utils.losses import weighted_sum_losses, l2_loss

# dynamically allocate GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# train both models
def train_models_step(tts_mel, tts_phonemes, tts_stop, tts_speaker,
                      mine_mel, mine_phonemes, mine_stop, mine_speaker,
                      tts_model, mine_model, mi_holder,
                      train_text_encoder: bool, train_style_encoder: bool, train_decoder: bool,
                      use_style_loss: bool):
    # TTS model
    style_loss = 0
    # prepare inputs
    tts_processed_real, tts_processed_mel, tts_processed_stop, tts_mel_len = tts_model.input_reshape(tts_mel, tts_stop)
    with tf.GradientTape() as tts_tape:
        model_out = tts_model.call(inputs=tts_phonemes,
                                   targets=tts_processed_mel,
                                   spk_embed=tts_speaker,
                                   train_text_encoder=train_text_encoder,
                                   train_style_encoder=train_style_encoder,
                                   train_decoder=train_decoder)

        # compute loss
        tts_loss, tts_loss_vals = weighted_sum_losses((tts_processed_real,
                                                       tts_processed_stop,
                                                       tts_processed_real),
                                                      (model_out['final_output'][:, :tts_mel_len, :],
                                                       model_out['stop_prob'][:, :tts_mel_len, :],
                                                       model_out['mel_linear'][:, :tts_mel_len, :]),
                                                      tts_model.loss,
                                                      tts_model.loss_weights)
        if use_style_loss:
            # Style encoder 2nd time
            gst_output2, _, _ = tts_model.style_encoder(model_out['final_output'],
                                                        drop_n_heads=tts_model.drop_n_heads,
                                                        training=train_style_encoder)  # batch x 1 x dim
            # style loss
            style_loss = l2_loss(gst_output2, model_out['gst_output'])
            # add with tts_loss
            tts_loss = tts_loss + style_loss

        # total loss
        total_loss = tts_loss + mi_holder['weight_factor'] * tf.maximum(0, mi_holder['mi_loss'])
    # gradients
    tts_gradients = tts_tape.gradient(total_loss, tts_model.trainable_variables)
    tts_model.optimizer.apply_gradients(zip(tts_gradients, tts_model.trainable_variables))

    # MINE
    if mi_holder['use_mine']:
        if mi_holder['mine_sep_call']:
            # prepare inputs
            _, mine_processed_mel, _, _ = tts_model.input_reshape(mine_mel, mine_stop)
            # Freeze weights and get style from tts encoder
            _, _, _, _, _, mine_gst_out, mine_text_enc_out = tts_model.call_encoder(inputs=mine_phonemes,
                                                                                    targets=mine_processed_mel,
                                                                                    spk_embed=mine_speaker,
                                                                                    training_text_encoder=True,
                                                                                    training_style_encoder=True)
        else:
            mine_gst_out = model_out['gst_output']
            mine_text_enc_out = model_out['text_enc_output']
            mine_speaker = tts_speaker

        # mi loss gather
        mi_loss_gather = {}
        for i, m in enumerate(mine_model):
            with tf.GradientTape() as mine_tape:
                mi_loss, exp_terms = m.call(mine_text_enc_out, mine_gst_out, mine_speaker, mi_holder)
                # negative for gradient ascent
                mi_gradients = mine_tape.gradient(-mi_loss, m.trainable_variables)
                m.optimizer.apply_gradients(zip(mi_gradients, m.trainable_variables))
                mi_loss_gather[m.name + ':' + m.pair_type] = mi_loss
        model_out.update({'mi_loss': {key: mi_loss_gather[key] for key in mi_loss_gather}})
        mi_holder.update({'mi_loss': sum(mi_loss_gather.values()), 'exp_terms': exp_terms})

    # update variables
    model_out.update({'tts_loss': tts_loss,
                      'style_loss': style_loss,
                      'loss': total_loss,
                      'losses': {'output': tts_loss_vals[0],
                                 'stop_prob': tts_loss_vals[1],
                                 'mel_linear': tts_loss_vals[2]},
                      'reduced_target': tts_processed_mel
                      })
    return model_out
