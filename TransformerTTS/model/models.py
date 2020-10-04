import sys

import tensorflow as tf

from model.layers import DecoderPrenet, Postnet, SelfAttentionBlocks, CrossAttentionBlocks, \
    ReferenceEncoderGST, MineNet
from model.transformer_utils import create_encoder_padding_mask, create_mel_padding_mask, create_look_ahead_mask
from preprocessing.text import Pipeline
from utils.losses import weighted_sum_losses, masked_mean_absolute_error, new_scaled_crossentropy

# dynamically allocate GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# train both models
def train_models_step(inp, tar, stop_prob, tts_model, mine_model, mi_loss_holder):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    tar_stop_prob = stop_prob[:, 1:]

    mel_len = int(tf.shape(tar_inp)[1])
    tar_mel = tar_inp[:, 0::tts_model.r, :]

    with tf.GradientTape() as tts_tape:
        model_out = tts_model.call(inputs=inp,
                                   targets=tar_mel,
                                   training=False)
        tts_loss, tts_loss_vals = weighted_sum_losses((tar_real,
                                                       tar_stop_prob,
                                                       tar_real),
                                                      (model_out['final_output'][:, :mel_len, :],
                                                       model_out['stop_prob'][:, :mel_len, :],
                                                       model_out['mel_linear'][:, :mel_len, :]),
                                                      tts_model.loss,
                                                      tts_model.loss_weights)
        total_loss = tts_loss + 0.1 * tf.maximum(0, mi_loss_holder['mi_loss'])
    tts_gradients = tts_tape.gradient(total_loss, tts_model.trainable_variables)
    tts_model.optimizer.apply_gradients(zip(tts_gradients, tts_model.trainable_variables))

    # MINE
    text_enc_output_joint, text_enc_output_marginal = mine_model.input_reshape(model_out['text_enc_output'])
    with tf.GradientTape() as mine_tape:
        # negative for gradient ascent later in mine_tape
        mi_loss = mine_model.call(text_enc_output_joint, text_enc_output_marginal, model_out['gst_output'])
        mi_gradients = mine_tape.gradient(-mi_loss, mine_model.trainable_variables)
    mine_model.optimizer.apply_gradients(zip(mi_gradients, mine_model.trainable_variables))

    model_out.update({'tts_loss': tts_loss, 'mi_loss': mi_loss, 'loss': total_loss})
    model_out.update(
        {'losses': {'output': tts_loss_vals[0], 'stop_prob': tts_loss_vals[1], 'mel_linear': tts_loss_vals[2]}})
    model_out.update({'reduced_target': tar_mel})
    mi_loss_holder.update({'mi_loss': mi_loss})

    return model_out


class MINE(tf.keras.models.Model):
    def __init__(self,
                 dense_hidden_units: list,
                 **kwargs):
        super(MINE, self).__init__(**kwargs)
        self.mine_net = MineNet(dense_hidden_units=dense_hidden_units, name='MineNet')

    @staticmethod
    def input_reshape(text_enc_output):
        one_random_char = tf.random.shuffle(tf.range(tf.shape(text_enc_output)[1]))[:1]
        joint = tf.gather(text_enc_output, one_random_char, axis=1)
        marginal = tf.random.shuffle(joint)
        return joint, marginal

    def call(self, text_enc_output_j, text_enc_output_m, gst_output):
        joint = tf.concat([gst_output, text_enc_output_j], axis=-1)
        joint = self.mine_net(joint)

        marginal = tf.concat([gst_output, text_enc_output_m], axis=-1)
        marginal = self.mine_net(marginal)

        # Mutual Information or KL Divergence
        loss = tf.cast(tf.reduce_mean(joint) - tf.math.log(tf.reduce_mean(tf.exp(marginal))), tf.float32)
        return loss


class AutoregressiveTransformer(tf.keras.models.Model):

    def __init__(self,
                 encoder_model_dimension: int,
                 decoder_model_dimension: int,
                 encoder_num_heads: list,
                 decoder_num_heads: list,
                 encoder_maximum_position_encoding: int,
                 decoder_maximum_position_encoding: int,
                 encoder_dense_blocks: int,
                 decoder_dense_blocks: int,
                 encoder_prenet_dimension: int,
                 decoder_prenet_dimension: int,
                 postnet_conv_filters: int,
                 postnet_conv_layers: int,
                 postnet_kernel_size: int,
                 dropout_rate: float,
                 mel_start_value: int,
                 mel_end_value: int,
                 mel_channels: int,
                 phoneme_language: str,
                 with_stress: bool,

                 ref_encoder_filters: list,
                 ref_encoder_kernel_size: int,
                 ref_encoder_strides: int,
                 ref_encoder_gru_cell_units: int,
                 gst_style_embed_dim: int,
                 gst_multi_num_heads: int,
                 gst_heads: int,
                 batch_size: int,

                 encoder_attention_conv_filters: int = None,
                 decoder_attention_conv_filters: int = None,
                 encoder_attention_conv_kernel: int = None,
                 decoder_attention_conv_kernel: int = None,
                 encoder_feed_forward_dimension: int = None,
                 decoder_feed_forward_dimension: int = None,
                 decoder_prenet_dropout=0.5,
                 max_r: int = 10,
                 debug=False,
                 **kwargs):
        super(AutoregressiveTransformer, self).__init__(**kwargs)
        self.start_vec = tf.ones((1, mel_channels), dtype=tf.float32) * mel_start_value
        self.end_vec = tf.ones((1, mel_channels), dtype=tf.float32) * mel_end_value
        self.stop_prob_index = 2
        self.max_r = max_r
        self.r = max_r
        self.mel_channels = mel_channels
        self.drop_n_heads = 0
        self.batch_size = batch_size
        self.text_pipeline = Pipeline.default_pipeline(phoneme_language,
                                                       add_start_end=True,
                                                       with_stress=with_stress)
        self.text_encoder_prenet = tf.keras.layers.Embedding(self.text_pipeline.tokenizer.vocab_size,
                                                             encoder_prenet_dimension,
                                                             name='TextEmbedding')
        self.text_encoder = SelfAttentionBlocks(model_dim=encoder_model_dimension,
                                                dropout_rate=dropout_rate,
                                                num_heads=encoder_num_heads,
                                                feed_forward_dimension=encoder_feed_forward_dimension,
                                                maximum_position_encoding=encoder_maximum_position_encoding,
                                                dense_blocks=encoder_dense_blocks,
                                                conv_filters=encoder_attention_conv_filters,
                                                kernel_size=encoder_attention_conv_kernel,
                                                conv_activation='relu',
                                                name='TextEncoder')
        self.style_encoder = ReferenceEncoderGST(conv_filters=ref_encoder_filters,
                                                 kernel_size=ref_encoder_kernel_size,
                                                 strides=ref_encoder_strides,
                                                 gru_cell_units=ref_encoder_gru_cell_units,
                                                 gst_style_embed_dim=gst_style_embed_dim,
                                                 multi_num_heads=gst_multi_num_heads,
                                                 gst_heads=gst_heads,
                                                 batch_size=self.batch_size,
                                                 name='RefEncoderGST')
        self.decoder_prenet = DecoderPrenet(model_dim=decoder_model_dimension,
                                            dense_hidden_units=decoder_prenet_dimension,
                                            dropout_rate=decoder_prenet_dropout,
                                            name='DecoderPrenet')
        self.decoder = CrossAttentionBlocks(model_dim=decoder_model_dimension,
                                            dropout_rate=dropout_rate,
                                            num_heads=decoder_num_heads,
                                            feed_forward_dimension=decoder_feed_forward_dimension,
                                            maximum_position_encoding=decoder_maximum_position_encoding,
                                            dense_blocks=decoder_dense_blocks,
                                            conv_filters=decoder_attention_conv_filters,
                                            conv_kernel=decoder_attention_conv_kernel,
                                            conv_activation='relu',
                                            conv_padding='causal',
                                            name='Decoder')
        self.final_proj_mel = tf.keras.layers.Dense(self.mel_channels * self.max_r, name='FinalProj')
        self.decoder_postnet = Postnet(mel_channels=mel_channels,
                                       conv_filters=postnet_conv_filters,
                                       conv_layers=postnet_conv_layers,
                                       kernel_size=postnet_kernel_size,
                                       name='Postnet')

        self.training_input_signature = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(None, None, mel_channels), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None), dtype=tf.int32)
        ]
        self.forward_input_signature = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(None, None, mel_channels), dtype=tf.float32),
        ]
        self.encoder_signature = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(None, None, mel_channels), dtype=tf.float32),
        ]
        self.decoder_signature = [
            tf.TensorSpec(shape=(None, None, encoder_model_dimension * 2), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None, mel_channels), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
        ]
        self.debug = debug
        self._apply_all_signatures()

    @property
    def step(self):
        return int(self.optimizer.iterations)

    def _apply_signature(self, function, signature):
        if self.debug:
            return function
        else:
            return tf.function(input_signature=signature)(function)

    def _apply_all_signatures(self):
        self.forward = self._apply_signature(self._forward, self.forward_input_signature)
        self.val_step = self._apply_signature(self._val_step, self.training_input_signature)
        self.forward_encoder = self._apply_signature(self._forward_encoder, self.encoder_signature)
        self.forward_decoder = self._apply_signature(self._forward_decoder, self.decoder_signature)

    def _call_encoder(self, inputs, targets, training):
        padding_mask = create_encoder_padding_mask(inputs)
        text_enc_input = self.text_encoder_prenet(inputs)
        text_enc_output, text_attn_weights = self.text_encoder(text_enc_input,
                                                               training=training,
                                                               padding_mask=padding_mask,
                                                               drop_n_heads=self.drop_n_heads)  # batch x phonemes x dim

        # GST encoder (ref encoder + multihead attention)
        gst_output, gst_attn_weights = self.style_encoder(targets, drop_n_heads=self.drop_n_heads)  # batch x 1 x dim

        # combine embeddings
        gst_output_tile = tf.tile(gst_output, [1, int(tf.shape(text_enc_output)[1]), 1])
        enc_output = tf.concat([text_enc_output, gst_output_tile], 2)  # batch x phonemes x dim*2

        padding_mask = create_mel_padding_mask(enc_output)
        return enc_output, padding_mask, text_attn_weights, gst_attn_weights, gst_output, text_enc_output

    def _call_decoder(self, encoder_output, targets, encoder_padding_mask, training):
        dec_target_padding_mask = create_mel_padding_mask(targets)
        look_ahead_mask = create_look_ahead_mask(tf.shape(targets)[1])
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        dec_input = self.decoder_prenet(targets)
        dec_output, attention_weights = self.decoder(inputs=dec_input,
                                                     enc_output=encoder_output,
                                                     training=training,
                                                     decoder_padding_mask=combined_mask,
                                                     encoder_padding_mask=encoder_padding_mask,
                                                     drop_n_heads=self.drop_n_heads,
                                                     reduction_factor=self.r)
        out_proj = self.final_proj_mel(dec_output)[:, :, :self.r * self.mel_channels]

        b = int(tf.shape(out_proj)[0])
        t = int(tf.shape(out_proj)[1])
        mel = tf.reshape(out_proj, (b, t * self.r, self.mel_channels))
        model_output = self.decoder_postnet(mel, training=training)
        model_output.update(
            {'decoder_attention': attention_weights, 'decoder_output': dec_output, 'linear': mel})
        return model_output

    def _forward(self, inp, output):
        model_out = self.__call__(inputs=inp,
                                  targets=output,
                                  training=False)
        return model_out

    def _forward_encoder(self, inputs, targets):
        return self._call_encoder(inputs, targets, training=False)

    def _forward_decoder(self, encoder_output, targets, encoder_padding_mask):
        return self._call_decoder(encoder_output, targets, encoder_padding_mask, training=False)

    def _gta_forward(self, inp, tar, stop_prob, training):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        tar_stop_prob = stop_prob[:, 1:]

        mel_len = int(tf.shape(tar_inp)[1])
        tar_mel = tar_inp[:, 0::self.r, :]

        # TTS
        model_out = self.__call__(inputs=inp, targets=tar_mel, training=training)
        tts_loss, tts_loss_vals = weighted_sum_losses((tar_real,
                                                       tar_stop_prob,
                                                       tar_real),
                                                      (model_out['final_output'][:, :mel_len, :],
                                                       model_out['stop_prob'][:, :mel_len, :],
                                                       model_out['mel_linear'][:, :mel_len, :]),
                                                      self.loss,
                                                      self.loss_weights)

        model_out.update({'tts_loss': tts_loss})
        model_out.update(
            {'losses': {'output': tts_loss_vals[0], 'stop_prob': tts_loss_vals[1], 'mel_linear': tts_loss_vals[2]}})
        model_out.update({'reduced_target': tar_mel})
        return model_out

    def _val_step(self, inp, tar, stop_prob):
        model_out = self._gta_forward(inp, tar, stop_prob, training=False)
        return model_out

    def _compile(self, stop_scaling, optimizer):
        # only take the first loss into account according to the apple paper
        self.loss_weights = [1., 1., 1.]
        self.compile(loss=[masked_mean_absolute_error,
                           new_scaled_crossentropy(index=2, scaling=stop_scaling),
                           masked_mean_absolute_error],
                     loss_weights=self.loss_weights,
                     optimizer=optimizer)

    def _set_r(self, r):
        if self.r == r:
            return
        self.r = r
        self._apply_all_signatures()

    def _set_heads(self, heads):
        if self.drop_n_heads == heads:
            return
        self.drop_n_heads = heads
        self._apply_all_signatures()

    def call(self, inputs, targets, training):
        encoder_output, padding_mask, text_encoder_attention, gst_encoder_attention, \
        gst_output, text_enc_output = self._call_encoder(inputs, targets, training)
        # decoder
        model_out = self._call_decoder(encoder_output, targets, padding_mask, training)
        model_out.update({'text_encoder_attention': text_encoder_attention,
                          'gst_encoder_attention': gst_encoder_attention,
                          'gst_output': gst_output,
                          'text_enc_output': text_enc_output})
        return model_out

    def predict(self, inp, targets, max_length=1000, encode=True, verbose=True):
        if targets is not None:
            targets = self.encode_ref(targets)
        if encode:
            inp = self.encode_text(inp)
        inp = tf.cast(tf.expand_dims(inp, 0), tf.int32)
        output = tf.cast(tf.expand_dims(self.start_vec, 0), tf.float32)
        output_concat = tf.cast(tf.expand_dims(self.start_vec, 0), tf.float32)
        out_dict = {}
        encoder_output, padding_mask, text_encoder_attention, gst_encoder_attention, _, _ = self.forward_encoder(inp,
                                                                                                                 targets)
        for i in range(int(max_length // self.r) + 1):
            model_out = self.forward_decoder(encoder_output, output, padding_mask)
            output = tf.concat([output, model_out['final_output'][:1, -1:, :]], axis=-2)
            output_concat = tf.concat([tf.cast(output_concat, tf.float32), model_out['final_output'][:1, -self.r:, :]],
                                      axis=-2)
            stop_pred = model_out['stop_prob'][:, -1]
            out_dict = {'mel': output_concat[0, 1:, :],
                        'decoder_attention': model_out['decoder_attention'],
                        'text_encoder_attention': text_encoder_attention,
                        'gst_encoder_attention': gst_encoder_attention}
            if verbose:
                sys.stdout.write(f'\rpred text mel: {i} stop out: {float(stop_pred[0, 2])}')
            if int(tf.argmax(stop_pred, axis=-1)) == self.stop_prob_index:
                if verbose:
                    print('Stopping')
                break
        return out_dict

    def set_constants(self, decoder_prenet_dropout: float = None, learning_rate: float = None,
                      reduction_factor: float = None, drop_n_heads: int = None):
        if decoder_prenet_dropout is not None:
            self.decoder_prenet.rate.assign(decoder_prenet_dropout)
        if learning_rate is not None:
            self.optimizer.lr.assign(learning_rate)
        if reduction_factor is not None:
            self._set_r(reduction_factor)
        if drop_n_heads is not None:
            self._set_heads(drop_n_heads)

    def encode_text(self, text):
        return self.text_pipeline(text)

    def encode_ref(self, targets):
        tar = tf.cast(tf.expand_dims(targets, 0), tf.float32)
        tar_inp = tar[:, :-1]
        tar_mel = tar_inp[:, 0::self.r, :]
        return tar_mel
