import tensorflow as tf

from model.transformer_utils import positional_encoding, scaled_dot_product_attention, shape_list, \
    create_mel_padding_mask


class CNNResNorm(tf.keras.layers.Layer):
    def __init__(self,
                 out_size: int,
                 n_layers: int,
                 hidden_size: int,
                 kernel_size: int,
                 inner_activation: str,
                 last_activation: str,
                 padding: str,
                 normalization: str,
                 **kwargs):
        super(CNNResNorm, self).__init__(**kwargs)
        self.convolutions = [tf.keras.layers.Conv1D(filters=hidden_size,
                                                    kernel_size=kernel_size,
                                                    padding=padding)
                             for _ in range(n_layers - 1)]
        self.inner_activations = [tf.keras.layers.Activation(inner_activation) for _ in range(n_layers - 1)]
        self.last_conv = tf.keras.layers.Conv1D(filters=out_size,
                                                kernel_size=kernel_size,
                                                padding=padding)
        self.last_activation = tf.keras.layers.Activation(last_activation)
        if normalization == 'layer':
            self.normalization = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(n_layers + 1)]
        elif normalization == 'batch':
            self.normalization = [tf.keras.layers.BatchNormalization() for _ in range(n_layers + 1)]
        else:
            assert False is True, f'normalization must be either "layer" or "batch", not {normalization}.'

    def call_convs(self, x, training):
        for i in range(0, len(self.convolutions)):
            x = self.convolutions[i](x)
            x = self.normalization[i](x, training=training)
            x = self.inner_activations[i](x)
        return x

    def call(self, inputs, training):
        x = self.call_convs(inputs, training=training)
        x = self.last_conv(x)
        x = self.normalization[-2](x, training=training)
        x = self.last_activation(x)
        return self.normalization[-1](inputs + x)


class FFNResNorm(tf.keras.layers.Layer):

    def __init__(self,
                 model_dim: int,
                 dense_hidden_units: int,
                 dropout_rate: float,
                 **kwargs):
        super(FFNResNorm, self).__init__(**kwargs)
        self.d1 = tf.keras.layers.Dense(dense_hidden_units)
        self.activation = tf.keras.layers.Activation('relu')
        self.d2 = tf.keras.layers.Dense(model_dim)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.last_ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training):
        ffn_out = self.d1(x)
        ffn_out = self.d2(ffn_out)  # (batch_size, input_seq_len, model_dim)
        ffn_out = self.ln(ffn_out)  # (batch_size, input_seq_len, model_dim)
        ffn_out = self.activation(ffn_out)
        ffn_out = self.dropout(ffn_out, training=training)
        return self.last_ln(ffn_out + x)


class HeadDrop(tf.keras.layers.Layer):
    """ Randomly drop n heads. """

    def __init__(self, **kwargs):
        super(HeadDrop, self).__init__(**kwargs)

    def call(self, batch, training: bool, drop_n_heads: int):
        if not training or (drop_n_heads == 0):
            return batch
        if len(tf.shape(batch)) != 4:
            raise Exception('attention values must be 4 dimensional')
        batch_size = tf.shape(batch)[0]
        head_n = tf.shape(batch)[1]
        if head_n == 1:
            return batch
        # assert drop_n_heads < head_n, 'drop_n_heads must less than number of heads'
        keep_head_batch = tf.TensorArray(tf.float32, size=batch_size)
        keep_mask = tf.concat([tf.ones(head_n - drop_n_heads), tf.zeros(drop_n_heads)], axis=0)
        for i in range(batch_size):
            t = tf.random.shuffle(keep_mask)
            keep_head_batch = keep_head_batch.write(i, t)
        keep_head_batch = keep_head_batch.stack()
        keep_head_batch = keep_head_batch[:, :, tf.newaxis, tf.newaxis]
        return batch * keep_head_batch * tf.cast(head_n / (head_n - drop_n_heads), tf.float32)


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, model_dim: int, num_heads: int, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.head_drop = HeadDrop()

        assert model_dim % self.num_heads == 0

        self.depth = model_dim // self.num_heads

        self.wq = tf.keras.layers.Dense(model_dim)
        self.wk = tf.keras.layers.Dense(model_dim)
        self.wv = tf.keras.layers.Dense(model_dim)

        self.dense = tf.keras.layers.Dense(model_dim)

    def split_heads(self, x, batch_size: int):
        """ Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """

        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q_in, mask, training, drop_n_heads):
        batch_size = tf.shape(q_in)[0]

        q = self.wq(q_in)  # (batch_size, seq_len, model_dim)
        k = self.wk(k)  # (batch_size, seq_len, model_dim)
        v = self.wv(v)  # (batch_size, seq_len, model_dim)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = self.head_drop(scaled_attention, training=training, drop_n_heads=drop_n_heads)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.model_dim))  # (batch_size, seq_len_q, model_dim)
        concat_query = tf.concat([q_in, concat_attention], axis=-1)
        output = self.dense(concat_query)  # (batch_size, seq_len_q, model_dim)

        return output, attention_weights


class SelfAttentionResNorm(tf.keras.layers.Layer):

    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 dropout_rate: float,
                 **kwargs):
        super(SelfAttentionResNorm, self).__init__(**kwargs)
        self.mha = MultiHeadAttention(model_dim, num_heads)
        self.ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.last_ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training, mask, drop_n_heads):
        attn_out, attn_weights = self.mha(x, x, x, mask, training=training,
                                          drop_n_heads=drop_n_heads)  # (batch_size, input_seq_len, model_dim)
        attn_out = self.ln(attn_out)  # (batch_size, input_seq_len, model_dim)
        out = self.dropout(attn_out, training=training)
        return self.last_ln(out + x), attn_weights


class SelfAttentionDenseBlock(tf.keras.layers.Layer):

    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 dense_hidden_units: int,
                 dropout_rate: float,
                 **kwargs):
        super(SelfAttentionDenseBlock, self).__init__(**kwargs)
        self.sarn = SelfAttentionResNorm(model_dim, num_heads, dropout_rate=dropout_rate)
        self.ffn = FFNResNorm(model_dim, dense_hidden_units, dropout_rate=dropout_rate)

    def call(self, x, training, mask, drop_n_heads):
        attn_out, attn_weights = self.sarn(x, mask=mask, training=training, drop_n_heads=drop_n_heads)
        return self.ffn(attn_out, training=training), attn_weights


class SelfAttentionConvBlock(tf.keras.layers.Layer):

    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 dropout_rate: float,
                 conv_filters: int,
                 kernel_size: int,
                 conv_activation: str,
                 **kwargs):
        super(SelfAttentionConvBlock, self).__init__(**kwargs)
        self.sarn = SelfAttentionResNorm(model_dim, num_heads, dropout_rate=dropout_rate)
        self.conv = CNNResNorm(out_size=model_dim,
                               n_layers=2,
                               hidden_size=conv_filters,
                               kernel_size=kernel_size,
                               inner_activation=conv_activation,
                               last_activation=conv_activation,
                               padding='same',
                               normalization='batch')

    def call(self, x, training, mask, drop_n_heads):
        attn_out, attn_weights = self.sarn(x, mask=mask, training=training, drop_n_heads=drop_n_heads)
        conv = self.conv(attn_out)
        return conv, attn_weights


class SelfAttentionBlocks(tf.keras.layers.Layer):
    def __init__(self,
                 model_dim: int,
                 feed_forward_dimension: int,
                 num_heads: list,
                 maximum_position_encoding: int,
                 conv_filters: int,
                 dropout_rate: float,
                 dense_blocks: int,
                 kernel_size: int,
                 conv_activation: str,
                 **kwargs):
        super(SelfAttentionBlocks, self).__init__(**kwargs)
        self.model_dim = model_dim
        self.pos_encoding_scalar = tf.Variable(1., trainable=False)
        self.pos_encoding = positional_encoding(maximum_position_encoding, model_dim)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.encoder_SADB = [
            SelfAttentionDenseBlock(model_dim=model_dim, dropout_rate=dropout_rate, num_heads=n_heads,
                                    dense_hidden_units=feed_forward_dimension, name=f'{self.name}_SADB_{i}')
            for i, n_heads in enumerate(num_heads[:dense_blocks])]
        self.encoder_SACB = [
            SelfAttentionConvBlock(model_dim=model_dim, dropout_rate=dropout_rate, num_heads=n_heads,
                                   name=f'{self.name}_SACB_{i}', kernel_size=kernel_size,
                                   conv_activation=conv_activation, conv_filters=conv_filters)
            for i, n_heads in enumerate(num_heads[dense_blocks:])]

    def call(self, inputs, training, padding_mask, drop_n_heads, reduction_factor=1):
        seq_len = tf.shape(inputs)[1]
        x = inputs * tf.math.sqrt(tf.cast(self.model_dim, tf.float32))
        x += self.pos_encoding_scalar * self.pos_encoding[:, :seq_len * reduction_factor:reduction_factor, :]
        x = self.dropout(x, training=training)
        attention_weights = {}
        for i, block in enumerate(self.encoder_SADB):
            x, attn_weights = block(x, training=training, mask=padding_mask, drop_n_heads=drop_n_heads)
            attention_weights[f'{self.name}_DenseBlock{i + 1}_SelfAttention'] = attn_weights
        for i, block in enumerate(self.encoder_SACB):
            x, attn_weights = block(x, training=training, mask=padding_mask, drop_n_heads=drop_n_heads)
            attention_weights[f'{self.name}_ConvBlock{i + 1}_SelfAttention'] = attn_weights

        return x, attention_weights


class CrossAttentionResnorm(tf.keras.layers.Layer):

    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 dropout_rate: float,
                 **kwargs):
        super(CrossAttentionResnorm, self).__init__(**kwargs)
        self.mha = MultiHeadAttention(model_dim, num_heads)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, q, k, v, training, mask, drop_n_heads):
        attn_values, attn_weights = self.mha(v, k=k, q_in=q, mask=mask, training=training, drop_n_heads=drop_n_heads)
        attn_values = self.dropout(attn_values, training=training)
        out = self.layernorm(attn_values + q)
        return out, attn_weights


class CrossAttentionDenseBlock(tf.keras.layers.Layer):

    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 dense_hidden_units: int,
                 dropout_rate: float,
                 **kwargs):
        super(CrossAttentionDenseBlock, self).__init__(**kwargs)
        self.sarn = SelfAttentionResNorm(model_dim, num_heads, dropout_rate=dropout_rate)
        self.carn = CrossAttentionResnorm(model_dim, num_heads, dropout_rate=dropout_rate)
        self.ffn = FFNResNorm(model_dim, dense_hidden_units, dropout_rate=dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask, drop_n_heads):
        attn1, attn_weights_block1 = self.sarn(x, mask=look_ahead_mask, training=training, drop_n_heads=drop_n_heads)

        attn2, attn_weights_block2 = self.carn(attn1, v=enc_output, k=enc_output,
                                               mask=padding_mask, training=training, drop_n_heads=drop_n_heads)
        ffn_out = self.ffn(attn2, training=training)
        return ffn_out, attn_weights_block1, attn_weights_block2


class CrossAttentionConvBlock(tf.keras.layers.Layer):

    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 conv_filters: int,
                 dropout_rate: float,
                 kernel_size: int,
                 conv_padding: str,
                 conv_activation: str,
                 **kwargs):
        super(CrossAttentionConvBlock, self).__init__(**kwargs)
        self.sarn = SelfAttentionResNorm(model_dim, num_heads, dropout_rate=dropout_rate)
        self.carn = CrossAttentionResnorm(model_dim, num_heads, dropout_rate=dropout_rate)
        self.conv = CNNResNorm(out_size=model_dim,
                               n_layers=2,
                               hidden_size=conv_filters,
                               kernel_size=kernel_size,
                               inner_activation=conv_activation,
                               last_activation=conv_activation,
                               padding=conv_padding,
                               normalization='batch')

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask, drop_n_heads):
        attn1, attn_weights_block1 = self.sarn(x, mask=look_ahead_mask, training=training, drop_n_heads=drop_n_heads)

        attn2, attn_weights_block2 = self.carn(attn1, v=enc_output, k=enc_output,
                                               mask=padding_mask, training=training, drop_n_heads=drop_n_heads)
        ffn_out = self.conv(attn2, training=training)
        return ffn_out, attn_weights_block1, attn_weights_block2


class CrossAttentionBlocks(tf.keras.layers.Layer):

    def __init__(self,
                 model_dim: int,
                 feed_forward_dimension: int,
                 num_heads: list,
                 maximum_position_encoding: int,
                 dropout_rate: float,
                 dense_blocks: int,
                 conv_filters: int,
                 conv_activation: str,
                 conv_padding: str,
                 conv_kernel: int,
                 **kwargs):
        super(CrossAttentionBlocks, self).__init__(**kwargs)
        self.model_dim = model_dim
        self.pos_encoding_scalar = tf.Variable(1., trainable=False)
        self.pos_encoding = positional_encoding(maximum_position_encoding, model_dim)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.CADB = [
            CrossAttentionDenseBlock(model_dim=model_dim, dropout_rate=dropout_rate, num_heads=n_heads,
                                     dense_hidden_units=feed_forward_dimension, name=f'{self.name}_CADB_{i}')
            for i, n_heads in enumerate(num_heads[:dense_blocks])]
        self.CACB = [
            CrossAttentionConvBlock(model_dim=model_dim, dropout_rate=dropout_rate, num_heads=n_heads,
                                    name=f'{self.name}_CACB_{i}', conv_filters=conv_filters,
                                    conv_activation=conv_activation, conv_padding=conv_padding, kernel_size=conv_kernel)
            for i, n_heads in enumerate(num_heads[dense_blocks:])]

    def call(self, inputs, enc_output, training, decoder_padding_mask, encoder_padding_mask, drop_n_heads,
             reduction_factor=1):
        seq_len = tf.shape(inputs)[1]
        x = inputs * tf.math.sqrt(tf.cast(self.model_dim, tf.float32))
        x += self.pos_encoding_scalar * self.pos_encoding[:, :seq_len * reduction_factor:reduction_factor, :]
        x = self.dropout(x, training=training)
        attention_weights = {}
        for i, block in enumerate(self.CADB):
            x, _, attn_weights = block(x, enc_output, training, decoder_padding_mask, encoder_padding_mask,
                                       drop_n_heads)
            attention_weights[f'{self.name}_DenseBlock{i + 1}_CrossAttention'] = attn_weights
        for i, block in enumerate(self.CACB):
            x, _, attn_weights = block(x, enc_output, training, decoder_padding_mask, encoder_padding_mask,
                                       drop_n_heads)
            attention_weights[f'{self.name}_ConvBlock{i + 1}_CrossAttention'] = attn_weights

        return x, attention_weights


class DecoderPrenet(tf.keras.layers.Layer):

    def __init__(self,
                 model_dim: int,
                 dense_hidden_units: int,
                 dropout_rate: float,
                 **kwargs):
        super(DecoderPrenet, self).__init__(**kwargs)
        self.d1 = tf.keras.layers.Dense(dense_hidden_units,
                                        activation='relu')  # (batch_size, seq_len, dense_hidden_units)
        self.d2 = tf.keras.layers.Dense(model_dim, activation='relu')  # (batch_size, seq_len, model_dim)
        self.rate = tf.Variable(dropout_rate, trainable=False)
        self.dropout_1 = tf.keras.layers.Dropout(self.rate)
        self.dropout_2 = tf.keras.layers.Dropout(self.rate)

    def call(self, x):
        self.dropout_1.rate = self.rate
        self.dropout_2.rate = self.rate
        x = self.d1(x)
        # use dropout also in inference for positional encoding relevance
        x = self.dropout_1(x, training=True)
        x = self.d2(x)
        x = self.dropout_2(x, training=True)
        return x


class Linear(tf.keras.layers.Layer):
    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer="random_normal", trainable=True, name='Linear_w')
        self.b = self.add_weight(shape=(self.units,), initializer="random_normal", trainable=True, name='Linear_b')

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class LinearQ(tf.keras.layers.Layer):
    def __init__(self):
        super(LinearQ, self).__init__()

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[1], input_shape[1]), initializer="random_normal", trainable=True, name='LinearQ_w')
        self.b = self.add_weight(shape=(input_shape[1],1), initializer="random_normal", trainable=True, name='LinearQ_b')

    def call(self, inputs):
        l_term = tf.matmul(inputs, self.b)
        tmp_term = tf.matmul(inputs, self.w)
        q_term = tf.reduce_sum(tf.multiply(inputs, tmp_term), 1, keepdims=True)
        out = l_term + q_term
        return out


class MineNetLinearQ(tf.keras.layers.Layer):
    def __init__(self, dense_hidden_units: list, **kwargs):
        super(MineNetLinearQ, self).__init__(**kwargs)
        self.fcs = [Linear(f) for f in dense_hidden_units]
        self.fc_q = LinearQ()
        self.inner_activations = [tf.keras.layers.Activation('relu') for _ in range(len(dense_hidden_units))]
        self.fc_proj = Linear(1)

    def call(self, x):
        x = tf.squeeze(x, axis=1)
        x_q = self.fc_q(x)
        for i in range(0, len(self.fcs)):
            x = self.fcs[i](x)
            x = self.inner_activations[i](x)
        x = self.fc_proj(x)
        x = x + x_q
        x = tf.expand_dims(x, axis=1)
        return x


class MineNetLinear(tf.keras.layers.Layer):
    def __init__(self, dense_hidden_units: list, **kwargs):
        super(MineNetLinear, self).__init__(**kwargs)
        self.fcs = [Linear(f) for f in dense_hidden_units]
        self.inner_activations = [tf.keras.layers.Activation('relu') for _ in range(len(dense_hidden_units))]
        self.fc_proj = Linear(1)

    def call(self, x):
        x = tf.squeeze(x, axis=1)
        for i in range(0, len(self.fcs)):
            x = self.fcs[i](x)
            x = self.inner_activations[i](x)
        x = self.fc_proj(x)
        x = tf.expand_dims(x, axis=1)
        return x


class MineNetFirstOrder(tf.keras.layers.Layer):
    def __init__(self, dense_hidden_units: list, **kwargs):
        super(MineNetFirstOrder, self).__init__(**kwargs)
        self.fcs = [tf.keras.layers.Dense(f) for f in dense_hidden_units]
        self.inner_activations = [tf.keras.layers.Activation('relu') for _ in range(len(dense_hidden_units))]
        self.fc_proj = tf.keras.layers.Dense(1)

    def call(self, x):
        for i in range(0, len(self.fcs)):
            x = self.fcs[i](x)
            x = self.inner_activations[i](x)
        x = self.fc_proj(x)
        return x


class MineNetSecondOrder(tf.keras.layers.Layer):
    def __init__(self, filters: list, kernel_size: int, dense_hidden_units: list, **kwargs):
        super(MineNetSecondOrder, self).__init__(**kwargs)
        self.convs = [tf.keras.layers.Conv1D(filters=f, kernel_size=kernel_size, activation='relu') for f in filters]
        self.flatten = tf.keras.layers.Flatten()

        self.fcs = [tf.keras.layers.Dense(f) for f in dense_hidden_units]
        self.inner_activations = [tf.keras.layers.Activation('relu') for _ in range(len(dense_hidden_units))]
        self.fc_proj = tf.keras.layers.Dense(1)

    def call(self, x):
        for i in range(0, len(self.convs)):
            x = self.convs[i](x)
        x = self.flatten(x)
        for i in range(0, len(self.fcs)):
            x = self.fcs[i](x)
            x = self.inner_activations[i](x)
        x = self.fc_proj(x)
        return x


class CLUBNet(tf.keras.layers.Layer):
    def __init__(self, dense_hidden_units: list, log_var: bool, **kwargs):
        super(CLUBNet, self).__init__(**kwargs)
        self.fcs = [tf.keras.layers.Dense(f) for f in dense_hidden_units]
        self.inner_activations = [tf.keras.layers.Activation('relu') for _ in range(len(dense_hidden_units))]
        self.fc_proj = tf.keras.layers.Dense(256)
        self.tanh = tf.keras.layers.Activation('tanh')
        self.log_var = log_var

    def call(self, x):
        for i in range(0, len(self.fcs)):
            x = self.fcs[i](x)
            x = self.inner_activations[i](x)
        x = self.fc_proj(x)
        if self.log_var:
            x = self.tanh(x)
        return x


class ReferenceEncoderGST(tf.keras.layers.Layer):
    def __init__(self,
                 kernel_size: int,
                 strides: int,
                 conv_filters: list,
                 gru_cell_units: int,
                 gst_style_embed_dim: int,
                 multi_num_heads: int,
                 gst_heads: int,
                 **kwargs):
        super(ReferenceEncoderGST, self).__init__(**kwargs)

        self.convolutions = [tf.keras.layers.Conv2D(filters=f, kernel_size=kernel_size,
                                                    strides=strides, padding='same')
                             for f in conv_filters]
        self.inner_activations = [tf.keras.layers.Activation('relu') for _ in range(len(conv_filters))]
        self.normalization = [tf.keras.layers.BatchNormalization() for _ in range(len(conv_filters))]
        self.rnn_net = tf.keras.layers.RNN(tf.keras.layers.GRUCell(gru_cell_units), return_sequences=True)

        self.rnn_proj_net = tf.keras.layers.Dense(gru_cell_units, activation='tanh')
        self.mha = MultiHeadAttention(gst_style_embed_dim, multi_num_heads)

        # Global style tokens (GST)
        self.initializer = tf.keras.initializers.TruncatedNormal(stddev=0.5)
        self.gst_tokens = tf.Variable(
            self.initializer(shape=tf.TensorShape([gst_heads, gst_style_embed_dim // multi_num_heads]),
                             dtype=tf.float32), trainable=True, name='GST_token')

    def call_convs(self, x):
        for i in range(0, len(self.convolutions)):
            x = self.convolutions[i](x)
            x = self.normalization[i](x)
            x = self.inner_activations[i](x)
        return x

    def call(self, x, drop_n_heads, training: bool):
        batch_size = tf.shape(x)[0]
        x = tf.expand_dims(x, axis=-1)
        # CNN
        conv_out = self.call_convs(x)
        shapes = shape_list(conv_out)
        conv_out = tf.reshape(conv_out, shapes[:-2] + [shapes[2] * shapes[3]])
        # RNN
        rnn_out = self.rnn_net(conv_out)
        rnn_proj = self.rnn_proj_net(rnn_out[:, -1, :])  # take the last prediction of rnn from gst paper
        rnn_proj = tf.expand_dims(rnn_proj, axis=1)

        # freeze/unfreeze gst tokens weights
        self.gst_tokens.training = training
        # reshape GST tokens
        temp_gst_tokens = tf.tanh(tf.tile(tf.expand_dims(self.gst_tokens, axis=0), [batch_size, 1, 1]))

        # Multi head attention
        enc_out, attention_weights = self.mha(temp_gst_tokens, temp_gst_tokens, rnn_proj, None, training=training,
                                              drop_n_heads=drop_n_heads)

        attention_weights = {f'{self.name}_attention': attention_weights}
        gst_tokens = {'GST_tokens': self.gst_tokens.numpy()}

        return enc_out, attention_weights, gst_tokens


class Postnet(tf.keras.layers.Layer):

    def __init__(self, mel_channels: int,
                 conv_filters: int,
                 conv_layers: int,
                 kernel_size: int,
                 **kwargs):
        super(Postnet, self).__init__(**kwargs)
        self.mel_channels = mel_channels
        self.stop_linear = tf.keras.layers.Dense(3)
        self.conv_blocks = CNNResNorm(out_size=mel_channels,
                                      kernel_size=kernel_size,
                                      padding='causal',
                                      inner_activation='tanh',
                                      last_activation='linear',
                                      hidden_size=conv_filters,
                                      n_layers=conv_layers,
                                      normalization='batch')
        self.add_layer = tf.keras.layers.Add()

    def call(self, x, training):
        stop = self.stop_linear(x)
        conv_out = self.conv_blocks(x, training=training)
        return {
            'mel_linear': x,
            'final_output': conv_out,
            'stop_prob': stop,
        }


class DurationPredictor(tf.keras.layers.Layer):
    def __init__(self,
                 model_dim: int,
                 kernel_size: int,
                 conv_padding: str,
                 conv_activation: str,
                 conv_block_n: int,
                 dense_activation: str,
                 **kwargs):
        super(DurationPredictor, self).__init__(**kwargs)
        self.conv_blocks = CNNResNorm(out_size=model_dim,
                                      kernel_size=kernel_size,
                                      padding=conv_padding,
                                      inner_activation=conv_activation,
                                      last_activation=conv_activation,
                                      hidden_size=model_dim,
                                      n_layers=conv_block_n,
                                      normalization='layer')
        self.linear = tf.keras.layers.Dense(1, activation=dense_activation,
                                            bias_initializer=tf.keras.initializers.Constant(value=1))

    def call(self, x, training):
        x = self.conv_blocks(x, training=training)
        x = self.linear(x)
        return x


class Expand(tf.keras.layers.Layer):
    """ Expands a 3D tensor on its second axis given a list of dimensions.
        Tensor should be:
            batch_size, seq_len, dimension

        E.g:
        input = tf.Tensor([[[0.54710746 0.8943467 ]
                          [0.7140938  0.97968304]
                          [0.5347662  0.15213418]]], shape=(1, 3, 2), dtype=float32)
        dimensions = tf.Tensor([1 3 2], shape=(3,), dtype=int32)
        output = tf.Tensor([[[0.54710746 0.8943467 ]
                           [0.7140938  0.97968304]
                           [0.7140938  0.97968304]
                           [0.7140938  0.97968304]
                           [0.5347662  0.15213418]
                           [0.5347662  0.15213418]]], shape=(1, 6, 2), dtype=float32)
    """

    def __init__(self, model_dim, **kwargs):
        super(Expand, self).__init__(**kwargs)
        self.model_dimension = model_dim

    def call(self, x, dimensions):
        dimensions = tf.squeeze(dimensions, axis=-1)
        dimensions = tf.cast(tf.math.round(dimensions), tf.int32)
        seq_len = tf.shape(x)[1]
        batch_size = tf.shape(x)[0]
        # build masks from dimensions
        max_dim = tf.math.reduce_max(dimensions)
        tot_dim = tf.math.reduce_sum(dimensions)
        index_masks = tf.RaggedTensor.from_row_lengths(tf.ones(tot_dim), tf.reshape(dimensions, [-1])).to_tensor()
        index_masks = tf.cast(tf.reshape(index_masks, (batch_size, seq_len * max_dim)), tf.float32)
        non_zeros = seq_len * max_dim - tf.reduce_sum(max_dim - dimensions, axis=1)
        # stack and mask
        tiled = tf.tile(x, [1, 1, max_dim])
        reshaped = tf.reshape(tiled, (batch_size, seq_len * max_dim, self.model_dimension))
        mask_reshape = tf.multiply(reshaped, index_masks[:, :, tf.newaxis])
        ragged = tf.RaggedTensor.from_row_lengths(mask_reshape[index_masks > 0], non_zeros)
        return ragged.to_tensor()
