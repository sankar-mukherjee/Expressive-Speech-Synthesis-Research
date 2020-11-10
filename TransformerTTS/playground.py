import librosa
import tensorflow as tf
import numpy as np

from model.transformer_utils import create_encoder_padding_mask
from utils.audio import Audio
from utils.config_manager import ConfigManager
from preprocessing.data_handling import load_files, Dataset, DataPrepper
import matplotlib.pyplot as plt
from model.layers import MultiHeadAttention
from utils.losses import l1_loss

np.random.seed(42)
tf.random.set_seed(42)


gst_output = tf.random.uniform([8,1,256])
text_enc_output_j = tf.random.uniform([8,1,256])
joint = tf.matmul(tf.transpose(gst_output, [0, 2, 1]), text_enc_output_j)
B, m1, m2 = tf.shape(joint)
joint = tf.keras.layers.Flatten(joint)


joint = tf.reshape(joint, [-1, m1 * m2])


def shape_list(x):
    """Return list of dims, statically where possible."""
    x = tf.convert_to_tensor(x)

    # If unknown rank, return dynamic shape
    if x.get_shape().dims is None:
        return tf.shape(x)

    static = x.get_shape().as_list()
    shape = tf.shape(x)

    ret = []
    for i in range(len(static)):
        dim = static[i]
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret


config_manager = ConfigManager(config_path='config/wavernn', model_kind='autoregressive',
                               session_name='test')
config = config_manager.config
# get model, prepare data for model, create datasets
model = config_manager.get_model()
config_manager.compile_model(model)
data_prep = DataPrepper(config=config,
                        tokenizer=model.text_pipeline.tokenizer)

train_samples, _ = load_files(metafile=str(config_manager.train_datadir / 'train_metafile.txt'),
                              meldir=str(config_manager.train_datadir / 'mels'),
                              num_samples=config['n_samples'])  # (phonemes, mel)
train_dataset = Dataset(samples=train_samples,
                        preprocessor=data_prep,
                        batch_size=config['batch_size'],
                        mel_channels=config['mel_channels'],
                        shuffle=True)
# _ = train_dataset.next_batch()
mel, phonemes, stop = train_dataset.next_batch()

###################################################################
layer1 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64),
    tf.keras.layers.Activation('relu')
])
layer2 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(8),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dense(64),
    tf.keras.layers.Activation('relu')
])
layer3 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(80),
    tf.keras.layers.Activation('relu')
])
model = tf.keras.models.Sequential([
    layer1,
    layer2,
    layer3
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4))

# TTS model
with tf.GradientTape() as tts_tape:
    predictions = model(mel)
    predictions1 = model(predictions)
    loss = l1_loss(mel, predictions)

tts_gradients = tts_tape.gradient(loss, model.trainable_variables)
model.optimizer.apply_gradients(zip(tts_gradients, model.trainable_variables))




###################################################################
kernel_size = 3
strides = 2
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=kernel_size, strides=strides, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(filters=32, kernel_size=kernel_size, strides=strides, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(filters=64, kernel_size=kernel_size, strides=strides, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(filters=64, kernel_size=kernel_size, strides=strides, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(filters=128, kernel_size=kernel_size, strides=strides, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(filters=128, kernel_size=kernel_size, strides=strides, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu')
])

ref_outputs = tf.expand_dims(mel, axis=-1)  # [8, 846, 80, 1]
predictions = model(ref_outputs)  # [8, 14, 2, 128]
model.summary()
tf.keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)

shapes = shape_list(predictions)
predictions_1 = tf.reshape(predictions, shapes[:-2] + [shapes[2] * shapes[3]])  # [8, 14, 256]
predictions = tf.keras.layers.RNN(tf.keras.layers.GRUCell(128), return_sequences=True)(predictions_1)  # [8, 14, 128]

# rnn_cells = [tf.keras.layers.GRUCell(128) for _ in range(1)]
# stacked_gru = tf.keras.layers.StackedRNNCells([tf.keras.layers.GRUCell(128) for _ in range(1)])
# predictions = tf.keras.layers.RNN(tf.keras.layers.StackedRNNCells([tf.keras.layers.GRUCell(128) for _ in range(1)]),
#                                   return_sequences=True)(predictions_1)  # [8, 14, 128]



reference_state = tf.keras.layers.Dense(128, activation='tanh')(predictions[:, -1, :])  # [8, 128]
reference_state = tf.expand_dims(reference_state, axis=1)
num_gst = 10
style_embed_depth = 256
num_heads = 4
batch_size = 8
# Global style tokens (GST)
initializer = tf.keras.initializers.TruncatedNormal(stddev=0.5)
gst_tokens = tf.Variable(initializer(shape=tf.TensorShape([num_gst, style_embed_depth // num_heads]), dtype=tf.float32))
gst_tokens = tf.tanh(tf.tile(tf.expand_dims(gst_tokens, axis=0), [batch_size, 1, 1]))

# Multi head attention
attn_out, attn_weights = MultiHeadAttention(style_embed_depth, num_heads)(gst_tokens, gst_tokens, reference_state,
                                                                  None, training=True, drop_n_heads=0)
reference_state
plt.imshow(attn_out[:,0,:])
plt.show()

t_vars = [var.name for var in tts_model.trainable_variables]

# *****************************Dip************************************

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv1D

# input shape: (8,256,256,1)
self.conv1 = Conv2D(32, 5, activation='relu')
self.d1 = Dense(128, activation='relu')
self.flatten = Flatten()

# input shape: (8,256,256)
self.conv1 = Conv1D(32, 5, activation='relu')
self.d1 = Dense(128, activation='relu')
self.flatten = Flatten()

# input shape: (8,256,256)
self.flatten = Flatten()
self.d1 = Dense(128, activation='relu')