import numpy as np
import tensorflow as tf

from hparams import hparams
from models import create_model
from text import text_to_sequence


class Synthesizer_embd:
    def __init__(self, teacher_forcing_generating=False, model_name='tacotron'):
        self.teacher_forcing_generating = teacher_forcing_generating
        self.model_name = model_name

    def load(self, checkpoint_path, reference_mel=None):
        print('Constructing model: %s' % self.model_name)
        inputs = tf.placeholder(tf.int32, [1, None], 'inputs')
        input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')
        if reference_mel is not None:
            reference_mel = tf.placeholder(tf.float32, [1, None, hparams.num_mels], 'reference_mel')
        # Only used in teacher-forcing generating mode
        if self.teacher_forcing_generating:
            mel_targets = tf.placeholder(tf.float32, [1, None, hparams.num_mels], 'mel_targets')
        else:
            mel_targets = None

        with tf.variable_scope('model') as scope:
            self.model = create_model(self.model_name, hparams)
            self.model.initialize(inputs, input_lengths, mel_targets=mel_targets, reference_mel=reference_mel)
            self.style_embeddings = self.model.style_embeddings
            self.mel_outputs = self.model.mel_outputs[0]
            self.alignments = self.model.alignments[0]

        print('Loading checkpoint: %s' % checkpoint_path)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.session, checkpoint_path)

    def synthesize_only_embed(self, text, mel_targets=None, reference_mel=None):
        cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
        seq = text_to_sequence(text, cleaner_names)
        feed_dict = {
            self.model.inputs: [np.asarray(seq, dtype=np.int32)],
            self.model.input_lengths: np.asarray([len(seq)], dtype=np.int32),
        }
        if mel_targets is not None:
            mel_targets = np.expand_dims(mel_targets, 0)
            feed_dict.update({self.model.mel_targets: np.asarray(mel_targets, dtype=np.float32)})
        if reference_mel is not None:
            reference_mel = np.expand_dims(reference_mel, 0)
            feed_dict.update({self.model.reference_mel: np.asarray(reference_mel, dtype=np.float32)})

        style_embeddings = self.session.run([self.style_embeddings], feed_dict=feed_dict)

        return style_embeddings

    def synthesize_text(self, text, mel_targets=None, reference_mel=None):
        cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
        seq = text_to_sequence(text, cleaner_names)
        feed_dict = {
            self.model.inputs: [np.asarray(seq, dtype=np.int32)],
            self.model.input_lengths: np.asarray([len(seq)], dtype=np.int32),
        }
        if mel_targets is not None:
            mel_targets = np.expand_dims(mel_targets, 0)
            feed_dict.update({self.model.mel_targets: np.asarray(mel_targets, dtype=np.float32)})
        if reference_mel is not None:
            reference_mel = np.expand_dims(reference_mel, 0)
            feed_dict.update({self.model.reference_mel: np.asarray(reference_mel, dtype=np.float32)})

        mel_outputs, alignments = self.session.run([self.mel_outputs, self.alignments], feed_dict=feed_dict)

        return mel_outputs, alignments
