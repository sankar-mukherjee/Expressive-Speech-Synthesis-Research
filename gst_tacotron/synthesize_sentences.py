import argparse
import os
import re
import numpy as np
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer
from util import audio
import librosa
import torch

def get_output_base_path(checkpoint_path):
    base_dir = os.path.dirname(checkpoint_path)
    m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
    name = 'eval-%d' % int(m.group(1)) if m else 'eval'
    return os.path.join(base_dir, name)


def run_eval(args):
    print(hparams_debug_string())
    is_teacher_force = False
    mel_targets = args.mel_targets
    reference_mel = None
    if args.mel_targets is not None:
        is_teacher_force = True
        mel_targets = np.load(args.mel_targets)
    synth = Synthesizer(teacher_forcing_generating=is_teacher_force)
    synth.load(args.checkpoint, args.reference_audio)
    base_path = get_output_base_path(args.checkpoint)

    ref_wav = audio.load_wav(args.reference_audio)
    reference_mel = audio.melspectrogram(ref_wav).astype(np.float32).T
    path = '%s_ref-%s.wav' % (base_path, os.path.splitext(os.path.basename(args.reference_audio))[0])
    alignment_path = '%s_ref-%s-align.png' % (
    base_path, os.path.splitext(os.path.basename(args.reference_audio))[0])
    synth.synthesize(args.text, mel_targets=mel_targets, reference_mel=reference_mel, alignment_path=alignment_path)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--text', required=True, default=None, help='Single test text sentence')
    parser.add_argument('--hparams', default='',
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--reference_audio', default=None, help='Reference audio path')
    parser.add_argument('--mel_targets', default=None, help='Mel-targets path, used when use teacher_force generation')
    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    hparams.parse(args.hparams)
    run_eval(args)


if __name__ == '__main__':
    checkpoint = 'logs-tacotron/gst_tacotron/model.ckpt-143000'


    wav_output_path = 'output/wavernn_gst_tacotron'

    # types of prosody
    ref_type = {'sarcasm', 'commanding', 'question'}
    ref_path = '../database/ref_audio/'
    # text to be synthesized
    f = open(ref_path + 'test_sentences', "r")
    test_sentence = list(f)
    f.close()

    if not os.path.isdir(wav_output_path):
        os.makedirs(wav_output_path)

    synth = Synthesizer(teacher_forcing_generating=False)
    synth.load(checkpoint, reference_audio)

    for rt in ref_type:
        y, _ = librosa.load(ref_path + rt + '.wav', sr=hparams.sample_rate)
        ref_mel = np.transpose(audio.mel_spectrogram(y))
        for i, sen in enumerate(test_sentence):
            u_pred = synth.synthesize(sen.rstrip(), mel_targets=None, reference_mel=ref_mel,alignment_path=alignment_path)
            # Normalize for WaveRNN
            mel = (u_pred['mel'].numpy().T + 4.) / 8.
            mel = torch.tensor(mel).unsqueeze(0)
            # Synthesize with vocoder
            out_wav = Vocoder_model.generate(mel=mel)

            out_filename = wav_output_path + '/predicted_' + str(i) + '-' + rt + '.wav'
            librosa.output.write_wav(out_filename, out_wav, hparams.sample_rate)
            print('\nSynthesized: ' + out_filename)
