import librosa
from models.fatchord_version import WaveRNN
import hparams as hp
import torch
from utility import dsp


class Synthesize:

    def __init__(self, model_path: str):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        print('\nInitialising WaveRNN Model...\n')
        # Instantiate WaveRNN Model
        self.voc_model = WaveRNN(rnn_dims=hp.voc_rnn_dims,
                                 fc_dims=hp.voc_fc_dims,
                                 bits=hp.bits,
                                 pad=hp.voc_pad,
                                 upsample_factors=hp.voc_upsample_factors,
                                 feat_dims=hp.num_mels,
                                 compute_dims=hp.voc_compute_dims,
                                 res_out_dims=hp.voc_res_out_dims,
                                 res_blocks=hp.voc_res_blocks,
                                 hop_length=hp.hop_length,
                                 sample_rate=hp.sample_rate,
                                 mode=hp.voc_mode).to(device)
        self.voc_model.restore(model_path)

    def generate(self, mel, batch_pred=True):
        wav_out = self.voc_model.generate(mel, batch_pred, hp.voc_target, hp.voc_overlap, hp.mu_law)
        return wav_out


if __name__ == "__main__":
    ref_wav_path = '../../database/ref_audio/sarcasm.wav'
    y, sr = librosa.load(ref_wav_path, sr=16000)
    mel = dsp.melspectrogram(y)
    mel = torch.tensor(mel).unsqueeze(0)
    vocoder_model_path = 'checkpoints/blizzard2013_mol.wavernn/latest_weights.pyt'
    Vocoder_model = Synthesize(model_path=vocoder_model_path)

    out_wav = Vocoder_model.generate(mel)
    librosa.output.write_wav('output.wav', out_wav, sr)
