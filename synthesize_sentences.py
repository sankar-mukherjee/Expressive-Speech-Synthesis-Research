import librosa
import numpy as np
import sys
import torch
import os

sys.path.append('TransformerTTS/')
from utils.config_manager import ConfigManager
from utils.audio import Audio

sys.path.append('WaveRNN/')
from synthesizer_wavernn import Synthesize

if __name__ == "__main__":
    main_dir_path = '/mnt/hdd1/dipjyoti/sankar/Expressive-Speech-Synthesis-Research/'

    tts_model_path = 'TransformerTTS/checkpoints/'
    tts_model_id = 'wavernn_gst_H-mine'

    vocoder_model_path = 'WaveRNN/checkpoints/'
    vocoder_model_id = 'blizzard2013_mol.wavernn/latest_weights.pyt'

    wav_output_path = 'output/' + tts_model_id

    is_style_encoder = False

    # TTS load
    tts_config = ConfigManager(main_dir_path + tts_model_path + tts_model_id, model_kind='autoregressive')
    TTS_model = tts_config.load_model()
    audio = Audio(tts_config.config)

    # Vocoder load
    vocoder_model_path = main_dir_path + vocoder_model_path + vocoder_model_id
    Vocoder_model = Synthesize(model_path=vocoder_model_path)

    # types of prosody
    ref_type = {'sarcasm', 'commanding', 'question'}
    ref_path = '../database/ref_audio/style_audio/'
    # text to be synthesized
    f = open(ref_path + 'test_sentences', "r")
    test_sentence = list(f)
    f.close()

    if not os.path.isdir(wav_output_path):
        os.makedirs(wav_output_path)

    if is_style_encoder:
        for rt in ref_type:
            y, _ = librosa.load(ref_path + rt + '.wav', sr=tts_config.config['sampling_rate'])
            ref_mel = np.transpose(audio.mel_spectrogram(y))
            for i, sen in enumerate(test_sentence):
                u_pred = TTS_model.predict(sen.rstrip(), ref_mel)
                # Normalize for WaveRNN
                mel = (u_pred['mel'].numpy().T + 4.) / 8.
                mel = torch.tensor(mel).unsqueeze(0)
                # Synthesize with vocoder
                out_wav = Vocoder_model.generate(mel=mel)

                out_filename = wav_output_path+'/predicted_' + str(i) + '-' + rt + '.wav'
                librosa.output.write_wav(out_filename, out_wav, tts_config.config['sampling_rate'])
                print('\nSynthesized: ' + out_filename)
    else:
        for i, sen in enumerate(test_sentence):
            u_pred = TTS_model.predict(sen.rstrip())
            # Normalize for WaveRNN
            mel = (u_pred['mel'].numpy().T + 4.) / 8.
            mel = torch.tensor(mel).unsqueeze(0)
            # Synthesize with vocoder
            out_wav = Vocoder_model.generate(mel=mel)

            out_filename = wav_output_path + '/predicted_' + str(i) + '.wav'
            librosa.output.write_wav(out_filename, out_wav, tts_config.config['sampling_rate'])
            print('\nSynthesized: ' + out_filename)
