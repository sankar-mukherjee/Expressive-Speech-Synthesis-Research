import glob
import pathlib
import librosa
import numpy as np
import sys
import torch
import os
import soundfile as sf
import tensorflow as tf

if __name__ == "__main__":
    syn_norm = False
    syn_rand = True

    main_dir_path = str(pathlib.Path().absolute()) + '/'

    tts_type = 'S30'
    tts_model_id = 'wavernn_test_mine'
    tts_weights = 'ckpt-40'
    tts_model_path = tts_type + '/checkpoints/' + tts_model_id
    # vocoder
    vocoder_model_path = 'WaveRNN/checkpoints/'
    vocoder_model_id = 'vctk_16k_mol.wavernn/latest_weights.pyt'
    # speaker embedding path
    spk_embed_path = '../database/ref_audio/for_speaker_tts/'
    # types of prosody
    ref_path = '../database/ref_audio/for_speaker_tts/'
    # output path
    wav_output_path = 'output/' + tts_type + '_' + tts_weights + '_' + tts_model_id + '/for_objective_measurement/'

    # append paths
    sys.path.append(tts_type)
    from utils.config_manager import ConfigManager
    from utils.audio import Audio

    sys.path.append('WaveRNN/')
    from synthesizer_wavernn import Synthesize

    # TTS load
    tts_config = ConfigManager(main_dir_path + tts_model_path, model_kind='autoregressive')
    TTS_model = tts_config.load_model(main_dir_path + tts_model_path + '/autoregressive_weights/' + tts_weights)
    audio = Audio(tts_config.config)

    # Vocoder load
    vocoder_model_path = main_dir_path + vocoder_model_path + vocoder_model_id
    Vocoder_model = Synthesize(model_path=vocoder_model_path)

    if syn_norm:
        # text to be synthesized
        f = open(ref_path + 'test_sentences', "r")
        test_sentence = list(f)
        f.close()

        if not os.path.isdir(wav_output_path):
            os.makedirs(wav_output_path)

        for i, sentence in enumerate(test_sentence):
            sen = sentence.split('|')
            y, _ = librosa.load(ref_path + 'style/' + sen[0] + '.wav', sr=tts_config.config['sampling_rate'])
            style_mel = np.transpose(audio.mel_spectrogram(y))

            spk_embed = np.load(ref_path + 'speaker_embedding/' + sen[0] + '.npy')
            spk_embed = tf.expand_dims(spk_embed, 0)
            u_pred = TTS_model.predict(sen[1].rstrip(), style_mel, spk_embed)
            # Normalize for WaveRNN
            mel = (u_pred['mel'].numpy().T + 4.) / 8.
            mel = torch.tensor(mel).unsqueeze(0)
            # Synthesize with vocoder
            out_wav = Vocoder_model.generate(mel=mel)

            out_filename = wav_output_path + 'synthesized-' + sen[0] + '.wav'
            sf.write(out_filename, out_wav, tts_config.config['sampling_rate'])
            print('\nSynthesized: ' + out_filename)

    # random combinations synthesized
    if syn_rand:
        f = open(ref_path + 'test_sentences_rand', "r")
        test_sentence = list(f)
        f.close()
        wav_output_path = 'output/' + tts_type + '_' + tts_weights + '_' \
                          + tts_model_id + '/for_objective_measurement_rand/'

        if not os.path.isdir(wav_output_path):
            os.makedirs(wav_output_path)

        for i, sentence in enumerate(test_sentence):
            sen = sentence.rstrip().split('|')
            y, _ = librosa.load(ref_path + 'style/' + sen[3] + '.wav', sr=tts_config.config['sampling_rate'])
            style_mel = np.transpose(audio.mel_spectrogram(y))

            spk_embed = np.load(ref_path + 'speaker_embedding/' + sen[4] + '.npy')
            spk_embed = tf.expand_dims(spk_embed, 0)
            u_pred = TTS_model.predict(sen[1].rstrip(), style_mel, spk_embed)
            # Normalize for WaveRNN
            mel = (u_pred['mel'].numpy().T + 4.) / 8.
            mel = torch.tensor(mel).unsqueeze(0)
            # Synthesize with vocoder
            out_wav = Vocoder_model.generate(mel=mel)

            out_filename = wav_output_path + 'text-' + sen[0] + ':style-' + sen[3] + ':speaker-' + sen[4] + '.wav'
            sf.write(out_filename, out_wav, tts_config.config['sampling_rate'])
            print('\nSynthesized: ' + out_filename)
