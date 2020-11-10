import librosa
import numpy as np
import sys
import torch
import os
import soundfile as sf


if __name__ == "__main__":
    main_dir_path = '/mnt/hdd1/dipjyoti/sankar/Expressive-Speech-Synthesis-Research/'

    tts_type = 'TransformerTTS_H-mine/'
    tts_model_id = 'wavernn_gst_H-mine'

    tts_model_path = tts_type+'checkpoints/' + tts_model_id
    # vocoder
    vocoder_model_path = 'WaveRNN/checkpoints/'
    vocoder_model_id = 'blizzard2013_mol.wavernn/latest_weights.pyt'
    # types of prosody
    ref_path = '../database/ref_audio/for_objective_measurement/'
    # output path
    wav_output_path = 'output/' + tts_model_id + '/for_objective_measurement'


    # append paths
    sys.path.append(tts_type)
    from utils.config_manager import ConfigManager
    from utils.audio import Audio
    sys.path.append('WaveRNN/')
    from synthesizer_wavernn import Synthesize

    # TTS load
    tts_config = ConfigManager(main_dir_path + tts_model_path, model_kind='autoregressive')
    TTS_model = tts_config.load_model()
    audio = Audio(tts_config.config)

    # Vocoder load
    vocoder_model_path = main_dir_path + vocoder_model_path + vocoder_model_id
    Vocoder_model = Synthesize(model_path=vocoder_model_path)

    # text to be synthesized
    f = open(ref_path + 'test_sentences', "r")
    test_sentence = list(f)
    f.close()

    if not os.path.isdir(wav_output_path):
        os.makedirs(wav_output_path)

    for i, sentence in enumerate(test_sentence):
        sen = sentence.split('|')
        y, _ = librosa.load(ref_path + sen[0] + '.wav', sr=tts_config.config['sampling_rate'])
        ref_mel = np.transpose(audio.mel_spectrogram(y))
        u_pred = TTS_model.predict(sen[1].rstrip(), ref_mel)
        # Normalize for WaveRNN
        mel = (u_pred['mel'].numpy().T + 4.) / 8.
        mel = torch.tensor(mel).unsqueeze(0)
        # Synthesize with vocoder
        out_wav = Vocoder_model.generate(mel=mel)

        out_filename = wav_output_path+'/synthesized-' + sen[0] + '.wav'
        sf.write(out_filename, out_wav, tts_config.config['sampling_rate'])
        print('\nSynthesized: ' + out_filename)
