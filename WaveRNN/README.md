# WaveRNN
WaveRNN Vocoder implementation of https://github.com/fatchord/WaveRNN

run training

    python train_wavernn.py

synthesize with pretrained model

    synthesizer_wavernn.py

to call synthesize with pretrained model

    Vocoder_model = Synthesize(model_path=vocoder_model_path)    
    predicted_mel = tts_model(...) # no_mels x frames: 80 x 349

    mel = (predicted_mel.numpy().T + 4.) / 8.
    mel = torch.tensor(mel).unsqueeze(0) # 1 x 80 x 349
    out_wav = Vocoder_model.generate(mel=predicted_mel)