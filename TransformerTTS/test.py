from utils.config_manager import ConfigManager
from utils.audio import Audio
import librosa
import numpy as np
import tensorflow as tf

config_loader = ConfigManager('/mnt/hdd1/dipjyoti/sankar/Expressive-Speech-Synthesis-Research/TransformerTTS/'
                              'checkpoints/wavernn_gst', model_kind='autoregressive')
ref_wav_path = '../../database/ref_audio/sarcasm.wav'
audio = Audio(config_loader.config)
y, sr = librosa.load(ref_wav_path, sr=config_loader.config['sampling_rate'])
mel = np.transpose(audio.mel_spectrogram(y))
model = config_loader.load_model()
out = model.predict('President Trump met with other leaders at the Group of twenty conference.', mel)
np.save('output.npy', out)

# Convert spectrogram to wav (with griffin lim)
wav = audio.reconstruct_waveform(out['mel'].numpy().T)
librosa.output.write_wav('output.wav', wav, 16000)
