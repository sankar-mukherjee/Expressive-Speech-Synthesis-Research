from utils.config_manager import ConfigManager
from utils.audio import Audio
import librosa
import numpy as np

config_loader = ConfigManager('/mnt/hdd1/dipjyoti/TransformerTTS/checkpoints/wavernn_gst', model_kind='autoregressive')
audio = Audio(config_loader.config)
model = config_loader.load_model()
out = model.predict('President Trump met with other leaders at the Group of twenty conference.')
np.save('output.npy', out)

# Convert spectrogram to wav (with griffin lim)
wav = audio.reconstruct_waveform(out['mel'].numpy().T)
librosa.output.write_wav('output.wav', wav, 16000)
