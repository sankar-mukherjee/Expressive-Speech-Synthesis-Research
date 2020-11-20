import numpy as np
from synthesizer_only_embd import Synthesizer_embd
from util import audio

if __name__ == '__main__':
    checkpoint = 'logs-tacotron/gst_tacotron/model.ckpt-143000'
    synth = Synthesizer_embd(teacher_forcing_generating=False, model_name='tacotron')
    dir_path = '../../database/blizzard2013/segmented/'
    out_text_file_object = open(dir_path+'metadata_wavn_small.train', 'r')
    lines = list(out_text_file_object)
    out_text_file_object.close()

    tmp = lines[0].split('|')
    ref_wav = audio.load_wav(dir_path+'wavn_small/'+tmp[0].rstrip()+'.wav')
    reference_mel = audio.melspectrogram(ref_wav).astype(np.float32).T
    synth.load(checkpoint, reference_mel)

    for line in lines:
        tmp = line.split('|')
        text = tmp[1].rstrip()
        ref_wav = audio.load_wav(dir_path + 'wavn_small/' + tmp[0].rstrip() + '.wav')
        reference_mel = audio.melspectrogram(ref_wav).astype(np.float32).T

        style_embeddings = synth.synthesize_only_embed(text, mel_targets=None, reference_mel=reference_mel)
        np.save('style_embeddings/' + tmp[0] + '.npy', style_embeddings)
