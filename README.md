# Expressive-Speech-Synthesis-Research

####Contains:
___

    gst-tacotron
    transformerTTS
    gst-transformerTTS
    gst-transformerTTS-mine
    universal TTS
___
Generate sentences:
 
    synthesize_sentences_speaker.py
    
synthesize_sentences.py uses trained tts and vocoder model and generate synthesized sentences.

Put the reference wav files in to the folder:
    
    ../database/ref_audio/

Rename the reference wav files (if you want) according to types of prosody
    
    ref_type = {'sarcasm', 'question', 'command', ...}
    
Put the test sentences into

    ../database/ref_audio/test_sentences
    
Change TTS model and/or vocoder model inside test.py