
# Universal TTS
A TTS system where we encode speaker, style and text independently resulting three different embeddings. Form those
embeddings we decode and synthesized speech. 

Additionally, Style, text and speaker was disentangled using Mutual Information. 

**Style Encoder**: [Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End 
Speech Synthesis](https://arxiv.org/abs/1803.09017)

**Text Encoder**: [Neural Speech Synthesis with Transformer Network](https://arxiv.org/abs/1809.08895)

**Speaker Encoder**: [Generalized End-to-End Loss for Speaker Verification](https://arxiv.org/abs/1710.10467)

Here speaker was trained beforehand with 10K speakers and from there 100 speaker was used for the speaker embeddings. 

**TTS Decoder**: TransformerTTS [Neural Speech Synthesis with Transformer Network](https://arxiv.org/abs/1809.08895)

**Mutual Information**: [MINE Mutual Information Neural Estimator](https://arxiv.org/abs/1801.04062).

Mutual information is measured via three different ways Kullback-Leibler divergence (KLD), Rényi divergences and CLUB 
[A Contrastive Log-ratio Upper Bound of Mutual Information](https://arxiv.org/abs/2006.12013)

---
*autoregressive_config.yaml*

Mutual Information measurement.

    mine_sep_call:              
       True                     compute the inputs of mine sepeartely from tts
       False                    use tts to compute the inputs of mine
       
    mine_type:
    'MINE'                      Kullback–Leibler divergence
    'CLUB'                      club
    'MINE_CLUB'                 Kullback–Leibler divergence + club
        
    divergence_type: 'reyni' [using Rényi divergences]
        mine_beta_values: [0]               Kullback–Leibler divergence (KLD)
        mine_beta_values: [0.5]             Hellinger distance
        mine_beta_values: [1]               reverse KLD
        mine_beta_values: [0, 0.5, 1]       KLD + Hellinger distance + reverse KLD (All three together)
    divergence_type: 'KL' [normal]          Kullback–Leibler divergence
    
MI was measured between pairs (style-text, style-speaker, text-speaker).

    mine_pair_types: 
        'style_text'            between style and text
        'style_speaker'         between style and speaker
        'text_speaker'          between text and speaker
    
For each pair we use one mine network. So total no of mine nets for above 3 mine_pair_types. 
After we sum the 3 measured MIs and add it to the tts loss.
    
    mine_no_of_nets: 3
    
Only encode style and text not use speaker embeddings.
    
    use_speaker_style_tts: False

Use pretrain model.

    use_pretrained: True
    use_pretrained_type: 
        'all'                   use all layer weights from the pretarined model
        'text'                  use only TextEncoder layer weights from the pretarined model
        
Freeze weights of text encoder.
    
    tts_model.text_encoder.trainable = False

use style encoder 2nd time with decoder output as input and l2 between 1st and 2nd style embeddings.

    use_style_loss: True
    
*data_config.yaml*

PreTrained Model

    pretrained_all_weights:             'pretrained_models/for_all/ckpt-67'
    pretrained_all_config:              'pretrained_models/for_all/autoregressive_config.yaml'
    pretrained_text_encoder_weights:    'pretrained_models/for_text_encoder/ckpt-25'
    pretrained_text_encoder_config:     'pretrained_models/for_text_encoder/autoregressive_config.yaml'
    
    
---
### run

    python train_autoregressive.py

checkpoints

    wavernn_test_mine/
    
logs with synthesized audio (griffin-lim).

    tensorboard --logdir=checkpoints/wavernn_test_mine/autoregressive_logs

### Test
    ref_audio\
        for_speaker_tts\            # for style + speaker TTS
            speaker_embedding\      # speaker embeddings
            style\                  # styles
            test_sentences          # text

### ENV

conda env create -f transformer_TTS.yml