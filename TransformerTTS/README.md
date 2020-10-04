
# GST Style Encoder and MINE with TransformerTTS

A tensorflow 2 implementation of the [Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End 
Speech Synthesis](https://arxiv.org/abs/1803.09017) with transformer TTS and 
[MINE Mutual Information Neural Estimator](https://arxiv.org/abs/1801.04062)

run

    python train_autoregressive.py

Option to load pretrained model. uncomment last line.

    # load TTS model assign textEncoder with pretrained model
    tts_model = config_manager.get_model()
    config_manager.compile_model(tts_model)
    # tts_model.text_encoder = pretrained_text_encoder_model


checkpoints

    wavernn_gst_mine
    
logs

    tensorboard --logdir=checkpoints/wavernn_gst_mine/autoregressive_logs

###env
conda env create -f transformer_TTS.yml