
# Style Encoder

Style encoder architecture same as text encoder.
Style-embeddings concatenated with text embeddings in the 2nd dim.

text-embeddings = batch x n1 x 512 (n1 no of phonemes)

Style-embeddings = batch x n2 x 512 (n2 no of frames)

Final = concat((text-embeddings , Style-embeddings) , 1)

run
python train_autoregressive.py

checkpoints
wavernn_stn_2nd_concat

tensorboard --logdir=checkpoints/wavernn_stn_2nd_concat/autoregressive_logs