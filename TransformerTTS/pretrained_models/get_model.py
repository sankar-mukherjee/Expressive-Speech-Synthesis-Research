import numpy as np

from pretrained_models.models import AutoregressiveTransformer


def get_model_AutoregressiveTransformer(config):
    return AutoregressiveTransformer(mel_channels=config['mel_channels'],
                                     encoder_model_dimension=config['encoder_model_dimension'],
                                     decoder_model_dimension=config['decoder_model_dimension'],
                                     encoder_num_heads=config['encoder_num_heads'],
                                     decoder_num_heads=config['decoder_num_heads'],
                                     encoder_feed_forward_dimension=config[
                                         'encoder_feed_forward_dimension'],
                                     decoder_feed_forward_dimension=config[
                                         'decoder_feed_forward_dimension'],
                                     encoder_maximum_position_encoding=config[
                                         'encoder_max_position_encoding'],
                                     decoder_maximum_position_encoding=config[
                                         'decoder_max_position_encoding'],
                                     encoder_dense_blocks=config['encoder_dense_blocks'],
                                     decoder_dense_blocks=config['decoder_dense_blocks'],
                                     decoder_prenet_dimension=config['decoder_prenet_dimension'],
                                     encoder_prenet_dimension=config['encoder_prenet_dimension'],
                                     encoder_attention_conv_kernel=config['encoder_attention_conv_kernel'],
                                     decoder_attention_conv_kernel=config['decoder_attention_conv_kernel'],
                                     encoder_attention_conv_filters=config['encoder_attention_conv_filters'],
                                     decoder_attention_conv_filters=config['decoder_attention_conv_filters'],
                                     postnet_conv_filters=config['postnet_conv_filters'],
                                     postnet_conv_layers=config['postnet_conv_layers'],
                                     postnet_kernel_size=config['postnet_kernel_size'],
                                     dropout_rate=config['dropout_rate'],
                                     max_r=np.array(config['reduction_factor_schedule'])[0, 1].astype(np.int32),
                                     mel_start_value=config['mel_start_value'],
                                     mel_end_value=config['mel_end_value'],
                                     phoneme_language=config['phoneme_language'],
                                     with_stress=config['with_stress'],
                                     debug=config['debug'])
