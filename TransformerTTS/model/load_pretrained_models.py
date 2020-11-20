import tensorflow as tf
import ruamel.yaml


def load_pretrained_models(config_path, model_weights_path, model_type):
    if model_type == 'all':
        from pretrained_models.for_all.get_model import get_model_AutoregressiveTransformer
    else:
        from pretrained_models.for_text_encoder.get_model import get_model_AutoregressiveTransformer

    # read config
    with open(str(config_path), 'rb') as model_yaml:
        config = ruamel.yaml.YAML().load(model_yaml)
    # create model instance
    model = get_model_AutoregressiveTransformer(config)
    # restore weights
    tf.train.Checkpoint(net=model).restore(model_weights_path).expect_partial()
    # dry run
    if model_type == 'all':
        _ = model.call(inputs=tf.random.uniform(shape=[1, 50], dtype=tf.float32),
                       targets=tf.random.uniform(shape=[1, 50, 80], dtype=tf.float32),
                       spk_embed=tf.random.uniform(shape=[1, 1, 256], dtype=tf.float32),
                       train_text_encoder=True,
                       train_style_encoder=True,
                       train_decoder=True)
    else:
        _ = model.call(inputs=tf.random.uniform(shape=[1, 50], dtype=tf.float32),
                       targets=tf.random.uniform(shape=[1, 50, 80], dtype=tf.float32),
                       training=True)

    return model
