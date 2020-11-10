import tensorflow as tf
from pretrained_models.get_model import get_model_AutoregressiveTransformer
import ruamel.yaml


def load_pretrained_models(config_path, model_weights_path):
    # read config
    with open(str(config_path), 'rb') as model_yaml:
        config = ruamel.yaml.YAML().load(model_yaml)
    # create model instance
    model = get_model_AutoregressiveTransformer(config)
    # restore weights
    tf.train.Checkpoint(net=model).restore(model_weights_path).expect_partial()
    # dry run
    _ = model.call(inputs=tf.random.uniform(shape=[1, 50], dtype=tf.float32),
                   targets=tf.random.uniform(shape=[1, 50, 80], dtype=tf.float32),
                   training=True)
    return model
