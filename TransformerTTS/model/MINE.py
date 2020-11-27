import tensorflow as tf

from model.layers import MineNetLinear, MineNetFirstOrder, MineNetSecondOrder, CLUBNet, MineNetLinearQ

# dynamically allocate GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


class MINE(tf.keras.models.Model):
    def __init__(self,
                 conv_filters: list,
                 conv_kernel: int,
                 dense_hidden_units: list,
                 beta_values: list,
                 divergence_type: str,
                 pair_type: str,
                 **kwargs):
        super(MINE, self).__init__(**kwargs)
        self.mine_net = MineNetFirstOrder(dense_hidden_units=dense_hidden_units, name='MineNet')
        # self.mine_net = MineNetSecondOrder(filters=conv_filters, kernel_size=conv_kernel, dense_hidden_units=dense_hidden_units, name='MineNet')
        # self.mine_net = MineNetLinear(dense_hidden_units=dense_hidden_units, name='MineNet')
        # self.mine_net = MineNetLinearQ(dense_hidden_units=dense_hidden_units, name='MineNet')

        self.beta = beta_values
        self.div_type = divergence_type
        self.pair_type = pair_type

        # self.text_proj = tf.keras.layers.RNN(tf.keras.layers.GRUCell(256), return_sequences=True)

    @property
    def step(self):
        return int(self.optimizer.iterations)

    def measure_mi(self, joint, marginal, mi_holder):
        loss = 0
        exp_terms = []
        curr_smooth = mi_holder['smoothing_factor']
        prev_smooth = 1 - curr_smooth
        if self.div_type == 'KL':
            # Mutual Information via KL Divergence Disentanglement information
            term1 = tf.reduce_mean(joint)

            term2_exp = tf.reduce_mean(tf.exp(marginal))
            # smoothing
            term2_exp = curr_smooth * term2_exp + prev_smooth * mi_holder['exp_terms'][0][1]
            term2 = tf.math.log(term2_exp)

            loss = tf.cast(term1 - term2, tf.float32)
            exp_terms.append([tf.cast(0.0, tf.float32), term2_exp])
        elif self.div_type == 'reyni':
            # Reyni Divengence Disentanglement information
            for i, beta_val in enumerate(self.beta):
                term1_exp, term2_exp, term1, term2 = tf.cast(0.0, tf.float32), tf.cast(0.0, tf.float32), 0.0, 0.0
                prev_term1_exp = mi_holder['exp_terms'][i][0]
                prev_term2_exp = mi_holder['exp_terms'][i][1]

                if beta_val == 0:
                    max_val = tf.reduce_max((1 - beta_val) * marginal)

                    term1 = tf.reduce_mean(joint)

                    term2_exp = tf.reduce_mean(tf.math.exp((1 - beta_val) * marginal - max_val))
                    # smoothing
                    term2_exp = curr_smooth * term2_exp + prev_smooth * prev_term2_exp
                    term2 = (1 / (1 - beta_val)) * (tf.math.log(term2_exp) + max_val)
                elif beta_val == 1:
                    max_val = tf.reduce_max(- beta_val * joint)

                    term1_exp = tf.reduce_mean(tf.math.exp(-beta_val * joint - max_val))
                    # smoothing
                    term1_exp = curr_smooth * term1_exp + prev_smooth * prev_term1_exp
                    term1 = -(1 / beta_val) * (tf.math.log(term1_exp) + max_val)

                    term2 = tf.reduce_mean(marginal)
                else:
                    max_val_1 = tf.reduce_max(- beta_val * joint)
                    max_val_2 = tf.reduce_max((1 - beta_val) * marginal)

                    term1_exp = tf.reduce_mean(tf.math.exp(-beta_val * joint - max_val_1))
                    # smoothing
                    term1_exp = curr_smooth * term1_exp + prev_smooth * prev_term1_exp
                    term1 = -(1 / beta_val) * (tf.math.log(term1_exp) + max_val_1)

                    term2_exp = tf.reduce_mean(tf.math.exp((1 - beta_val) * marginal - max_val_2))
                    # smoothing
                    term2_exp = curr_smooth * term2_exp + prev_smooth * prev_term2_exp
                    term2 = (1 / (1 - beta_val)) * (tf.math.log(term2_exp) + max_val_2)

                loss += tf.cast(term1 - term2, tf.float32)
                exp_terms.append([term1_exp, term2_exp])
        return loss, exp_terms

    def call(self, text_embed, style_embed, speaker_embed, exp_terms_holder):
        # create Joint and Marginal distribution by random shuffling
        joint, marginal = 0, 0

        # select randomly a single char
        one_random_char = tf.random.shuffle(tf.range(tf.shape(text_embed)[1]))[:1]
        text_embed = tf.gather(text_embed, one_random_char, axis=1)
        text_embed_shuffle = tf.random.shuffle(text_embed)

        # select all char and project them to single dimension
        # text projection
        # text_embed = self.text_proj(text_embed)
        # text_embed = tf.expand_dims(text_embed[:, -1, :], axis=1)
        # # random shuffle for marginal distribution creation
        # rand_shuffle_idx = tf.random.shuffle(tf.range(tf.shape(text_embed)[0]))
        # text_embed_shuffle = tf.gather(text_embed, rand_shuffle_idx, axis=0)

        if self.pair_type == 'style_text':
            joint = tf.concat([style_embed, text_embed], axis=-1)
            marginal = tf.concat([style_embed, text_embed_shuffle], axis=-1)
        elif self.pair_type == 'style_speaker':
            joint = tf.concat([style_embed, speaker_embed], axis=-1)
            marginal = tf.concat([style_embed, tf.random.shuffle(speaker_embed)], axis=-1)
        elif self.pair_type == 'text_speaker':
            joint = tf.concat([text_embed, speaker_embed], axis=-1)
            marginal = tf.concat([text_embed, tf.random.shuffle(speaker_embed)], axis=-1)
        elif self.pair_type == 'style_text_speaker':
            joint = tf.concat([style_embed, text_embed, speaker_embed], axis=-1)
            marginal = tf.concat([style_embed, text_embed_shuffle, tf.random.shuffle(speaker_embed)], axis=-1)
        else:
            print('pair_type is not supported')

        # if speaker_embed is None:
        #     joint = tf.concat([style_embed, text_embed_joint], axis=-1)
        #     # 2nd order statistics
        #     # joint = tf.matmul(tf.transpose(style_embed, [0, 2, 1]), text_embed_joint)
        #     # for 2d conv
        #     # joint = tf.expand_dims(joint, axis=-1)
        #
        #     marginal = tf.concat([style_embed, text_embed_marginal], axis=-1)
        #     # 2nd order statistics
        #     # marginal = tf.matmul(tf.transpose(style_embed, [0, 2, 1]), text_embed_marginal)
        #     # for 2d conv
        #     # marginal = tf.expand_dims(marginal, axis=-1)
        # else:
        #     speaker_embed_joint = tf.expand_dims(speaker_embed, 1)  # batch x 1 x embed_dim
        #     speaker_embed_marginal = tf.random.shuffle(speaker_embed_joint)
        #     # merge for Joint
        #     joint = tf.concat([style_embed, text_embed_joint, speaker_embed_joint], axis=-1)
        #     # merge for marginal distribution
        #     marginal = tf.concat([style_embed, text_embed_marginal, speaker_embed_marginal], axis=-1)

        joint = self.mine_net(joint)
        marginal = self.mine_net(marginal)
        mi, exp_terms_holder = self.measure_mi(joint, marginal, exp_terms_holder)
        return mi, exp_terms_holder


class CLUB(tf.keras.models.Model):
    # CLUB: A Contrastive Log-ratio Upper Bound of Mutual Information
    def __init__(self,
                 dense_hidden_units: list,
                 pair_type: str,
                 **kwargs):
        super(CLUB, self).__init__(**kwargs)
        self.net_mu = CLUBNet(dense_hidden_units=dense_hidden_units, log_var=False, name='ClubNet_mu')
        self.net_log_var = CLUBNet(dense_hidden_units=dense_hidden_units, log_var=True, name='ClubNet_log_var')
        self.pair_type = pair_type

    @property
    def step(self):
        return int(self.optimizer.iterations)

    def call(self, text_embed, style_embed, speaker_embed, exp_terms_holder):
        # create positive and negative samples
        positive, negative = 0, 0

        # select randomly a single char
        one_random_char = tf.random.shuffle(tf.range(tf.shape(text_embed)[1]))[:1]
        text_embed = tf.gather(text_embed, one_random_char, axis=1)
        text_embed_shuffle = tf.random.shuffle(text_embed)
        speaker_embed_shuffle = tf.random.shuffle(speaker_embed)

        if self.pair_type == 'style_text':
            mu = self.net_mu(style_embed)
            log_var = self.net_log_var(style_embed)
            positive = -(mu - text_embed) ** 2 / 2. / tf.exp(log_var)
            negative = -(mu - text_embed_shuffle) ** 2 / 2. / tf.exp(log_var)
        elif self.pair_type == 'style_speaker':
            mu = self.net_mu(style_embed)
            log_var = self.net_log_var(style_embed)
            positive = -(mu - speaker_embed) ** 2 / 2. / tf.exp(log_var)
            negative = -(mu - speaker_embed_shuffle) ** 2 / 2. / tf.exp(log_var)
        elif self.pair_type == 'text_speaker':
            mu = self.net_mu(text_embed)
            log_var = self.net_log_var(text_embed)
            positive = -(mu - speaker_embed) ** 2 / 2. / tf.exp(log_var)
            negative = -(mu - speaker_embed_shuffle) ** 2 / 2. / tf.exp(log_var)
        else:
            print('pair_type is not supported')

        lld = tf.reduce_mean(tf.reduce_sum(positive, -1))
        bound = tf.reduce_mean(tf.reduce_sum(positive, -1) - tf.reduce_sum(negative, -1))
        return lld, bound
