# -*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow.python.keras.layers import Concatenate
import tensorflow.contrib.layers as layers
from tensorflow.python.keras.layers import Add, Lambda, Dropout, Dense, TimeDistributed, Activation
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import Ones, Zeros
import numpy as np
from tensorflow.python.keras.layers import Conv1D

class Encoder():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, layers=2, dropout=0.1):
        self.emb_dropout = Dropout(dropout)
        self.layers = [EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout) for _ in range(layers)]

    def __call__(self, x, return_att=False, mask=None, active_layers=999):
        x = self.emb_dropout(x)
        if return_att: atts = []
        for enc_layer in self.layers[:active_layers]:
            x, att = enc_layer(x, mask)
            if return_att: atts.append(att)
        return (x, atts) if return_att else x


class EncoderLayer():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        self.self_att_layer = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.ffn_layer = FeedForward(d_model, d_inner_hid, dropout=dropout)

    def __call__(self, enc_input, mask=None):
        output, slf_attn = self.self_att_layer(enc_input, enc_input, enc_input, mask=mask)
        output = self.ffn_layer(output)
        return output, slf_attn


class FeedForward():
    def __init__(self, d_hid, d_inner_hid, dropout=0.1):
        self.w_1 = Conv1D(d_inner_hid, 1, activation='relu')
        self.w_2 = Conv1D(d_hid, 1)
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(dropout)

    def __call__(self, x):
        output = self.w_1(x)
        output = self.w_2(output)
        output = self.dropout(output)
        output = Add()([output, x])
        return self.layer_norm(output)


class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


class ScaledDotProductAttention():
    def __init__(self, d_model, attn_dropout=0.1):
        self.temper = np.sqrt(d_model)
        self.dropout = Dropout(attn_dropout)

    def __call__(self, q, k, v, mask):
        attn = Lambda(lambda x:K.batch_dot(x[0],x[1],axes=[2,2])/self.temper)([q, k])
        if mask is not None:
            mmask = Lambda(lambda x:(-1e+10)*(1-x))(mask)
            attn = Add()([attn, mmask])
        attn = Activation('softmax')(attn)
        attn = self.dropout(attn)
        output = Lambda(lambda x:K.batch_dot(x[0], x[1]))([attn, v])
        return output, attn


class MultiHeadAttention():
    def __init__(self, n_head, d_model, d_k, d_v, dropout, mode=0, use_norm=True):
        self.mode = mode
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        if mode == 0:
            self.qs_layer = Dense(n_head*d_k, use_bias=False)
            self.ks_layer = Dense(n_head*d_k, use_bias=False)
            self.vs_layer = Dense(n_head*d_v, use_bias=False)
        elif mode == 1:
            self.qs_layers = []
            self.ks_layers = []
            self.vs_layers = []
            for _ in range(n_head):
                self.qs_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.ks_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.vs_layers.append(TimeDistributed(Dense(d_v, use_bias=False)))
        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = LayerNormalization() if use_norm else None
        self.w_o = TimeDistributed(Dense(d_model))

    def __call__(self, q, k, v, mask=None):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        if self.mode == 0:
            qs = self.qs_layer(q)  # [batch_size, len_q, n_head*d_k]
            ks = self.ks_layer(k)
            vs = self.vs_layer(v)

            def reshape1(x):
                s = tf.shape(x)   # [batch_size, len_q, n_head * d_k]
                x = tf.reshape(x, [s[0], s[1], n_head, d_k])
                x = tf.transpose(x, [2, 0, 1, 3])
                x = tf.reshape(x, [-1, s[1], d_k])  # [n_head * batch_size, len_q, d_k]
                return x
            qs = Lambda(reshape1)(qs)
            ks = Lambda(reshape1)(ks)
            vs = Lambda(reshape1)(vs)

            if mask is not None:
                mask = Lambda(lambda x:K.repeat_elements(x, n_head, 0))(mask)
            head, attn = self.attention(qs, ks, vs, mask=mask)

            def reshape2(x):
                s = tf.shape(x)   # [n_head * batch_size, len_v, d_v]
                x = tf.reshape(x, [n_head, -1, s[1], s[2]])
                x = tf.transpose(x, [1, 2, 0, 3])
                x = tf.reshape(x, [-1, s[1], n_head*d_v])  # [batch_size, len_v, n_head * d_v]
                return x
            head = Lambda(reshape2)(head)
        elif self.mode == 1:
            heads = []; attns = []
            for i in range(n_head):
                qs = self.qs_layers[i](q)
                ks = self.ks_layers[i](k)
                vs = self.vs_layers[i](v)
                head, attn = self.attention(qs, ks, vs, mask)
                heads.append(head); attns.append(attn)
            head = Concatenate()(heads) if n_head > 1 else heads[0]
            attn = Concatenate()(attns) if n_head > 1 else attns[0]
        outputs = self.w_o(head)
        outputs = Dropout(self.dropout)(outputs)
        if not self.layer_norm: return outputs, attn
        outputs = Add()([outputs, q])
        return self.layer_norm(outputs), attn


def task_specific_attention(inputs, output_size,
                            initializer=layers.xavier_initializer(),
                            activation_fn=tf.tanh, scope=None,index=0):
    with tf.variable_scope(scope or 'attention'+str(index)) as scope:
        attention_context_vector = tf.get_variable(name='attention_context_vector',
                                                   shape=[output_size],
                                                   initializer=initializer,
                                                   dtype=tf.float32)
        input_projection = layers.fully_connected(inputs, output_size,
                                                  activation_fn=activation_fn,
                                                  scope=scope)
        vector_attn = tf.reduce_sum(tf.multiply(input_projection, attention_context_vector), axis=2, keep_dims=True)
        attention_weights = tf.nn.softmax(vector_attn, dim=1)
        weighted_projection = tf.multiply(input_projection, attention_weights)
        outputs = tf.reduce_sum(weighted_projection, axis=1)
        return outputs


############################### Main ######################################3
max_seq_len = 10
model_dir = './output/dhanr'
train_epochs = 8
epochs_per_eval = 2
batch_size = 40
d_model = 7
train_file = './data/channel.data'
test_file = './data/channel.test'

# 1. Set raw feature_column
channel_id = tf.feature_column.categorical_column_with_hash_bucket('channel_id', hash_bucket_size=3, dtype=tf.string)
channel_id = tf.feature_column.indicator_column(channel_id)
channel_feature_1 = tf.feature_column.numeric_column('channel_feature_1')
channel_feature_2 = tf.feature_column.numeric_column('channel_feature_2')
user_id_origin = tf.feature_column.categorical_column_with_hash_bucket('user_id', hash_bucket_size=100, dtype=tf.string)
user_id = tf.feature_column.indicator_column(user_id_origin)
user_feature_3 = tf.feature_column.numeric_column('user_feature_3')
user_feature_4 = tf.feature_column.categorical_column_with_vocabulary_list(
    'user_feature_4', ['cntyhot', 'OLI2I', 'OLS2I', 'RTS2I', 'compl', 'RTI2I', 'RTB2I', 'OLC2I', 'RTC2I', 'OLB2I'])
item_id_origin = tf.feature_column.categorical_column_with_hash_bucket('item_id', hash_bucket_size=10000, dtype=tf.string)
item_id = tf.feature_column.indicator_column(item_id_origin)
item_feature_5 = tf.feature_column.numeric_column('item_feature_5')
item_feature_6 = tf.feature_column.numeric_column('item_feature_6')
item_feature_7 = tf.feature_column.numeric_column('item_feature_7')
ids_columns = []
for i in range(max_seq_len):
    ids_columns.append(tf.feature_column.categorical_column_with_hash_bucket('sid_' + str(i), hash_bucket_size=10000))
ids_columns.append(item_id_origin)
ids_columns = tf.feature_column.shared_embedding_columns(ids_columns, dimension=4)


sf1_list = []
for i in range(max_seq_len):
    sf1_list.append(tf.feature_column.numeric_column('sf1_' + str(i)))
sf2_list = []
for i in range(max_seq_len):
    sf2_list.append(tf.feature_column.numeric_column('sf2_' + str(i)))
sf3_list = []
for i in range(max_seq_len):
    sf3_list.append(tf.feature_column.numeric_column('sf3_' + str(i)))

# Transformations
user_feature_3_buckets = tf.feature_column.bucketized_column(
    user_feature_3, boundaries=[0., 0.1, 0.2, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
)

# 2. Add  columns
user_item_columns = [
    user_id,
    user_feature_3,
    item_feature_7,
    item_feature_6,
    item_feature_5,
    item_id,
    tf.feature_column.indicator_column(user_feature_4),
    user_feature_3_buckets
]

channel_columns = [
    channel_id,
    channel_feature_1,
    channel_feature_2
]


seq_i_list = []
for i in range(max_seq_len):
    seq_i_list.append([ids_columns[i], sf1_list[i], sf2_list[i], sf3_list[i]])

channel_0 = []
for item_i in seq_i_list[0:3]:
    channel_0.extend(item_i)

channel_1 = []
for item_i in seq_i_list[3:6]:
    channel_1.extend(item_i)

channel_2 = []
for item_i in seq_i_list[7:10]:
    channel_2.extend(item_i)


# 3. Build model
def my_model(features, labels, mode, params):
    if mode != tf.estimator.ModeKeys.PREDICT:
        labels=tf.reshape(labels,(-1,1))
    # deep network
    user_item_net = tf.feature_column.input_layer(features, params['user_item_columns'])
    channel_net = tf.feature_column.input_layer(features, params['channel_columns'])

    for units in params['hidden_units']:
        user_item_net = tf.layers.dense(user_item_net, units=units, activation=tf.nn.relu)

    for units in params['hidden_units']:
        channel_net = tf.layers.dense(channel_net, units=units, activation=tf.nn.relu)

    net = tf.concat([user_item_net, channel_net], axis=1)

    # hierarchical attention network
    chanel_att_list = []
    channel_len = 3
    for channel_index in range(channel_len):
        channel_input = tf.feature_column.input_layer(features, params['channel_'+str(channel_index)])
        channel_input = tf.reshape(channel_input, [-1, channel_len, d_model])
        # item encoder
        encoder = Encoder(d_model=7, d_inner_hid=128, n_head=1, d_k=1, d_v=1, layers=2, dropout=0.1)
        encoder_output = encoder(channel_input, mask=None, active_layers=999)
        # item-level attention layer
        output_size = 64
        tsa = task_specific_attention(inputs=encoder_output, output_size=output_size,index=channel_index)
        chanel_att_list.append(tsa)
    chanel_att_list_all = tf.concat(chanel_att_list, -1)
    chanel_att_list_all = tf.reshape(chanel_att_list_all, [-1, channel_len, output_size])
    # channel encoder
    encoder2 = Encoder(d_model=output_size, d_inner_hid=128, n_head=1, d_k=1, d_v=1, layers=2, dropout=0.1)
    list_encode = encoder2(chanel_att_list_all, mask=None, active_layers=999)
    # channel-level attention layer
    output_size = 64
    tsas = task_specific_attention(inputs=list_encode, output_size=output_size, index=-1)

    # Concat deep & hierarchical attention network
    net = tf.concat([net, tsas], -1)
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # Compute logits.
    logits = tf.layers.dense(net, 1, activation=None)
    # Compute predictions.
    prop = tf.nn.sigmoid(logits)
    predicted_classes = tf.greater_equal(prop, 0.5)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.sigmoid(logits),
            'logits': logits
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))

    # Compute evaluation metrics.
    accuracy = tf.compat.v1.metrics.accuracy(labels=labels,
                                             predictions=predicted_classes,
                                             name='acc_op')

    metrics = {'accuracy': accuracy}
    tf.compat.v1.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


params = {
    'user_item_columns': user_item_columns,
    'channel_columns': channel_columns,
    'hidden_units': [100, 10],
    'hidden_units_2': [10],
    'n_classes': 1,
    'channel_0': channel_0,
    'channel_1': channel_1,
    'channel_2': channel_2
}

model = tf.estimator.Estimator(
    model_fn=my_model,
    config=tf.estimator.RunConfig(
        model_dir=model_dir,
        save_checkpoints_steps=100
    ),
    params=params
)


# 5. Read input data in csv
_CSV_COLUMNS = []
_CSV_COLUMN_DEFAULTS = []
_NUM_EXAMPLES = {
    'train': 32561,
    'validation': 16281,
}
for i in range(max_seq_len):
    _CSV_COLUMNS.append('sf1_' + str(i))
    _CSV_COLUMNS.append('sf2_' + str(i))
    _CSV_COLUMNS.append('sf3_' + str(i))
    _CSV_COLUMNS.append('sid_' + str(i))
    _CSV_COLUMN_DEFAULTS.extend([[0.], [0.], [0.], ['']])

_CSV_COLUMNS.extend([
    'item_id', 'channel_id', 'user_id',
    'channel_feature_1', 'channel_feature_2', 'user_feature_4',
    'user_feature_3', 'item_feature_7', 'item_feature_5',
    'item_feature_6', 'label'
])
print _CSV_COLUMNS
_CSV_COLUMN_DEFAULTS.extend([[''], [''], [''], [0.], [0.], [''], [0.], [0.], [0.], [0.], [0.]])


def input_fn(data_file, num_epochs, shuffle, batch_size):
    assert tf.gfile.Exists(data_file), "{0} not found.".format(data_file)

    def parse_csv(line):
        print("Parsing", data_file)
        columns = tf.decode_csv(line, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('label')
        return features, labels

    dataset = tf.data.TextLineDataset(data_file) \
        .map(parse_csv, num_parallel_calls=5)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'] + _NUM_EXAMPLES['validation'])

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


# Train + Eval
for n in range(train_epochs // epochs_per_eval):
    model.train(input_fn=lambda: input_fn(train_file, epochs_per_eval, True, batch_size))
    results = model.evaluate(input_fn=lambda: input_fn(
        test_file, 1, False, batch_size))

    # Display Eval results
    print("Results at epoch {0}".format((n + 1) * epochs_per_eval))
    print('-' * 30)

    for key in sorted(results):
        print("{0:20}: {1:.4f}".format(key, results[key]))

preds = model.predict(input_fn=lambda: input_fn(test_file, 1, False, batch_size), predict_keys=None)
with open("./output/dhanr_pred.txt", "w") as fo:
    for prob in preds:
        print prob
        fo.write("%s\n" % (prob))

