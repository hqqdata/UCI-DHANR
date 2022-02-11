# -*- coding: UTF-8 -*-
import tensorflow as tf

max_seq_len = 10

# 1. user/channel/item features：
channel_id = tf.feature_column.categorical_column_with_hash_bucket('channel_id', hash_bucket_size=3, dtype=tf.string)
channel_id = tf.feature_column.indicator_column(channel_id)
channel_feature_1 = tf.feature_column.numeric_column('channel_feature_1')
channel_feature_2 = tf.feature_column.numeric_column('channel_feature_2')
user_id = tf.feature_column.categorical_column_with_hash_bucket('user_id', hash_bucket_size=100, dtype=tf.string)
user_id = tf.feature_column.indicator_column(user_id)
user_feature_3 = tf.feature_column.numeric_column('user_feature_3')
user_feature_4 = tf.feature_column.categorical_column_with_vocabulary_list(
    'user_feature_4', ['cntyhot', 'OLI2I', 'OLS2I', 'RTS2I', 'compl', 'RTI2I', 'RTB2I', 'OLC2I', 'RTC2I', 'OLB2I'])
item_id = tf.feature_column.categorical_column_with_hash_bucket('item_id', hash_bucket_size=10000, dtype=tf.string)
item_id = tf.feature_column.indicator_column(item_id)
item_feature_5 = tf.feature_column.numeric_column('item_feature_5')
item_feature_6 = tf.feature_column.numeric_column('item_feature_6')
item_feature_7 = tf.feature_column.numeric_column('item_feature_7')
# Transformations
user_feature_3_buckets = tf.feature_column.bucketized_column(
    user_feature_3, boundaries=[0., 0.1, 0.2, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
)

# 2. Embeddings
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

model_dir = './output/uci'


# 3. Deep model
def my_model(features, labels, mode, params):
    # Create three fully connected layers.
    import tensorflow as tf
    if mode != tf.estimator.ModeKeys.PREDICT:
        labels = tf.reshape(labels,(-1,1))
        print "labels.get_shape=", labels.get_shape().as_list()

    user_item_net = tf.feature_column.input_layer(features, params['user_item_columns'])
    channel_net = tf.feature_column.input_layer(features, params['channel_columns'])

    for units in params['hidden_units']:
        user_item_net = tf.layers.dense(user_item_net, units=units, activation=tf.nn.relu)

    for units in params['hidden_units']:
        channel_net = tf.layers.dense(channel_net, units=units, activation=tf.nn.relu)

    net = tf.concat([user_item_net, channel_net], axis=1)

    for units in params['hidden_units_2']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # Compute logits (1 per class).
    logits = tf.layers.dense(net, 1, activation=None)
    print "logits.get_shape=",logits.get_shape().as_list()
    # Compute predictions.
    prop = tf.nn.sigmoid(logits)
    predicted_classes = tf.greater_equal(prop,0.5)

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
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

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
    'n_classes': 2,
}

model = tf.estimator.Estimator(
    model_fn=my_model,
    config=tf.estimator.RunConfig(
        model_dir=model_dir,
        save_checkpoints_steps=100
    ),
    params=params
)


# 4. Train & Evaluate
_CSV_COLUMNS = []
_CSV_COLUMN_DEFAULTS = []

for i in range(max_seq_len):
    _CSV_COLUMNS.append('sf1_' + str(i))
    _CSV_COLUMNS.append('sf2_' + str(i))
    _CSV_COLUMNS.append('sf3_' + str(i))
    _CSV_COLUMNS.append('sid_' + str(i))
    _CSV_COLUMN_DEFAULTS.extend([[0.], [0.], [0.], [0.]])

_CSV_COLUMNS.extend([
    'item_id', 'channel_id', 'user_id',
    'channel_feature_1', 'channel_feature_2', 'user_feature_4',
    'user_feature_3', 'item_feature_7', 'item_feature_5',
    'item_feature_6', 'label'
])
print _CSV_COLUMNS
_CSV_COLUMN_DEFAULTS.extend([[''], [''], [''], [0.], [0.], [''], [0.], [0.], [0.], [0.], [0.]])

_NUM_EXAMPLES = {
    'train': 32561,
    'validation': 16281,
}


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
train_epochs = 6
epochs_per_eval = 2
batch_size = 40
train_file = './data/channel.data'
test_file = './data/channel.test'

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
with open("./output/uci_pred.txt", "w") as fo:
    for prob in preds:
        print prob
        fo.write("%s\n" % (prob))

