import tensorflow as tf
import net

slim = tf.contrib.slim

def _parse_function(example_proto):
    print('parsing')
    features = {"image": tf.FixedLenFeature([56*56*3], tf.string),
                "points": tf.FixedLenFeature([68*2], tf.float32)}
    parsed_features = tf.parse_single_example(example_proto, features)

    img = tf.reshape(tf.decode_raw(parsed_features["image"], tf.uint8), (56, 56, 3))
    normed = tf.subtract(tf.multiply(tf.cast(img, tf.float32), 2.0 / 255.0), 1.0)

    pts = tf.reshape(tf.cast(parsed_features['points'], tf.float32), (136, ))
    print(normed, pts)
    return normed, pts


train_log_dir = '/Users/gglee/Landmark/train_logs'
if not tf.gfile.Exists(train_log_dir):
    tf.gfile.MakeDirs(train_log_dir)

with tf.Graph().as_default():
    TRAIN_TFR_PATH = '/Users/gglee/Data/Landmark/export/160v5.val.tfrecord'

    # slim.dataset.Dataset(TRAIN_TFR_PATH, )
    dataset = tf.data.TFRecordDataset(TRAIN_TFR_PATH)
    dataset = dataset.map(_parse_function)
    dataset = dataset.repeat().batch(64)
    iterator = dataset.make_initializable_iterator()

    image, points = iterator.get_next()
    predictions, _ = net.lannet(image, is_training=True)

    slim.losses.absolute_difference(points, predictions)

    total_loss = slim.losses.get_total_loss()
    tf.summary.scalar('losses/total_loss', total_loss)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=.001)

    train_tensor = slim.learning.create_train_op(total_loss, optimizer)

    def init_fn(sess):
        sess.run(iterator.initializer)

    slim.learning.train(train_tensor, train_log_dir,
                        local_init_op=tf.group(tf.local_variables_initializer(), tf.tables_initializer(),
                                               iterator.initializer),
                        number_of_steps=1000,
                        save_summaries_secs=300,
                        save_interval_secs=600)