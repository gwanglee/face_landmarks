import tensorflow as tf
import net

slim = tf.contrib.slim

tf.app.flags.DEFINE_string('train_dir', '/home/gglee/Data/Landmark/train', 'Directory for training and logging')
tf.app.flags.DEFINE_float('weight_decay', 0.0005, 'The weight decay on the model weights')
tf.app.flags.DEFINE_string('optimizer', 'sgd', 'Optimizer to use: [adadelt, adagrad, adam, ftrl, momentum, sgd or rmsprop]')
tf.app.flags.DEFINE_integer('quantize_delay', -1, 'Number of steps to start quantized training. -1 to disable it')
tf.app.flags.DEFINE_string('learning_rate_decay_type', 'exponential', 'Which learning rate decay to use: [fixed, exponential, or polynomial]')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate')
tf.app.flags.DEFINE_float('end_learning_rate', 0.0001, 'The minimal lr used by a polynomial lr decay')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.995, 'Learning rate decay factor')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 2.0, 'Number of epochs after which lr decays')
tf.app.flags.DEFINE_float('moving_average_decay', None, 'The decay to use for moving average decay. If left as None, no moving average decay')
tf.app.flags.DEFINE_integer('max_number_of_steps', None, 'The maximum number of training steps')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size to use')

FLAGS = tf.app.flags.FLAGS


def _config_optimizer(learning_rate):
    if FLAGS.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    elif FLAGS.optimizer == 'adagrad':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    elif FLAGS.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(learning_rate)
    elif FLAGS.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate)
    elif FLAGS.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
    elif FLAGS.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized' % FLAGS.optimizer)

    return optimizer


def _config_learning_rate(num_samples_per_epoch, global_step):
    decay_steps = int(num_samples_per_epoch * FLAGS.num_epochs_per_decay / FLAGS.batch_size)
    if FLAGS.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(FLAGS.learning_rate,
                                          global_step,
                                          decay_steps,
                                          FLAGS.learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'fixed':
        return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(FLAGS.learning_rate, global_step, decay_steps,
                                         FLAGS.end_learning_rate, power=1.0, cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized' % FLAGS.learning_rate_decay_type)


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


def train_step_fn(sess, train_op, global_step, train_step_kwargs):
    train_step_fn.step = global_step.eval(sess)
    total_loss, should_stop = slim.learning.train_step(sess, train_op, global_step, train_step_kwargs)

    if train_step_fn.step % 1000 == 0:    # validation for every 1000 steps
        validation_loss = sess.run([val_loss])
        print('>> global step {}: train={}, validation={}'.format(train_step_fn.step, total_loss, validation_loss))

    return total_loss, should_stop

# train_log_dir = '/home/gglee/Data/Landmark/train_logs/conv5+decay'
# if not tf.gfile.Exists(train_log_dir):
#     tf.gfile.MakeDirs(train_log_dir)

with tf.Graph().as_default():
    TRAIN_TFR_PATH = '/home/gglee/Data/160v5.0322.train.tfrecord'
    VAL_TFR_PATH = '/home/gglee/Data/160v5.0322.val.tfrecord'

    count_train = 0
    for record in tf.python_io.tf_record_iterator(TRAIN_TFR_PATH):
        count_train += 1

    count_val = 0
    for record in tf.python_io.tf_record_iterator(VAL_TFR_PATH):
        count_val += 1

    dataset = tf.data.TFRecordDataset(TRAIN_TFR_PATH)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(1000)
    dataset = dataset.map(_parse_function, num_parallel_calls=8)
    dataset = dataset.batch(FLAGS.batch_size)
    dataset.prefetch(buffer_size=FLAGS.batch_size)
    iterator = dataset.make_initializable_iterator()

    data_val = tf.data.TFRecordDataset(VAL_TFR_PATH)
    data_val = data_val.repeat()
    data_val = data_val.map(_parse_function, num_parallel_calls=8)
    data_val = dataset.batch(count_val)
    data_val.prefetch(buffer_size=count_val)
    iter_val = data_val.make_initializable_iterator()

    image, points = iterator.get_next()
    val_imgs, val_pts = iter_val.get_next()

    # predictions, _ = net.lannet(image, is_training=True)
    with tf.variable_scope('model') as scope:
        predictions, _ = net.lannet(image, is_training=True)
        val_pred, _ = net.lannet(image, is_training=False)

    loss = slim.losses.absolute_difference(points, predictions)
    total_loss = slim.losses.get_total_loss()
    val_loss = tf.losses.absolute_difference(tf.squeeze(points), tf.squeeze(val_pred), loss_collection='validation')

    global_step = slim.create_global_step()

    learning_rate = _config_learning_rate(count_train, global_step)
    optimizer = _config_optimizer(learning_rate)

    if FLAGS.moving_average_decay:
        moving_average_variables = slim.get_model_variables()
        variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, global_step)

    train_tensor = slim.learning.create_train_op(total_loss, optimizer)
    summaries = set([tf.summary.scalar('losses/total_loss', total_loss)])
    summaries.add(tf.summary.scalar('learning_rate', learning_rate))
    summaries.add(tf.summary.scalar('losses/validation', val_loss))
    summary_op = tf.summary.merge(list(summaries), name='summary_op')
    # tf.summary.scalar('loss/total_loss', total_loss)
    # tf.summary.scalar('learning_rate', learning_rate)

    def init_fn(sess):
        sess.run(iterator.initializer)

    slim.learning.train(train_tensor,
                        logdir=FLAGS.train_dir,
                        local_init_op=tf.group(tf.local_variables_initializer(), tf.tables_initializer(),
                                               iterator.initializer, iter_val.initializer),
                        number_of_steps=FLAGS.max_number_of_steps,
                        save_summaries_secs=150,
                        save_interval_secs=300,
                        summary_op=summary_op,
                        train_step_fn=train_step_fn,
                        )