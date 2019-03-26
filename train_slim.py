import tensorflow as tf
import net

slim = tf.contrib.slim

tf.app.flags.DEFINE_string('train_dir', '/home/gglee/Data/Landmark/train', 'Directory for training and logging')
tf.app.flags.DEFINE_string('train_tfr', '/home/gglee/Data/160v5.0322.train.tfrecord', '.tfrecord for training')
tf.app.flags.DEFINE_string('val_tfr', '/home/gglee/Data/160v5.0322.val.tfrecord', '.tfrecord for validation')
tf.app.flags.DEFINE_float('weight_decay', 0.0005, 'The weight decay on the model weights')
tf.app.flags.DEFINE_string('optimizer', 'sgd', 'Optimizer to use: [adadelt, adagrad, adam, ftrl, momentum, sgd or rmsprop]')
tf.app.flags.DEFINE_integer('quantize_delay', -1, 'Number of steps to start quantized training. -1 to disable it')
tf.app.flags.DEFINE_string('learning_rate_decay_type', 'exponential', 'Which learning rate decay to use: [fixed, exponential, or polynomial]')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate')
tf.app.flags.DEFINE_float('momentum', 0.99, 'Initial learning rate')
tf.app.flags.DEFINE_float('end_learning_rate', 0.00005, 'The minimal lr used by a polynomial lr decay')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.995, 'Learning rate decay factor')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 2.0, 'Number of epochs after which lr decays')
tf.app.flags.DEFINE_float('moving_average_decay', None, 'The decay to use for moving average decay. If left as None, no moving average decay')
tf.app.flags.DEFINE_integer('max_number_of_steps', None, 'The maximum number of training steps')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size to use')
tf.app.flags.DEFINE_boolean('use_batch_norm', False, 'To use or not BatchNorm on conv layers')
tf.app.flags.DEFINE_string('regularizer', None, 'l1, l2 or l1_12')
tf.app.flags.DEFINE_integer('depth_multiplier', 4, '')

FLAGS = tf.app.flags.FLAGS


def _config_weights_regularizer(regularizer, scale, scale2=None):
    if regularizer == 'l1':
        return slim.l1_regularizer(scale)
    elif regularizer == 'l2':
        return slim.l2_regularizer(scale)
    elif regularizer == 'l1_l2':
        return slim.l1_l2_regularizer(scale, scale2)
    else:
        raise ValueError('Regularizer [%s] was not recognized' % FLAGS.regularizer)


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
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=FLAGS.momentum)
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
    elif FLAGS.learning_rate_decay_type == 'fixed' or FLAGS.optimizer == 'adam':
        return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(FLAGS.learning_rate, global_step, decay_steps,
                                         FLAGS.end_learning_rate, power=1.0, cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized' % FLAGS.learning_rate_decay_type)


def _parse_function(example_proto):
    features = {"image": tf.FixedLenFeature([56*56*3], tf.string),
                "points": tf.FixedLenFeature([68*2], tf.float32)}
    parsed_features = tf.parse_single_example(example_proto, features)

    img = tf.reshape(tf.decode_raw(parsed_features["image"], tf.uint8), (56, 56, 3))
    normed = tf.subtract(tf.multiply(tf.cast(img, tf.float32), 2.0 / 255.0), 1.0)

    pts = tf.reshape(tf.cast(parsed_features['points'], tf.float32), (136, ))
    print('parsing >> ', normed, pts)
    return normed, pts


def train_step_fn(sess, train_op, global_step, train_step_kwargs):
    train_step_fn.step = global_step.eval(sess)
    total_loss, should_stop = slim.learning.train_step(sess, train_op, global_step, train_step_kwargs)

    if train_step_fn.step % 1000 == 0:    # validation for every 1000 steps
        # validation_loss = sess.run([val_loss])
        # print('>> global step {}: train={}, validation={}'.format(train_step_fn.step, total_loss, validation_loss))
        print('>> global step {}: train={}'.format(train_step_fn.step, total_loss))

    return total_loss, should_stop

# train_log_dir = '/home/gglee/Data/Landmark/train_logs/conv5+decay'
# if not tf.gfile.Exists(train_log_dir):
#     tf.gfile.MakeDirs(train_log_dir)

if __name__=='__main__':
    with tf.Graph().as_default():
        TRAIN_TFR_PATH = FLAGS.train_tfr
        VAL_TFR_PATH = FLAGS.val_tfr

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
        data_val = data_val.shuffle(1000)
        data_val = data_val.map(_parse_function, num_parallel_calls=8)
        data_val = dataset.batch(FLAGS.batch_size)
        data_val.prefetch(buffer_size=FLAGS.batch_size)
        iter_val = data_val.make_initializable_iterator()

        image, points = iterator.get_next()
        val_imgs, val_pts = iter_val.get_next()

        norm_fn = None
        norm_params = {}
        if FLAGS.use_batch_norm:
            norm_fn = slim.batch_norm
            norm_params = {'is_training': True}

        regularizer = None
        if FLAGS.regularizer:
            regularizer = _config_weights_regularizer(FLAGS.regularizer, 0.001)

    # predictions, _ = net.lannet(image, is_training=True)
    with tf.variable_scope('model') as scope:
        intensor = tf.identity(image, 'input')
        predictions, _ = net.lannet(intensor, is_training=True, normalizer_fn=norm_fn, normalizer_params=norm_params,
                                    regularizer=regularizer, depth_mul=FLAGS.depth_multiplier)
        # val_pred, _ = net.lannet(val_imgs, is_training=False)

        loss = slim.losses.absolute_difference(points, predictions)
        total_loss = slim.losses.get_total_loss()
        # val_loss = tf.losses.absolute_difference(val_pts, val_pred, loss_collection='validation')

        global_step = slim.create_global_step()

        learning_rate = _config_learning_rate(count_train, global_step)
        optimizer = _config_optimizer(learning_rate)

        if FLAGS.moving_average_decay:
            moving_average_variables = slim.get_model_variables()
            variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, global_step)

        train_tensor = slim.learning.create_train_op(total_loss, optimizer)
        summaries = set([tf.summary.scalar('losses/total_loss', total_loss)])
        summaries.add(tf.summary.scalar('learning_rate', learning_rate))
        # summaries.add(tf.summary.scalar('losses/validation', val_loss))
        summary_op = tf.summary.merge(list(summaries), name='summary_op')

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