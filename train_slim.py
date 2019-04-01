import tensorflow as tf
import net
import os
import numpy as np

# https://github.com/tensorflow/models/blob/master/research/slim/train_image_classifier.py

slim = tf.contrib.slim

tf.app.flags.DEFINE_string('train_dir', '/home/gglee/Data/Landmark/train', 'Directory for training and logging')
tf.app.flags.DEFINE_string('train_tfr', '/home/gglee/Data/160v5.0322.train.tfrecord', '.tfrecord for training')
tf.app.flags.DEFINE_string('val_tfr', '/home/gglee/Data/160v5.0322.val.tfrecord', '.tfrecord for validation')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size to use')

tf.app.flags.DEFINE_string('loss', 'wing', 'Loss func: [l1, l2, wing]')
tf.app.flags.DEFINE_string('optimizer', 'sgd', 'Optimizer to use: [adadelt, adagrad, adam, ftrl, momentum, sgd or rmsprop]')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate')

tf.app.flags.DEFINE_float('momentum', 0.99, 'Initial learning rate')
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')
tf.app.flags.DEFINE_float('adam_beta1', 0.9, 'The exponential decay rate for the 1st moment estimates.')
tf.app.flags.DEFINE_float('adam_beta2', 0.999, 'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_string('learning_rate_decay_type', 'exponential', 'Which learning rate decay to use: [fixed, exponential, or polynomial]')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.94, 'Learning rate decay factor')
tf.app.flags.DEFINE_integer('learning_rate_decay_step', 50000, 'Decay lr at every N steps')
tf.app.flags.DEFINE_boolean('learning_rate_decay_staircase', True, 'Staircase decay or not')
tf.app.flags.DEFINE_float('end_learning_rate', 0.00005, 'The minimal lr used by a polynomial lr decay')

tf.app.flags.DEFINE_float('moving_average_decay', None, 'The decay to use for moving average decay. If left as None, no moving average decay')
tf.app.flags.DEFINE_integer('max_number_of_steps', 300000, 'The maximum number of training steps')
tf.app.flags.DEFINE_boolean('use_batch_norm', False, 'To use or not BatchNorm on conv layers')

tf.app.flags.DEFINE_string('regularizer', None, 'l1, l2 or l1_12')
tf.app.flags.DEFINE_float('regularizer_lambda', 0.004, 'Lambda for the regularization')
tf.app.flags.DEFINE_float('regularizer_lambda_2', 0.004, 'Lambda for the regularization')

tf.app.flags.DEFINE_integer('quantize_delay', -1, 'Number of steps to start quantized training. -1 to disable it')
tf.app.flags.DEFINE_integer('depth_multiplier', 4, '')

FLAGS = tf.app.flags.FLAGS


def _write_current_setting(train_path):
    if not os.path.exists(train_path):
        os.makedirs(train_path)

    with open(os.path.join(train_path, 'train_setting.txt'), 'w') as wf:
        wf.write('%s\n' % train_path)
        wf.write('train_tfr: %s\n' % FLAGS.train_tfr)

        wf.write('optimizer: %s\n' % FLAGS.optimizer)
        wf.write('learning_rate: %f\n' % FLAGS.learning_rate)
        if FLAGS.optimizer == 'momentum':
            wf.write('momentum: %f\n' % FLAGS.momentum)
        elif FLAGS.optimizer == 'rmsprop':
            wf.write('rmsprop_momentum: %f, rmsprop_decay: %f\n' % (FLAGS.rmsprop_momentum, FLAGS.rmsprop_decay))
        elif FLAGS.optimizer == 'adam':
            wf.write('adam_beta1: %f, adam_beta2: %f\n' % (FLAGS.adam_beta1, FLAGS.adam_beta2))
        wf.write('learning_rate_decay_type: %s\n' % FLAGS.learning_rate_decay_type)
        wf.write('learning_rate_decay_factor: %f\n' % FLAGS.learning_rate_decay_factor)

        if FLAGS.use_batch_norm:
            wf.write('use_batch_norm\n')

        if FLAGS.moving_average_decay:
            wf.write('moving_average_decay: %f\n' % FLAGS.moving_average_decay)

        wf.write('regularizer: %s\n' % FLAGS.regularizer)
        if FLAGS.regularizer:
            wf.write('lambda1: %f, lambda2: %f\n' % (FLAGS.regularizer_lambda, FLAGS.regularizer_lambda_2)) \
                if FLAGS.regularizer == 'l1_l2' else wf.write('lambda: %f\n' % FLAGS.regularizer_lambda)

        if FLAGS.quantize_delay >= 0:
            wf.write('quantize_delay: %d\n' % FLAGS.quantize_delay)

        wf.write('depth_multiplier: %d\n' % FLAGS.depth_multiplier)


def _config_weights_regularizer(reg, scale, scale2=None):
    if reg == 'l1':
        return slim.l1_regularizer(scale)
    elif reg == 'l2':
        return slim.l2_regularizer(scale)
    elif reg == 'l1_l2':
        return slim.l1_l2_regularizer(scale, scale2)
    else:
        raise ValueError('Regularizer [%s] was not recognized' % FLAGS.regularizer)


def _config_optimizer(lr):
    if FLAGS.optimizer == 'adadelta':
        opt = tf.train.AdadeltaOptimizer(lr)
    elif FLAGS.optimizer == 'adagrad':
        opt = tf.train.AdadeltaOptimizer(lr)
    elif FLAGS.optimizer == 'adam':
        opt = tf.train.AdamOptimizer(lr, beta1=FLAGS.adam_beta1, beta2=FLAGS.adam_beta2)
    elif FLAGS.optimizer == 'ftrl':
        opt = tf.train.FtrlOptimizer(lr)
    elif FLAGS.optimizer == 'momentum':
        opt = tf.train.MomentumOptimizer(lr, momentum=FLAGS.momentum)
    elif FLAGS.optimizer == 'rmsprop':
        opt = tf.train.RMSPropOptimizer(lr, decay=FLAGS.rmsprop_decay, momentum=FLAGS.rmsprop_momentum)
    elif FLAGS.optimizer == 'sgd':
        opt = tf.train.GradientDescentOptimizer(lr)
    else:
        raise ValueError('Optimizer [%s] was not recognized' % FLAGS.optimizer)

    return opt


def _config_learning_rate(gstep):

    if FLAGS.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(FLAGS.learning_rate,
                                          gstep,
                                          FLAGS.learning_rate_decay_step,
                                          FLAGS.learning_rate_decay_factor,
                                          staircase=FLAGS.learning_rate_decay_staircase,
                                          name='exponential_decay_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'fixed':
        return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(FLAGS.learning_rate, gstep, FLAGS.learning_rate_decay_step,
                                         FLAGS.end_learning_rate, power=1.0, cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized' % FLAGS.learning_rate_decay_type)


def _config_loss_function(points, predictions):

    if FLAGS.loss == 'l1':
        return slim.losses.absolute_difference(points, predictions)
    elif FLAGS.loss == 'l2':
        return slim.losses.mean_squared_error(points, predictions)
    elif FLAGS.loss == 'wing':
        return wing_loss(points, predictions)


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


def _get_trainable_variables(t_scopes):
    if t_scopes is None:
        return tf.trainable_variables()

    ts = [s.strip() for s in t_scopes]

    var2train = []
    for s in ts:
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, s)
        var2train.extend(vars)

    return var2train


def wing_loss(points, estimates, w=10, e=2):
    if len(points) != len(estimates):
        raise ValueError('point and landmark estimate should have the same length')

    C = w - w/np.log(1+w/e)

    def wing(x, _w, _e):
        x = abs(x)
        return _w * np.log(1+x/_e) if x < _w else x - C

    # with tf.name_scoep(tf.GraphKeys.LOSSES) as scope:
    wings = map(lambda x: wing(x[0]-x[1], w, e), zip(points, estimates))
    wing_loss = tf.reduce_sum(wings)
    tf.losses.add_loss(wing_loss, tf.GraphKeys.LoSSES)

    return wing_loss


# train_log_dir = '/home/gglee/Data/Landmark/train_logs/conv5+decay'
# if not tf.gfile.Exists(train_log_dir):
#     tf.gfile.MakeDirs(train_log_dir)

if __name__=='__main__':
    with tf.Graph().as_default():
        TRAIN_TFR_PATH = FLAGS.train_tfr
        VAL_TFR_PATH = FLAGS.val_tfr

        _write_current_setting(FLAGS.train_dir)

        count_train = 0
        for record in tf.python_io.tf_record_iterator(TRAIN_TFR_PATH):
            count_train += 1
        print('%d samples in training data' % count_train)

        count_val = 0
        for record in tf.python_io.tf_record_iterator(VAL_TFR_PATH):
            count_val += 1

        dataset = tf.data.TFRecordDataset(TRAIN_TFR_PATH)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(1000)
        dataset = dataset.map(_parse_function, num_parallel_calls=16)
        dataset = dataset.batch(FLAGS.batch_size)
        dataset.prefetch(buffer_size=FLAGS.batch_size)
        iterator = dataset.make_initializable_iterator()

        data_val = tf.data.TFRecordDataset(VAL_TFR_PATH)
        data_val = data_val.repeat()
        data_val = data_val.shuffle(1000)
        data_val = data_val.map(_parse_function, num_parallel_calls=16)
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
            if FLAGS.regularizer == 'l1' or FLAGS.regularizer == 'l2':
                regularizer = _config_weights_regularizer(FLAGS.regularizer, FLAGS.regularizer_lambda)
            else:
                regularizer = _config_weights_regularizer(FLAGS.regularizer, FLAGS.regularizer_lambda, FLAGS.regularizer_lambda_2)

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

        learning_rate = _config_learning_rate(global_step)
        optimizer = _config_optimizer(learning_rate)

        moving_average_variables, variable_averages = None, None
        if FLAGS.moving_average_decay:
            moving_average_variables = slim.get_model_variables()
            variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, global_step)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, variable_averages.apply(moving_average_variables))

        if FLAGS.quantize_delay > 0:
            tf.contrib.quantize.create_training_graph(
                quant_delay=FLAGS.quantize_delay)

        trainable_scopes = None
        trainable_vars = _get_trainable_variables(trainable_scopes)
        train_tensor = slim.learning.create_train_op(total_loss, optimizer,
                                                     global_step=global_step,
                                                     variables_to_train=trainable_vars)
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