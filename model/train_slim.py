import tensorflow as tf
from model import net
from model import mnet2
import os
import math
from data import data_landmark as data

import sys

# https://github.com/tensorflow/models/blob/master/research/slim/train_image_classifier.py

slim = tf.contrib.slim

tf.app.flags.DEFINE_string('train_dir', '/home/gglee/Data/Landmark/train', 'Directory for training and logging')
tf.app.flags.DEFINE_string('train_tfr', '/home/gglee/Data/160v5.0322.train.tfrecord', '.tfrecord for training')
tf.app.flags.DEFINE_string('val_tfr', '/home/gglee/Data/160v5.0322.val.tfrecord', '.tfrecord for validation')
tf.app.flags.DEFINE_boolean('is_color', True, 'RGB or gray input')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size to use')
tf.app.flags.DEFINE_integer('input_size', 56, 'N x N for the network')

tf.app.flags.DEFINE_string('loss', 'l1', 'Loss func: [l1, l2, wing, euc_wing, pointwise_l2, chain, sqrt]')
tf.app.flags.DEFINE_string('optimizer', 'sgd', 'Optimizer to use: [adadelt, adagrad, adam, ftrl, momentum, sgd or rmsprop]')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate')
tf.app.flags.DEFINE_float('wing_w', 0.5, 'w for wing_loss')
tf.app.flags.DEFINE_float('wing_eps', 2, 'eps for wing_loss')
# tf.app.flags.DEFINE_float('min_learning_rate', None, 'Minimum value of learning rate to use')

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
tf.app.flags.DEFINE_float('depth_multiplier', 2.0, '')
tf.app.flags.DEFINE_float('depth_gamma', 1.0, '')

tf.app.flags.DEFINE_string('basenet', 'lan', '[lan, mnet2]')

FLAGS = tf.app.flags.FLAGS


def _write_current_setting(train_path):
    if not os.path.exists(train_path):
        os.makedirs(train_path)

    with open(os.path.join(train_path, 'train_setting.txt'), 'w') as wf:
        wf.write('%s\n' % train_path)
        wf.write('train_tfr: %s\n' % FLAGS.train_tfr)
        wf.write('is_color: %r\n' % FLAGS.is_color)
        wf.write('input_size: %d\n' % FLAGS.input_size)

        wf.write('optimizer: %s\n' % FLAGS.optimizer)
        wf.write('loss: %s\n' % FLAGS.loss)
        wf.write('learning_rate: %f\n' % FLAGS.learning_rate)

        if FLAGS.optimizer == 'momentum':
            wf.write('momentum: %f\n' % FLAGS.momentum)
        elif FLAGS.optimizer == 'rmsprop':
            wf.write('rmsprop_momentum: %f, rmsprop_decay: %f\n' % (FLAGS.rmsprop_momentum, FLAGS.rmsprop_decay))
        elif FLAGS.optimizer == 'adam':
            wf.write('adam_beta1: %f, adam_beta2: %f\n' % (FLAGS.adam_beta1, FLAGS.adam_beta2))

        wf.write('learning_rate_decay_type: %s\n' % FLAGS.learning_rate_decay_type)
        wf.write('learning_rate_decay_factor: %f\n' % FLAGS.learning_rate_decay_factor)

        wf.write('use_batch_norm: %r\n' % FLAGS.use_batch_norm)

        if FLAGS.moving_average_decay:
            wf.write('moving_average_decay: %f\n' % FLAGS.moving_average_decay)

        wf.write('regularizer: %s\n' % FLAGS.regularizer)
        if FLAGS.regularizer:
            wf.write('lambda1: %f, lambda2: %f\n' % (FLAGS.regularizer_lambda, FLAGS.regularizer_lambda_2)) \
                if FLAGS.regularizer == 'l1_l2' else wf.write('lambda: %f\n' % FLAGS.regularizer_lambda)

        if FLAGS.quantize_delay >= 0:
            wf.write('quantize_delay: %d\n' % FLAGS.quantize_delay)

        wf.write('depth_multiplier: %f\n' % FLAGS.depth_multiplier)
        wf.write('depth_gamma: %f\n' % FLAGS.depth_gamma)


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
        # return slim.losses.absolute_difference(points, predictions)
        return l1_loss(points, predictions)
    elif FLAGS.loss == 'l2':
        # return slim.losses.mean_squared_error(points, predictions)
        return l2_loss(points, predictions)
    elif FLAGS.loss == 'pointwise_l2':
        return pointwise_l2_loss(points, predictions)
    elif FLAGS.loss == 'wing':
        return wing_loss(points, predictions, FLAGS.wing_w, FLAGS.wing_eps)
    elif FLAGS.loss == 'euc_wing':
        return euc_wing_loss(points, predictions, FLAGS.wing_w, FLAGS.wing_eps)
    elif FLAGS.loss == 'chain':
        return chain_loss(points, predictions)
    elif FLAGS.loss == 'sqrt':
        return sqrt_loss(points, predictions)
    else:
        raise ValueError('Could not recog. loss fn.')


def chain_loss(landmarks, labels):
    with tf.name_scope('chain_loss'):
        N = FLAGS.batch_size
        p2d = tf.reshape(landmarks, (-1, 68, 2))
        g2d = tf.reshape(labels, (-1, 68, 2))

        p2d0 = tf.slice(p2d, [0, 0, 0], [N, 68-1, 2])
        p2d1 = tf.slice(p2d, [0, 1, 0], [N, 68-1, 2])

        g2d0 = tf.slice(g2d, [0, 0, 0], [N, 68 - 1, 2])
        g2d1 = tf.slice(g2d, [0, 1, 0], [N, 68 - 1, 2])

        pd = tf.subtract(p2d0, p2d1)        # (dx, dy) for prediction
        gd = tf.subtract(g2d0, g2d1)        # (dx, dy) for ground truth

        # fixme: change ops ordering >> abs(sub(slice(pd, gd))) -> slice(abs(sub(pd, gd)))

        begin, size = 1-1, 17-1
        pp = tf.slice(pd, [0, begin, 0], [N, size, 2])
        pg = tf.slice(gd, [0, begin, 0], [N, size, 2])
        l1_face = tf.reshape(tf.reduce_mean(tf.abs(tf.subtract(pp, pg))), [1, 1])

        begin, size = 18-1, 22-18
        pp = tf.slice(pd, [0, begin, 0], [N, size, 2])
        pg = tf.slice(gd, [0, begin, 0], [N, size, 2])
        l1_lebrow = tf.reshape(tf.reduce_mean(tf.abs(tf.subtract(pp, pg))), [1, 1])

        begin, size = 23-1, 27-23
        pp = tf.slice(pd, [0, begin, 0], [N, size, 2])
        pg = tf.slice(gd, [0, begin, 0], [N, size, 2])
        l1_rebrow = tf.reshape(tf.reduce_mean(tf.abs(tf.subtract(pp, pg))), [1, 1])

        begin, size = 37 - 1, 42-37
        pp = tf.slice(pd, [0, begin, 0], [N, size, 2])
        pg = tf.slice(gd, [0, begin, 0], [N, size, 2])
        l1_leye = tf.reshape(tf.reduce_mean(tf.abs(tf.subtract(pp, pg))), [1, 1])

        begin, size = 43 - 1, 48 - 43
        pp = tf.slice(pd, [0, begin, 0], [N, size, 2])
        pg = tf.slice(gd, [0, begin, 0], [N, size, 2])
        l1_reye = tf.reshape(tf.reduce_mean(tf.abs(tf.subtract(pp, pg))), [1, 1])

        begin, size = 28 - 1, 31-28
        pp = tf.slice(pd, [0, begin, 0], [N, size, 2])
        pg = tf.slice(gd, [0, begin, 0], [N, size, 2])
        l1_nose_1 = tf.reshape(tf.reduce_mean(tf.abs(tf.subtract(pp, pg))), [1, 1])

        begin, size = 32 - 1, 36 - 32
        pp = tf.slice(pd, [0, begin, 0], [N, size, 2])
        pg = tf.slice(gd, [0, begin, 0], [N, size, 2])
        l1_nose_2 = tf.reshape(tf.reduce_mean(tf.abs(tf.subtract(pp, pg))), [1, 1])

        begin, size = 49 - 1, 60-49
        pp = tf.slice(pd, [0, begin, 0], [N, size, 2])
        pg = tf.slice(gd, [0, begin, 0], [N, size, 2])
        l1_mouth_out = tf.reshape(tf.reduce_mean(tf.abs(tf.subtract(pp, pg))), [1, 1])

        begin, size = 61 - 1, 68 - 61
        pp = tf.slice(pd, [0, begin, 0], [N, size, 2])
        pg = tf.slice(gd, [0, begin, 0], [N, size, 2])
        l1_mouth_in = tf.reshape(tf.reduce_mean(tf.abs(tf.subtract(pp, pg))), [1, 1])

        losses = tf.concat([l1_face, l1_lebrow, l1_rebrow, l1_leye, l1_reye, l1_nose_1, l1_nose_2, l1_mouth_out, l1_mouth_in], 1)
        loss_shape = tf.reduce_mean(losses)
        loss_l1 = l1_loss(landmarks, labels)
        loss = loss_l1 + loss_shape

        tf.losses.add_loss(loss, tf.GraphKeys.LOSSES)
        return loss


def wing_loss(landmarks, labels, w, epsilon):
    """
    Arguments:
        landmarks, labels: float tensors with shape [batch_size, num_landmarks, 2].
        w, epsilon: a float numbers.
    Returns:
        a float tensor with shape [].
    """
    with tf.name_scope('wing_loss'):
        x = landmarks - labels
        c = w * (1.0 - math.log(1.0 + w / epsilon))
        absolute_x = tf.abs(x)
        losses = tf.where(
            tf.greater(w, absolute_x),
            w * tf.log(1.0 + absolute_x / epsilon),
            absolute_x - c
        )

        loss = tf.reduce_mean(losses)
        tf.losses.add_loss(loss, tf.GraphKeys.LOSSES)
        return loss


def euc_wing_loss(landmarks, labels, w, epsilon):
    """
    Arguments:
        landmarks, labels: float tensors with shape [batch_size, num_landmarks, 2].
        w, epsilon: a float numbers.
    Returns:
        a float tensor with shape [].
    """
    with tf.name_scope('wing_loss'):
        dx = landmarks - labels
        dx2 = tf.reshape(tf.multiply(dx, dx), [-1, 68, 2])  # [[dx0^2, dy0^2], [dx1^2, dy1^2], ..., [dxn^2, dyn^2]]
        euc = tf.sqrt(tf.reduce_sum(dx2, 2))       # point-wise euclidean distance

        c = w * (1.0 - math.log(1.0 + w / epsilon))
        losses = tf.where(
            tf.greater(w, euc),
            w * tf.log(1.0 + euc / epsilon),
            euc - c
        )
        # loss = tf.reduce_mean(tf.reduce_sum(losses, axis=[1, 2]), axis=0)
        loss = tf.reduce_mean(losses)
        tf.losses.add_loss(loss, tf.GraphKeys.LOSSES)
        return loss

def pointwise_l2_loss(landmarks, labels):
    """
    Arguments:
        landmarks, labels: float tensors with shape [batch_size, num_landmarks, 2].
        w, epsilon: a float numbers.
    Returns:
        a float tensor with shape [].
    """
    with tf.name_scope('pointwise_l2_loss'):
        dx = landmarks - labels
        dx2 = tf.reshape(tf.multiply(dx, dx), [-1, 68, 2])  # [[dx0^2, dy0^2], [dx1^2, dy1^2], ..., [dxn^2, dyn^2]]
        l2 = tf.sqrt(tf.reduce_sum(dx2, 2))  # point-wise euclidean distance
        loss = tf.reduce_mean(l2)
        tf.losses.add_loss(loss, tf.GraphKeys.LOSSES)
        return loss


def sqrt_loss(landmarks, predicts):
    with tf.name_scope('sqrt_loss'):
        loss = tf.reduce_mean(tf.sqrt(tf.abs(landmarks - predicts)))
        tf.losses.add_loss(loss, tf.GraphKeys.LOSSES)
        return loss


def l1_loss(landmarks, predicts):
    with tf.name_scope('l1_loss'):
        loss = tf.reduce_mean(tf.abs(tf.subtract(landmarks, predicts)))
        tf.losses.add_loss(loss, tf.GraphKeys.LOSSES)
        return loss


def l2_loss(landmarks, predicts):
    with tf.name_scope('l2_loss'):
        diff = tf.subtract(landmarks, predicts)
        loss = tf.reduce_mean(tf.multiply(diff, diff))
        tf.losses.add_loss(loss, tf.GraphKeys.LOSSES)
        return loss


def train_step_fn(sess, train_op, global_step, train_step_kwargs):
    train_step_fn.step = global_step.eval(sess)
    total_loss, should_stop = slim.learning.train_step(sess, train_op, global_step, train_step_kwargs)

    if train_step_fn.step % 1000 == 0:    # validation for every 1000 steps
        print('>> global step {}: loss={}'.format(train_step_fn.step, total_loss))
        # validation_loss = sess.run([val_loss])
        # print('>> global step {}: train={}, validation={}'.format(train_step_fn.step, total_loss, validation_loss))

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

# train_log_dir = '/home/gglee/Data/Landmark/train_logs/conv5+decay'
# if not tf.gfile.Exists(train_log_dir):
#     tf.gfile.MakeDirs(train_log_dir)

if __name__=='__main__':
    with tf.Graph().as_default():
        TRAIN_TFR_PATH = FLAGS.train_tfr
        VAL_TFR_PATH = FLAGS.val_tfr

        _write_current_setting(FLAGS.train_dir)

        train_data_count = data.get_tfr_record_count(TRAIN_TFR_PATH)
        print('%d samples in training data' % train_data_count)

        train_data = data.load_tfrecord(TRAIN_TFR_PATH, input_size=FLAGS.input_size, batch_size=FLAGS.batch_size,
                                        num_parallel_calls=16, is_color=FLAGS.is_color, shuffle=True, augment=True)
        train_data_itr = train_data.make_initializable_iterator()

        val_data_count = data.get_tfr_record_count(VAL_TFR_PATH)
        val_data = data.load_tfrecord(VAL_TFR_PATH, input_size=FLAGS.input_size, batch_size=FLAGS.batch_size,
                                      num_parallel_calls=16, is_color=FLAGS.is_color, shuffle=False, augment=False)
        val_data_itr = val_data.make_initializable_iterator()

        image, points = train_data_itr.get_next()
        val_imgs, val_pts = val_data_itr.get_next()

        print(image, val_imgs)

        norm_fn = None
        norm_params = {}
        if FLAGS.use_batch_norm:
            norm_fn = slim.batch_norm
            norm_params = {'is_training': True}

        regularizer = None
        if FLAGS.regularizer:
            if FLAGS.regularizer == 'l1' or FLAGS.regularizer == 'l2':
                regularizer = _config_weights_regularizer(FLAGS.regularizer, FLAGS.regularizer_lambda)
            elif FLAGS.regularizer == 'None':
                regularizer = None
            else:
                regularizer = _config_weights_regularizer(FLAGS.regularizer, FLAGS.regularizer_lambda, FLAGS.regularizer_lambda_2)

        with tf.variable_scope('model') as scope:
            intensor = tf.identity(image, 'input')
            if FLAGS.basenet == 'lan':
                predictions, _ = net.lannet(intensor, normalizer_fn=norm_fn, normalizer_params=norm_params,
                                            regularizer=regularizer, depth_mul=FLAGS.depth_multiplier, depth_gamma=FLAGS.depth_gamma)
                val_pred, _ = net.lannet(val_imgs, normalizer_fn=norm_fn, normalizer_params={'is_training': False},
                                         regularizer=regularizer, depth_mul=FLAGS.depth_multiplier, depth_gamma=FLAGS.depth_gamma)
            elif FLAGS.basenet == 'mnet2':
                predictions, _ = mnet2.mnet2(intensor)

        loss = _config_loss_function(points, predictions)
        total_loss = slim.losses.get_total_loss()
        # val_loss = l2_loss(val_pts, val_pred)

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
            sess.run(train_data_itr.initializer)

        slim.learning.train(train_tensor,
                            logdir=FLAGS.train_dir,
                            local_init_op=tf.group(tf.local_variables_initializer(), tf.tables_initializer(),
                                                   train_data_itr.initializer, val_data_itr.initializer),
                            number_of_steps=FLAGS.max_number_of_steps,
                            save_summaries_secs=150,
                            save_interval_secs=300,
                            summary_op=summary_op,
                            train_step_fn=train_step_fn,
                            )