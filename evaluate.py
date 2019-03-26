import tensorflow as tf
import net
import os
from operator import itemgetter

slim = tf.contrib.slim

tf.app.flags.DEFINE_string('val_tfr', '/home/gglee/Data/160v5.0322.val.tfrecord', '.tfrecord for validation')
tf.app.flags.DEFINE_string('models_dir', '/home/gglee/Data/Landmark/train', 'where trained models are stored')

FLAGS = tf.app.flags.FLAGS


def _parse_function(example_proto):
    features = {"image": tf.FixedLenFeature([56*56*3], tf.string),
                "points": tf.FixedLenFeature([68*2], tf.float32)}
    parsed_features = tf.parse_single_example(example_proto, features)

    img = tf.reshape(tf.decode_raw(parsed_features["image"], tf.uint8), (56, 56, 3))
    normed = tf.subtract(tf.multiply(tf.cast(img, tf.float32), 2.0 / 255.0), 1.0)

    pts = tf.reshape(tf.cast(parsed_features['points'], tf.float32), (136, ))
    print('parsing >> ', normed, pts)
    return normed, pts


def evaluate(ckpt_path, tfr_path):
    with tf.Graph().as_default():
        dataset = tf.data.TFRecordDataset(tfr_path)
        dataset.map(_parse_function, num_parallel_calls=4)
        dataset.prefetch(buffer_size=64)
        iterator = dataset.make_initializable_iterator()

        image, points = iterator.get_next()
        preds, _ = net.lannet(image, is_training=False)

        eval_mae = slim.metrics.aggregate_metric_map({'loss': slim.metrics.streaming_mean_absolute_error(points, preds)})

        slim.evaluation.evaluate_once(checkpoint_path=ckpt_path,
                                      logdir=ckpt_path,
                                      local_init_op=tf.group(tf.local_variables_initializer(), tf.tables_initializer(),
                                                            iterator.initializer),
                                      eval_op=eval_mae,
                                      num_evals=128)


if __name__=='__main__':

    if not os.path.exists(FLAGS.models_dir) or not os.path.isdir(FLAGS.models_dir):
        print('check models_dir (not a dir or not exist): %s' % FLAGS.models_dir)
        exit()

    for d in os.listdir(FLAGS.models_dir):
        path = os.path.join(FLAGS.models_dir, d)
        files = []
        for f in os.listdir(path):
            if f.endswith('.index'):
                step_num = int(f.split('-')[1].split('.')[0])
                files.append({'name': f, 'steps': step_num})

        largest = sorted(files, key=itemgetter('steps'), reverse=True)[0]['name']

        ckpt2use = os.path.join(path, largest)
        # ckpt2use = tf.train.latest_checkpoint(path)
        print('evaluating %s on %s', ckpt2use, FLAGS.val_tfr)

        evaluate(ckpt2use, FLAGS.val_tfr)
