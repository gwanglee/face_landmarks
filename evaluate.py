import tensorflow as tf
import net
import os
import numpy as np
import cv2
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
    return normed, pts


def evaluate(ckpt_path, tfr_path):
    count_records = 0
    for record in tf.python_io.tf_record_iterator(tfr_path):
        count_records += 1

    BATCH_WIDTH = 8
    BATCH_SIZE = BATCH_WIDTH*BATCH_WIDTH
    NUM_ITER = int(count_records/BATCH_SIZE)

    dataset = tf.data.TFRecordDataset(tfr_path)
    dataset = dataset.map(_parse_function, num_parallel_calls=16)
    dataset = dataset.batch(BATCH_SIZE)
    dataset.prefetch(buffer_size=BATCH_SIZE)
    iterator = dataset.make_initializable_iterator()

    image, points = iterator.get_next()
    predicts, _ = net.lannet(image, is_training=False)

    with tf.Session() as sess:
        init = [tf.initialize_all_variables(), iterator.initializer]
        sess.run(init)

        for i in range(NUM_ITER):
            img, pts, prs = sess.run([image, points, predicts])
            img = np.asarray((img + 1.0)*255.0/2.0, dtype=np.uint8)
            mosaic = np.zeros((56*BATCH_WIDTH, 56*BATCH_WIDTH, 3), dtype=np.uint8)

            for y in range(BATCH_WIDTH):
                for x in range(BATCH_WIDTH):
                    pos = y*BATCH_WIDTH + x
                    cur_img = img[pos, :, :, :]
                    cur_pts = pts[pos]
                    cur_prs = prs[pos]

                    for p in range(68):
                        px = int(cur_pts[p * 2]*56)
                        py = int(cur_pts[p * 2 + 1]*56)
                        cv2.circle(cur_img, (px, py), 2, (0, 0, 255), 1)

                        px = int(cur_prs[p * 2] * 56)
                        py = int(cur_prs[p * 2 + 1] * 56)
                        cv2.circle(cur_img, (px, py), 2, (0, 255, 0), 1)

                    mosaic[56*y:56*(y+1), 56*x:56*(x+1), :] = cur_img

            cv2.imshow("mosaic", mosaic)
            cv2.waitKey(100)


if __name__=='__main__':

    if not os.path.exists(FLAGS.models_dir) or not os.path.isdir(FLAGS.models_dir):
        print('check models_dir (not a dir or not exist): %s' % FLAGS.models_dir)
        exit()

    for d in os.listdir(FLAGS.models_dir):
        path = os.path.join(FLAGS.models_dir, d)
        if not os.path.isdir(path):
            continue

        files = []
        for f in os.listdir(path):
            if f.endswith('.index'):
                step_num = int(f.split('-')[1].split('.')[0])
                files.append({'name': f, 'steps': step_num})

        largest = sorted(files, key=itemgetter('steps'), reverse=True)[0]['name']

        ckpt2use = os.path.join(path, largest)
        print('evaluating %s on %s' % (ckpt2use, FLAGS.val_tfr))

        evaluate(ckpt2use, FLAGS.val_tfr)
