import tensorflow as tf
import inference
import net
import os
import numpy as np
import cv2
from operator import itemgetter

slim = tf.contrib.slim

tf.app.flags.DEFINE_string('tfrecord', '/home/gglee/Data/160v5.0322.val.tfrecord', '.tfrecord for validation')
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
    for r in tf.python_io.tf_record_iterator(tfr_path):
        count_records += 1

    BATCH_WIDTH = 8
    BATCH_SIZE = BATCH_WIDTH*BATCH_WIDTH
    NUM_ITER = int(count_records/BATCH_SIZE)

    dataset = tf.data.TFRecordDataset(tfr_path)
    dataset = dataset.map(_parse_function, num_parallel_calls=16)
    dataset = dataset.batch(BATCH_SIZE)
    dataset.prefetch(buffer_size=BATCH_SIZE)
    iterator = dataset.make_initializable_iterator()

    path_setting = os.path.join(os.path.dirname(ckpt_path), 'train_setting.txt')
    assert os.path.exists(path_setting) and os.path.isfile(path_setting), 'train_setting.txt not exist for [%s]' % ckpt_path

    normalizer_fn = None
    normalizer_params = {}
    depth_multiplier = 1

    with open(path_setting, 'r') as rf:
        for l in rf:
            print(l)
            if 'use_batch_norm' in l:
                _, bn_val = l.split(':')
                if float(bn_val) > 0.0:
                    normalizer_fn = slim.batch_norm
                    normalizer_params = {'is_training': False}
                    print(normalizer_fn, normalizer_params)
            elif 'depth_multiplier' in l:
                _, dm_val = l.split(':')
                depth_multiplier = int(dm_val)
                print('\t>>>depth_multiplier: %d' % depth_multiplier)

    image, points = iterator.get_next()

    # ld = inference.Classifier(56, ckpt_path, depth_multiplier=depth_multiplier,
    #                           normalizer_fn=normalizer_fn, normalizer_params=normalizer_params)

    with tf.variable_scope('model') as scope:
        predicts, _ = net.lannet(image, is_training=False, depth_mul=depth_multiplier,
                             normalizer_fn=normalizer_fn, normalizer_params=normalizer_params)

    with tf.Session() as sess:
        init = [tf.initialize_all_variables(), iterator.initializer]
        sess.run(init)

        saver = tf.train.Saver()
        saver.restore(sess, ckpt_path)

        errs = []

        for i in range(NUM_ITER):
            # img, pts = sess.run([image, points])
            # prs = ld.predict(img)

            img, pts, prs = sess.run([image, points, predicts])
            img = np.asarray((img + 1.0)*255.0/2.0, dtype=np.uint8)
            mosaic = np.zeros((56*BATCH_WIDTH, 56*BATCH_WIDTH, 3), dtype=np.uint8)

            PX = 28
            PY = 10
            FONT_SCALE = 0.85

            for y in range(BATCH_WIDTH):
                for x in range(BATCH_WIDTH):
                    pos = y*BATCH_WIDTH + x
                    cur_img = img[pos, :, :, :]
                    cur_pts = pts[pos]
                    cur_prs = prs[pos]
                    # print(cur_pts)
                    # print(cur_prs)

                    diff = cur_pts - cur_prs
                    err = 0

                    for p in range(68):
                        ix, iy = p * 2, p * 2 + 1

                        px = int(cur_pts[ix]*56)
                        py = int(cur_pts[iy]*56)
                        cv2.circle(cur_img, (px, py), 2, (0, 0, 255), 1)

                        # px = int(cur_prs[ix] * 28 + 28)
                        # py = int(cur_prs[iy] * 28 + 28)
                        px = int(cur_prs[ix] * 56)
                        py = int(cur_prs[iy] * 56)
                        cv2.circle(cur_img, (px, py), 2, (0, 255, 0), 1)

                        e = np.sqrt(diff[ix]*diff[ix] + diff[iy]*diff[iy])
                        err += e

                    err /= 68
                    errs.append(err)
                    cv2.putText(cur_img, '%.2f' % err, (PX, PY), cv2.FONT_HERSHEY_PLAIN, FONT_SCALE, (255, 255, 255))
                    cv2.putText(cur_img, '%.2f' % err, (PX-1, PY-1), cv2.FONT_HERSHEY_PLAIN, FONT_SCALE, (0, 0, 0))

                    mosaic[56*y:56*(y+1), 56*x:56*(x+1), :] = cur_img

            err_total = np.mean(errs)
            print('error: %.2f' % err_total)
            cv2.imshow("mosaic", mosaic)
            cv2.waitKey(1000)


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
                print('found model: %s' % f)
                files.append({'name': os.path.splitext(f)[0], 'steps': step_num})

        if len(files) == 0:
            continue

        largest = sorted(files, key=itemgetter('steps'), reverse=True)[0]['name']

        ckpt2use = os.path.join(path, largest)
        print('evaluating %s on %s' % (ckpt2use, FLAGS.tfrecord))

        evaluate(ckpt2use, FLAGS.tfrecord)
        exit()
