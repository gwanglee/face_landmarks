import tensorflow as tf
import data_landmark as data
import net
import os
import numpy as np
import cv2
from operator import itemgetter

slim = tf.contrib.slim

tf.app.flags.DEFINE_string('tfrecord', '/home/gglee/Data/160v5.0322.val.tfrecord', '.tfrecord for validation')
tf.app.flags.DEFINE_string('models_dir', '/home/gglee/Data/Landmark/train/0408', 'where trained models are stored')

FLAGS = tf.app.flags.FLAGS

def evaluate(ckpt_path, tfr_path):

    # load train_setting
    path_setting = os.path.join(os.path.dirname(ckpt_path), 'train_setting.txt')
    assert os.path.exists(path_setting) and os.path.isfile(
        path_setting), 'train_setting.txt not exist for [%s]' % ckpt_path

    normalizer_fn = None
    normalizer_params = {}
    depth_multiplier = 1
    is_color = True

    with open(path_setting, 'r') as rf:
        for l in rf:
            if 'is_color' in l:
                _, is_color = l.split(':')
            if 'use_batch_norm' in l:
                _, bn_val = l.split(':')
                if bool(bn_val):
                    normalizer_fn = slim.batch_norm
                    normalizer_params = {'is_training': False}
                    # print(normalizer_fn, normalizer_params)
            elif 'depth_multiplier' in l:
                _, dm_val = l.split(':')
                depth_multiplier = int(dm_val)

    count_records = data.get_tfr_record_count(tfr_path)
    dataset = data.load_tfrecord(tfr_path, batch_size=64, num_parallel_calls=16, is_color=is_color)
    iterator = dataset.make_initializable_iterator()

    BATCH_WIDTH = 8
    BATCH_SIZE = BATCH_WIDTH*BATCH_WIDTH
    NUM_ITER = int(count_records/BATCH_SIZE)

    image, points = iterator.get_next()

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
                    if not is_color:
                        cur_img = cv2.cvtColor(cur_img, cv2.COLOR_GRAY2BGR)

                    cur_pts = pts[pos]
                    cur_prs = prs[pos]
                    # print(cur_pts)
                    # print(cur_prs)

                    diff = cur_pts - cur_prs
                    err = 0

                    for p in range(68):
                        ix, iy = p * 2, p * 2 + 1

                        px = int(cur_pts[ix]*56.0+0.5)
                        py = int(cur_pts[iy]*56.0+0.5)
                        cv2.rectangle(cur_img, (px-1, py-1), (px+1, py+1), (0, 0, 255), 1)

                        # px = int(cur_prs[ix] * 28 + 28)
                        # py = int(cur_prs[iy] * 28 + 28)
                        px = int(cur_prs[ix] * 56 + 0.5)
                        py = int(cur_prs[iy] * 56 + 0.5)
                        cv2.line(cur_img, (px - 1, py + 1), (px + 1, py - 1), (0, 255, 0), 1)
                        cv2.line(cur_img, (px - 1, py - 1), (px + 1, py + 1), (0, 255, 0), 1)
                        e = np.sqrt(diff[ix]*diff[ix] + diff[iy]*diff[iy])
                        err += e

                    err /= 68
                    errs.append(err)
                    cv2.putText(cur_img, '%.2f' % err, (PX, PY), cv2.FONT_HERSHEY_PLAIN, FONT_SCALE, (255, 255, 255))
                    cv2.putText(cur_img, '%.2f' % err, (PX-1, PY-1), cv2.FONT_HERSHEY_PLAIN, FONT_SCALE, (0, 0, 0))

                    mosaic[56*y:56*(y+1), 56*x:56*(x+1), :] = cur_img

            err_total = np.mean(errs)
            cv2.imshow("mosaic", mosaic)
            img_save_path =('%s_%03d.jpg' % (ckpt_path, i))
            cv2.imwrite(img_save_path, mosaic)
            cv2.waitKey(1000)

        return err_total


if __name__=='__main__':

    if not os.path.exists(FLAGS.models_dir) or not os.path.isdir(FLAGS.models_dir):
        print('check models_dir (not a dir or not exist): %s' % FLAGS.models_dir)
        exit()

    with open(os.path.join(FLAGS.models_dir, 'eval.txt'), 'w') as wf:
        for d in os.listdir(FLAGS.models_dir):
            path = os.path.join(FLAGS.models_dir, d)
            if not os.path.isdir(path):
                continue

            files = []
            for f in os.listdir(path):
                if f.endswith('.index'):
                    step_num = int(f.split('-')[1].split('.')[0])
                    files.append({'name': os.path.splitext(f)[0], 'steps': step_num})

            if len(files) == 0:
                continue

            largest = sorted(files, key=itemgetter('steps'), reverse=True)[0]['name']

            ckpt2use = os.path.join(path, largest)
            err = evaluate(ckpt2use, FLAGS.tfrecord)
            print('eval err = %.2f on \'%s\' and \'%s\'' % (err, ckpt2use, FLAGS.tfrecord))
            wf.write('%s\t%s\t%f\n' % (os.path.basename(path), largest, err))