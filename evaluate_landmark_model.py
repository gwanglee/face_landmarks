'''Evaluate landmark models: compute average error and save prediction images in mosaic
'''

import tensorflow as tf
import data_landmark as data
import net
import os
import numpy as np
import cv2
from operator import itemgetter
from copy import copy

slim = tf.contrib.slim

tf.app.flags.DEFINE_string('tfrecord', '/home/gglee/Data/160v5.0322.val.tfrecord', '.tfrecord for validation')
tf.app.flags.DEFINE_string('models_dir', '/home/gglee/Data/Landmark/train/0408', 'where trained models are stored')

FLAGS = tf.app.flags.FLAGS

def load_settings(ckpt_path):
    path_setting = os.path.join(os.path.dirname(ckpt_path), 'train_setting.txt')
    assert os.path.exists(path_setting) and os.path.isfile(
        path_setting), 'train_setting.txt not exist for [%s]' % ckpt_path

    normalizer_fn = None
    normalizer_params = {}
    depth_multiplier = 1.0
    depth_gamma = 1.0
    is_color = True

    with open(path_setting, 'r') as rf:
        for l in rf:
            if 'is_color' in l:
                _, is_color = l.split(':')
            elif 'use_batch_norm' in l:
                _, bn_val = l.split(':')
                if bn_val.strip() == 'True':
                    normalizer_fn = slim.batch_norm
                    normalizer_params = {'is_training': False}
            elif 'depth_multiplier' in l:
                _, dm_val = l.split(':')
                depth_multiplier = float(dm_val.strip())
            elif 'depth_gamma' in l:
                _, dg_val = l.split(':')
                depth_gamma = float(dg_val.strip())

    return {'normalizer_fn': normalizer_fn, 'normalizer_params': normalizer_params, 'depth_multiplier': depth_multiplier,
            'depth_gamma': depth_gamma, 'is_color': is_color }


def paste_mosaic_patch(mosaic, index, patch, points, predicts, error, size=8):
    PX = 4
    PY = 12
    FONT_SCALE = 0.85

    cur_img = copy(patch)
    y = index / size
    x = index % size

    error = error*100

    for p in range(68):
        ix, iy = p * 2, p * 2 + 1

        px = int(points[ix] * 56.0 + 0.5)
        py = int(points[iy] * 56.0 + 0.5)
        cv2.rectangle(cur_img, (px - 1, py - 1), (px + 1, py + 1), (0, 0, 255), 1)

        px = int(predicts[ix] * 56 + 0.5)
        py = int(predicts[iy] * 56 + 0.5)
        cv2.line(cur_img, (px - 1, py + 1), (px + 1, py - 1), (0, 255, 0), 1)
        cv2.line(cur_img, (px - 1, py - 1), (px + 1, py + 1), (0, 255, 0), 1)

        cv2.putText(cur_img, '%.2f' % error, (PX-1, PY), cv2.FONT_HERSHEY_PLAIN, FONT_SCALE, (255, 255, 255))
        cv2.putText(cur_img, '%.2f' % error, (PX+1, PY), cv2.FONT_HERSHEY_PLAIN, FONT_SCALE, (255, 255, 255))
        cv2.putText(cur_img, '%.2f' % error, (PX, PY-1), cv2.FONT_HERSHEY_PLAIN, FONT_SCALE, (255, 255, 255))
        cv2.putText(cur_img, '%.2f' % error, (PX, PY+1), cv2.FONT_HERSHEY_PLAIN, FONT_SCALE, (255, 255, 255))
        cv2.putText(cur_img, '%.2f' % error, (PX, PY), cv2.FONT_HERSHEY_PLAIN, FONT_SCALE, (0, 0, 0))

        mosaic[56*y:56*(y+1), 56*x:56*(x+1), :] = cur_img


def evaluate(ckpt_path, tfr_path):

    # load train_setting
    settings = load_settings(ckpt_path)
    normalizer_fn = settings['normalizer_fn']
    normalizer_params = settings['normalizer_params']
    depth_multiplier = settings['depth_multiplier']
    depth_gamma = settings['depth_gamma']
    is_color = settings['is_color'].strip() == 'True'

    count_records = data.get_tfr_record_count(tfr_path)
    dataset = data.load_tfrecord(tfr_path, batch_size=64, num_parallel_calls=16, is_color=is_color)
    iterator = dataset.make_initializable_iterator()

    BATCH_WIDTH = 8
    BATCH_SIZE = BATCH_WIDTH*BATCH_WIDTH
    NUM_ITER = int(count_records/BATCH_SIZE)

    KEEP_WIDTH = 10
    MAX_KEEP = KEEP_WIDTH*KEEP_WIDTH

    CH = 3 if is_color else 1

    bests = []
    worsts = []

    image, points = iterator.get_next()

    with tf.variable_scope('model') as scope:
        predicts, _ = net.lannet(image, depth_mul=depth_multiplier, depth_gamma=depth_gamma,
                                 normalizer_fn=normalizer_fn, normalizer_params=normalizer_params)

    with tf.Session() as sess:
        init = [tf.initialize_all_variables(), iterator.initializer]
        sess.run(init)

        saver = tf.train.Saver()
        saver.restore(sess, ckpt_path)

        errs = []

        with open(os.path.join(os.path.dirname(ckpt_path), 'err.csv'), 'w') as ewf:
            for i in range(68):
                ewf.write('x%d, y%d' % (i, i))
                if i < 68-1:
                    ewf.write(', ')
                else:
                    ewf.write('\n')

            for i in range(NUM_ITER):
                img, pts, prs = sess.run([image, points, predicts])
                img = np.asarray((img + 1.0)*255.0/2.0, dtype=np.uint8)
                mosaic = np.zeros((56*BATCH_WIDTH, 56*BATCH_WIDTH, 3), dtype=np.uint8)

                perr = np.subtract(pts, prs)
                for pes in perr:
                    for j, pe in enumerate(pes):
                        ewf.write('%f' % abs(pe))
                        if j < len(pes)-1:
                            ewf.write(', ')
                        else:
                            ewf.write('\n')

                for y in range(BATCH_WIDTH):
                    for x in range(BATCH_WIDTH):
                        pos = y*BATCH_WIDTH + x
                        cur_img = img[pos, :, :, :]

                        if not is_color:
                            cur_img = cv2.cvtColor(cur_img, cv2.COLOR_GRAY2BGR)

                        cur_pts = pts[pos]
                        cur_prs = prs[pos]

                        diff = cur_pts - cur_prs
                        err = 0

                        for p in range(68):
                            ix, iy = p * 2, p * 2 + 1
                            e = np.sqrt(diff[ix]*diff[ix] + diff[iy]*diff[iy])
                            err += e

                        err /= 68

                        paste_mosaic_patch(mosaic, pos, cur_img, cur_pts, cur_prs, err)
                        errs.append(err)

                        bests.append({'err': err, 'img': copy(cur_img), 'pts': copy(pts[pos]), 'prs': copy(prs[pos])})
                        worsts.append({'err': err, 'img': copy(cur_img), 'pts': copy(pts[pos]), 'prs': copy(prs[pos])})

                        if len(bests) > MAX_KEEP:
                            bests = sorted(bests, key=itemgetter('err'), reverse=False)[:MAX_KEEP]

                        if len(worsts) > MAX_KEEP:
                            worsts = sorted(worsts, key=itemgetter('err'), reverse=True)[:MAX_KEEP]

                cv2.imshow("mosaic", mosaic)
                img_save_path =('%s_%03d.jpg' % (ckpt_path, i))
                cv2.imwrite(img_save_path, mosaic)
                cv2.waitKey(1000)

        err_total = np.mean(errs)
        cv2.imshow("mosaic", mosaic)
        img_save_path = ('%s_%03d.jpg' % (ckpt_path, i))
        cv2.imwrite(img_save_path, mosaic)
        cv2.waitKey(1000)

        # make mosaic images for best & worst
        img_bests = np.zeros((56 * KEEP_WIDTH, 56 * KEEP_WIDTH, 3), dtype=np.uint8)
        img_worsts = np.zeros((56 * KEEP_WIDTH, 56 * KEEP_WIDTH, 3), dtype=np.uint8)

        for i in range(MAX_KEEP):
            paste_mosaic_patch(img_bests, i, bests[i]['img'], bests[i]['pts'], bests[i]['prs'], bests[i]['err'],
                               KEEP_WIDTH)
            paste_mosaic_patch(img_worsts, i, worsts[i]['img'], worsts[i]['pts'], worsts[i]['prs'], worsts[i]['err'],
                               KEEP_WIDTH)

        cv2.imshow('bests', img_bests)
        cv2.imshow('worsts', img_worsts)
        cv2.imwrite('%s_bests.jpg' % ckpt_path, img_bests)
        cv2.imwrite('%s_worsts.jpg' % ckpt_path, img_worsts)

        cv2.waitKey(100)

        return err_total


if __name__=='__main__':

    if not os.path.exists(FLAGS.models_dir) or not os.path.isdir(FLAGS.models_dir):
        print('check models_dir (not a dir or not exist): %s' % FLAGS.models_dir)
        exit()

    with open(os.path.join(FLAGS.models_dir, 'eval.txt'), 'w') as wf:
        dirs = []
        for d in os.listdir(FLAGS.models_dir):
            path = os.path.join(FLAGS.models_dir, d)
            if os.path.isdir(path):
                dirs.append(path)

        dirs = sorted(dirs)

        for path in dirs:
            files = []
            for f in os.listdir(path):
                if f.endswith('.index'):
                    step_num = int(f.split('-')[1].split('.')[0])
                    files.append({'name': os.path.splitext(f)[0], 'steps': step_num})

            if len(files) == 0:
                continue

            largest = sorted(files, key=itemgetter('steps'), reverse=True)[0]['name']

            with tf.Graph().as_default():
                ckpt2use = os.path.join(path, largest)
                err = evaluate(ckpt2use, FLAGS.tfrecord)
                print('eval err = %.4f on \'%s\' and \'%s\'' % (err*100, ckpt2use, FLAGS.tfrecord))
                wf.write('%s\t%s\t%f\n' % (os.path.basename(path), largest, err))