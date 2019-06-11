# -*- coding: utf-8 -*-

"""Evaluate detection result on wider face db

This module evaluates face detection result on WiderFaceDB. If the detection resutls
contains confidence, it will produce ROC curve. If not, only recall and precision
will be obtained.

Example:
    python eval_wider_dets.py \
        --det_root [or det_dir]=/dataset/root/wider_face \
        --gt_path=/path/to/gt.txt \
        --image_dir=/where/images/are/stored/'images'
        --steps=24
        --recompute=False

Attributes:
    det_dir: where detection results (.txt) are stored. 'det_dir' contains 61 sub-foldres
        of events
    image_dir: where the original images are stored
    gt_path: file path of the ground truth text file
    steps: number of bins for ROC curve

Todo:
    * implement image verification
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import sys
from pprint import pprint

from operator import itemgetter
import tensorflow as tf

sys.path.append('..')
from data.widerface_explorer import wider_face_db

import pickle
import matplotlib.pyplot as plt

flags = tf.app.flags
flags.DEFINE_string('det_root', '', 'where detection results from'
                    'various algorithms are stored')
flags.DEFINE_string('det_dir', '', 'where detection results from'
                    'one algorithm are stored')
flags.DEFINE_string('image_dir', '',
                    'Directory containing checkpoints to evaluate, typically '
                    'set to `train_dir` used in the training job.')
flags.DEFINE_string('gt_path', '',
                    'Directory containing checkpoints to evaluate, typically '
                    'set to `train_dir` used in the training job.')
flags.DEFINE_integer('steps', 31,
                    'Directory containing checkpoints to evaluate, typically '
                    'set to `train_dir` used in the training job.')
flags.DEFINE_bool('recompute', True, 'If False, no evaluation for sub-dirs which'
                    'already has result' 'If True, delete existing evalutations and'
                    'recompute them')

FLAGS = tf.app.flags.FLAGS

DET_ROOT = FLAGS.det_root
DET_DIR = FLAGS.det_dir
GT_PATH = FLAGS.gt_path
N_STEPS = FLAGS.steps
IMAGE_DIR = FLAGS.image_dir
RECOMPUTE = FLAGS.recompute

def find_conf_range(det_dir):
    """ Find confidence range (min, max values) for a given detection

    Args:
        det_dir: directory root where detection results are stored
    Returns:
        [min, max] of confidence values. None if all confidences have the same level
    """
    min_c = 1.0
    max_c = 0.0

    for d in os.listdir(det_dir):
        sub_dir = os.path.join(det_dir, d)
        if os.path.isdir(sub_dir):
            for f in os.listdir(sub_dir):
                filepath = os.path.join(sub_dir,f)
                if os.path.isfile(filepath) and filepath.endswith('txt'):
                    with open(filepath, 'r') as rf:
                        header = rf.readline()
                        num_dets = int(rf.readline())
                        for i in range(num_dets):
                            x, y, w, h, c = rf.readline().split()
                            x, y, w, h, c = float(x), float(y), float(w), float(h), float(c)
                            if c > max_c:
                                max_c = c
                            if c < min_c:
                                min_c = c

    return min_c, max_c

def get_threshold_levels(min_conf, max_conf, steps):
    """ Get threshold levels by deviding [min, max] range uniformly

    Return:
        Array of threshold levels for evaluation
    """

    if min_conf >= max_conf:
        return np.asarray([0.0])

    bins = np.arange(steps)                     # 0 ~ (steps-1)
    vals = bins / (steps-1)                     # float(max(bins))  # 0.0 ~ 1.0
    val_range = max_conf - min_conf
    norm_vals = (vals*val_range + min_conf)     # min_conf ~ max_conf

    return norm_vals

def is_overlap(bb1, bb2):
    """Check if two boexes are overlap each other

    Return:
        True if overlap, False if not
    """
    l1, t1, r1, b1 = bb1['x'], bb1['y'], bb1['x']+bb1['w'], bb1['y']+bb1['h']
    l2, t2, r2, b2 = bb2['x'], bb2['y'], bb2['x']+bb2['w'], bb2['y']+bb2['h']

    if r1 > l2 and r2 > l1 and b2 > t1 and b1 > t2:
        return True
    else:
        return False

def get_iou(bb1, bb2):
    """Get IOU ratio for two boxes

    Return:
        0 if bb1 and bb2 do not overlap. IOU (in range of [0, 1]) if they overlap.
    """
    if not is_overlap(bb1, bb2):
        return 0

    l1, t1, r1, b1 = bb1['x'], bb1['y'], bb1['x']+bb1['w'], bb1['y']+bb1['h']
    l2, t2, r2, b2 = bb2['x'], bb2['y'], bb2['x']+bb2['w'], bb2['y']+bb2['h']

    xa, ya = max(l1, l2), max(t1, t2)
    xb, yb = min(r1, r2), min(b1, b2)

    inter_area = abs((xb-xa+1)*(yb-ya+1))
    assert xb>=xa, "(xb, xa) = (%f, %f)"%(xb, xa)
    assert yb>=ya, "(yb, ya) = (%f, %f)"%(yb, ya)

    area_1 = (bb1['w']+1)*(bb1['h']+1)
    area_2 = (bb2['w']+1)*(bb2['h']+1)

    iou = inter_area / float(area_1 + area_2 - inter_area)
    assert iou >= 0.0, "(%f, %f, %f, %f), (%f, %f, %f, %f) -> %f"%(bb1['x'], bb1['y'], bb1['w'], bb1['h'], \
                                                                bb2['x'], bb2['y'], bb2['w'], bb2['h'], iou)

    return iou

def load_detections(det_dir):
    """Load detection results stored text files.
    Returns are detections sorted in descending order by confidence.
    """
    detections = []

    for curd in os.listdir(det_dir):
        sub_dir = os.path.join(det_dir, curd)
        if os.path.isdir(sub_dir):

            for f in os.listdir(sub_dir):
                filepath = os.path.join(sub_dir, f)

                if os.path.isfile(filepath) and filepath.endswith('txt'):
                    # find detection and read it
                    with open(filepath, 'r') as rf:
                        # filename = rf.readline().strip('\n')
                        # dets2read = int(rf.readline())
                        data = []

                        for l in rf.readlines():
                        # for i in range(dets2read):
                        #     x, y, w, h, c = rf.readline().split()
                            x, y, w, h, c = l.split()
                            data.append({'x':float(x), 'y':float(y), 'w':float(w)-float(x), 'h':float(h)-float(y), 'c':float(c)})
                    #detections.append({'filename': filename, 'data': data})
                    detections.append({'filename': os.path.splitext(f)[0], 'data': data})

    return detections


def get_recall_and_precision(wdb, dets, th_conf, th_iou):
    """Compute recall and precision for given detection results
    """
    hit_total, false_total, miss_total = 0, 0, 0

    for idx in range(wdb.get_image_count()):
        data = wdb.get_annos_by_image_index(idx)
        img_path = data['image_path']
        fname = os.path.splitext(os.path.basename(img_path))[0]

        dets4fname = []

        for d in dets:
            if d['filename'] == fname:  # we have only ONE corresponding element
                dets4fname = d['data']
                break

        dets2compare = [e for e in dets4fname if e['c'] >= th_conf]          # detections having confidence larger than threshold only

        hit, false, miss = count_hits(data['annos'], dets2compare, th_iou)
        hit_total += hit
        false_total += false
        miss_total += miss

    recall = hit_total/float(hit_total+miss_total+0.000001)
    precision = hit_total/float(hit_total+false_total+0.000001)

    return recall, precision


def count_hits(gt, dt, th_iou):
    num_dets = 0
    count_hit = 0

    if len(dt) > 0 and len(gt) > 0:
        hit_mat = np.zeros((len(dt), len(gt)), dtype=float)  # count hit, miss, false   dt x gt

        for i, d in enumerate(dt):
            for j, a in enumerate(gt):
                iou = get_iou(d, a)
                # print(iou)
                if iou >= th_iou:
                    hit_mat[i, j] = iou
                    num_dets += 1

        while np.max(hit_mat) > 0:
            max_idx = np.argmax(hit_mat)
            r, c = np.unravel_index(max_idx, hit_mat.shape)
            hit_mat[r, :] = 0.0
            hit_mat[:, c] = 0.0
            #hit_mat[r, c] = 0.01
            count_hit += 1

        hit, false, miss = count_hit, len(dt)-count_hit, len(gt)-count_hit
        assert hit >= 0 and false >= 0 and miss >=0, "hit:%d, false:%d, miss:%d"%(hit, false, miss)

    elif len(dt) == 0 and len(gt) == 0:
        return 0, 0, 0

    elif len(dt) > 0 and len(gt) == 0:
        return 0, len(dt), 0

    elif len(dt) == 0 and len(gt) > 0:
        return 0, 0, len(gt)

    return hit, false, miss


def eval_single_det(wdb, det_dir, nsteps):
    roc = []
    detections = load_detections(det_dir)

    iou_th = 0.5
    if det_dir.find('LBP') > 0:
        iou_th = 0.3

    print('starting evaluation for %s' % det_dir)

    if nsteps == 0:
        dets = [d for d in detections if d['c'] > 0.5]
        r, p = get_recall_and_precision(wdb, dets, 0.5, iou_th)
        roc.append({'recall': r, 'precision': p})
    else:
        min_c, max_c = 0.001, 0.999#find_conf_range(det_dir)
        levels = get_threshold_levels(min_c, max_c, N_STEPS)

        for l in levels:
            r, p = get_recall_and_precision(wdb, detections, l, iou_th)
            roc.append({'recall': r, 'precision': p})
            print('threshold: %.2f, recall: %.2f, precision: %.2f'%(l, r, p))

    return roc


def safe_save(save_path, obj2save):
    if os.path.exists(save_path):
        os.remove(save_path)
    with open(save_path, 'wb') as handle:
        pickle.dump(obj2save, handle, protocol=pickle.HIGHEST_PROTOCOL)


def compute_ap(roc):
    if len(roc) == 1:
        return 0
    else:
        NUM_SAMPLES = 10
        rp = np.linspace(0.05, 0.95, NUM_SAMPLES)
        roc = sorted(roc, key=itemgetter('recall')) # sort by recall, ascending order

        precision = [p['precision'] for p in roc]
        recall = [r['recall'] for r in roc]

        psamples = []

        for r in rp:    # find closest recall point
            diffs = np.abs(recall - r)
            idx = np.argmin(diffs)
            if idx == 0 and r < recall[0]:
                psamples.append(precision[idx])
            elif idx == len(recall)-1 and r > recall[idx]:
                psamples.append(precision[idx])
            else:
                if r < recall[idx]:
                    left = idx-1
                    right = idx
                else:
                    left = idx
                    right = idx+1
                p = ((recall[right]-r)*precision[left]+(r-recall[left])*precision[right]) / (recall[right]-recall[left])
                psamples.append(p)

        return np.mean(psamples)
        #return np.mean(prec)


def load(load_path):
    '''if os.path.exists(load_path):
        return pickle.load(load_path)'''
    with open(load_path, 'rb') as handle:
        return pickle.load(handle)

def main(_):
    wdb = wider_face_db(IMAGE_DIR, GT_PATH)

    if DET_DIR:
        pkl_path = DET_DIR + '.pkl'
        if not RECOMPUTE and os.path.exists(pkl_path) and os.path.isfile(pkl_path):
            print('Result already exist(%s)'%pkl_path)
        else:
            roc = eval_single_det(wdb, DET_DIR, N_STEPS)
            safe_save(pkl_path, roc)

            pprint(roc)

            fig = plt.figure()
            recall = [x['recall'] for x in roc]
            prec = [x['precision'] for x in roc]
            plt.plot(recall, prec, '-x')

            ax = fig.gca()
            ax.set_xticks(np.arange(0, 1, 0.1))
            ax.set_yticks(np.arange(0, 1, 0.1))
            ax.grid(which='both')

            plt.show()

            print('AP: %.2f' % compute_ap(roc))


    elif DET_ROOT:
        results = []
        for sub_dir in os.listdir(DET_ROOT):
            if os.path.isdir(os.path.join(DET_ROOT, sub_dir)) is False:
                continue

            pkl_path = os.path.join(DET_ROOT, sub_dir + '.pkl')

            if not RECOMPUTE and os.path.exists(pkl_path) and os.path.isfile(pkl_path):
                print('Result exist for %s. Skip w/o re-evaluation' % (sub_dir))
            else:
                roc = eval_single_det(wdb, os.path.join(DET_ROOT, sub_dir), N_STEPS)
                safe_save(pkl_path, roc)
                print('ROC computed for %s' % sub_dir)

            roc = load(pkl_path)
            ap = compute_ap(roc)
            results.append({'name': sub_dir, 'roc': roc, 'ap': ap})

        pprint(results)

        roc_sorted = sorted(results, key=itemgetter('ap'), reverse=True)

        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        markers = ['o', 'x', '^', 's', '*', 'v', 'd', 'h', '|', '_']
        lines = ['-', '--', '-.', ':']

        fig = plt.figure(1)
        ax = fig.gca()

        for i, r in enumerate(roc_sorted):
            recall = [x['recall'] for x in r['roc']]
            prec = [x['precision'] for x in r['roc']]
            #line_spec = colors[i%len(colors)] + markers[i%len(markers)] + lines[i%len(lines)]
            line_spec = colors[i % len(colors)] + lines[i % len(lines)]
            if r['ap'] > 0.0:
                str = '%s(%.2f)' % (r['name'], r['ap'])
            else:
                str = '%s' % r['name']
            plt.plot(recall, prec, line_spec, label=str)

        ax.set_xticks(np.arange(0, 1, 0.1))
        ax.set_yticks(np.arange(0, 1, 0.1))
        ax.grid(which='both')

        plt.legend(loc=0, borderaxespad=1., fontsize='small')
        plt.grid()
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.show()

        fig.savefig(os.path.join(DET_ROOT, 'RoC.png'))

    print('Evaluation Finished')

if __name__ == '__main__':
    tf.app.run()

    #python eval_wider_dets.py --img_dir=/Volumes/Data/FaceDetectionDB/WiderFace/WIDER_val_0.02/ --gt_path=/Volumes/Data/FaceDetectionDB/WiderFace/wider_face_split/wider_face_val_bbx_gt_0.02.txt  --det_dir=/Volumes/Data/FaceDetectionDB/WiderFace/Result/split_0.02/WIDER_VAL_0.02_MTCNN_AICLOUD/ --steps=0