#-*- coding: utf-8 -*-


"""Create a sub-set of Wider Face DB by given criteria (e.g. minimum face size)

Add more explanation

Example usage: (need to update)
    python split_wider_face.py \
        --image_dir=/dataset/root/wider_face/whatever/'images' \
        --gt_path=/dataset/root/wider_face/somewhere/gt.txt \
        --output_image_dir=/dataset/where/to/save/'images' \
        --output_gt_path=/path/to/resulting/gt.txt
        --min_size=relative_size_threshold

Todo:
    * add directory check in main func.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from shutil import copyfile

import os
import sys
import cv2

import tensorflow as tf
import widerface_explorer
from random import randrange
from copy import deepcopy
from operator import itemgetter

DEBUG = True
DEBUG_DISPLAY_TIME = 10

sys.path.append('/Users/gglee/Develop/models/research')
sys.path.append('/Users/gglee/Develop/models/research/slim')

tf.app.flags.DEFINE_string('image_dir', '', 'Where the source WiderFaceDB images are'
                            'located. The image_Dir contains sub-folders of event scenes')
tf.app.flags.DEFINE_string('gt_path', '', 'Filepath of the ground truth txt file')
tf.app.flags.DEFINE_string('output_image_dir', '', 'Where to store the split sets.'
                            'Resulting folder will have the same sub-folders w/ image_dir'
                            'but only images containing faces larger than min_size')
tf.app.flags.DEFINE_string('output_gt_path', '', 'Ground truth txtfile path corresponds to'
                            'dataset in output_image_dir')
tf.app.flags.DEFINE_integer('min_abs_size', 30, 'ignore faces smaller than min_size pixels')
tf.app.flags.DEFINE_float('min_rel_size', 0.06, 'ignore faces small than min_rel_size * image_size')

FLAGS = tf.app.flags.FLAGS


def find_small_and_large_faces(annos, threshold):
    '''
    return faces smaller than and larger than a given threshold
    :param annos:
    :param threshold: because we crop from the original image, threshold is in pixels
    :return:
    '''

    larges = filter(lambda x: x['w'] >= threshold, annos)
    smalls = filter(lambda x: x['w'] < threshold, annos)

    return smalls, larges


def find_smallest_and_largest_faces(annos):
    if len(annos) > 1:
        sorted_ = sorted(annos, key=lambda k: k['w'])
        return sorted_[0], sorted_[-1]
    elif len(annos) == 1:
        return annos[0], annos[0]
    else:
        return None, None


def find_bounding_box(annos):
    if annos is None or len(annos) == 0:
        return None

    l = min([a['x'] for a in annos])
    t = min([a['y'] for a in annos])
    r = max([a['x'] + a['w'] for a in annos])
    b = max([a['y'] + a['h'] for a in annos])
    return {'l': l, 't': t, 'r': r, 'b': b, 'w': r-l, 'h': b-t}


def check_overlap(bb1, bb2):
    '''
    check if two boxes overlap each other
    :param bb1:
    :param bb2:
    :return:
    '''
    if bb1 is None or bb2 is None:
        return False

    if bb1['l'] > bb2['r'] or bb1['r'] < bb2['l']:
        return False
    if bb1['t'] > bb2['b'] or bb1['b'] < bb2['t']:
        return \

    return True


def refine_annos(annos):
    '''remove annoations that not to be used: labeld as invalid or hard occlusion'''

    survive = []
    drop = []

    for a in annos:
        if a['invalid'] > 0:# or a['occlusion'] > 1 or a['blur'] > 1:
            drop.append(a)
        else:
            survive.append(a)
    return survive, drop


def check_inside(inbox, outbox):
    if inbox['l'] >= outbox['l'] and inbox['t'] >= outbox['t'] and \
        inbox['r'] <= outbox['r'] and inbox['b'] <= outbox['b']:
        return True
    else:
        return False


def refine_widerface_db(db_path, gt_path, write_db_path, write_gt_path, REL_TH):

    wdb = widerface_explorer.wider_face_db(db_path, gt_path)
    tmp_path = os.path.join(os.path.dirname(db_path), 'tmp')

    MIN_FACE_TH = REL_TH
    MAX_TRY = 256
    MIN_ASPECT_RATIO = 1.0
    MAX_ASPECT_RATIO = 1.5

    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)
        assert os.path.exists(tmp_path)

    if not os.path.exists(os.path.dirname(write_gt_path)):
        os.makedirs(os.path.dirname(write_gt_path))

    with open(write_gt_path, 'w') as wgt:
        total_images = wdb.get_image_count()

        cv2.namedWindow('image')
        cv2.moveWindow('image', 0, 0);

        for idx in range(total_images):
            data = wdb.get_annos_by_image_index(idx)

            image_path = data['image_path']
            image = cv2.imread(image_path)
            H, W = image.shape[0:2]

            print('%d: %s' % (idx, image_path))

            CROP_SX, CROP_SY = 0, 0
            CROP_W, CROP_H = W, H
            CROP_FOUND = False
            CROP_ANNOS = []
            IS_NEGATIVE = False

            annos, invannos = refine_annos(data['annos'])               # remove invalid and heavily occluded faces
            bba = find_bounding_box(annos)

            smallest, largest = find_smallest_and_largest_faces(annos)
            MIN_FRAME_SIZE = 300
            ABS_TH = int(MIN_FRAME_SIZE*MIN_FACE_TH)

            small, large = find_small_and_large_faces(annos, ABS_TH)  # find faces smaller and larger faces (small: not meet the requirement by cropping)
            bbs = find_bounding_box(small)
            bbl = find_bounding_box(large)

            '''
            1) smallest > threshold  ==>  no need to crop or modify
               else
               2) w(smallest) / w(bbox(all)) ==> bbox as the whole image
            '''
            if (smallest is None and largest is None) or len(large) == 0:
                IS_NEGATIVE = True
            elif smallest['w'] > MIN_FACE_TH * W and MIN_ASPECT_RATIO <= W/float(H) < MAX_ASPECT_RATIO:       # smallest face > threshold -> safe to use
                CROP_FOUND = True
                CROP_ANNOS = annos
                print('case 1: safe to use the original image')
            else:
                target_crop_width = smallest['w'] / MIN_FACE_TH     # maximum crop width we can take

                if bba['w'] <= target_crop_width:       # find crop that include bba
                    CROP_W = int(target_crop_width)
                    CROP_H = max(int(CROP_W*3/4), bba['h']+smallest['h'])

                    for r in range(MAX_TRY):
                        rand_begin = max(0, int(bba['l'] - (target_crop_width - bba['w'])))
                        CROP_SX = rand_begin if rand_begin == bba['l'] else randrange(rand_begin, bba['l'])

                        rand_begin = max(0, int(bba['t'] - (CROP_H - bba['h'])))
                        CROP_SY = rand_begin if rand_begin == bba['t'] else randrange(rand_begin, bba['t'])

                        if CROP_SX + CROP_W < W and CROP_SY + CROP_H < H and \
                                check_inside(bba, {'l': CROP_SX, 't': CROP_SY, 'r': CROP_SX+CROP_W, 'b': CROP_SY+CROP_H}):
                            CROP_FOUND = True
                            CROP_ANNOS = annos
                            print('case 2: crop to include all rects')
                            break

                else:                       # pick random crop that includes bbl only
                    if len(large) == 0:
                        continue

                    def check_point_in_box(x, y, box):
                        '''
                        :return: True if a point (x, y) is in box, False if box is None or (x, y) is outside of box
                        '''
                        if box is None:
                            return False

                        if box['l'] <= x <= box['r'] and box['t'] <= y <= box['b']:
                            return True
                        else:
                            return False

                    large_smallest, _ = find_smallest_and_largest_faces(large)
                    CROP_ANNOS = large

                    max_score = 0
                    for r in range(MAX_TRY):
                        for p in range(MAX_TRY):
                            cl = 0 if bbl['l'] <= 0 else randrange(0, bbl['l'])
                            ct = 0 if bbl['t'] <= 0 else randrange(0, bbl['t'])

                            if not check_point_in_box(cl, ct, bbs):
                                break

                        for p in range(MAX_TRY):
                            cr = W if bbl['r'] >= W else randrange(bbl['r'], W)
                            cb = H if bbl['b'] >= H else randrange(bbl['b'], H)

                            if not check_point_in_box(cr, cb, bbs):
                                break

                        crop_box = {'l': cl, 't': ct, 'r': cr, 'b': cb, 'w': cr-cl, 'h': cb-ct}

                        if large_smallest['w'] > crop_box['w'] * MIN_FACE_TH and crop_box['w'] > MIN_FRAME_SIZE:
                            if not check_overlap(crop_box, bbs) and check_inside(bbl, crop_box):
                                ratio = crop_box['w'] / float(crop_box['h'])

                                if MIN_ASPECT_RATIO < ratio < MAX_ASPECT_RATIO:
                                    CROP_FOUND = True
                                    rscore = max(0.0, 1-abs(0.75 - ratio))      # max at 1 when ratio = 0.75
                                    cur_score = rscore * crop_box['w'] * crop_box['h']

                                    if cur_score > max_score:
                                        CROP_SX, CROP_SY = crop_box['l'], crop_box['t']
                                        CROP_W, CROP_H = crop_box['w'], crop_box['h']
                                        max_score = cur_score
                                        print('case 3: crop to include large rects only')


            def draw_box(image, anno, color, line_width=1):
                cv2.rectangle(image, (anno['l'], anno['t']), (anno['r'], anno['b']), color, line_width)

            if DEBUG:       # draw all annotations
                image_boxes = deepcopy(image)

                if bba:     # white
                    draw_box(image_boxes, bba, (255, 255, 255), 2)
                if bbl:
                    draw_box(image_boxes, bbl, (180, 255, 180), 2)
                if bbs:
                    draw_box(image_boxes, bbs, (180, 180, 255), 2)

                for ia in invannos:             # invalids: black
                    draw_box(image_boxes, ia, (0, 0, 0), 2)
                for l in large:                 # large: green
                    draw_box(image_boxes, l, (0, 255, 0), 1)
                for s in small:                 # small: red
                    draw_box(image_boxes, s, (0, 0, 255), 1)

                if CROP_FOUND and DEBUG:      # crop_box: light blue
                    print(CROP_SX, CROP_SY, CROP_W, CROP_H)
                    cv2.rectangle(image_boxes, (CROP_SX, CROP_SY), (CROP_SX+CROP_W, CROP_SY+CROP_H), (255, 255, 0), 2)
                cv2.imshow('image', image_boxes)

            if not IS_NEGATIVE and not CROP_FOUND:
                continue

            dst_path = image_path.replace(db_path, write_db_path)
            dir_path = os.path.dirname(dst_path)

            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            CL, CT = CROP_SX, CROP_SY
            CR, CB = min(CROP_SX+CROP_W, W), min(CROP_SY+CROP_H, H)

            image_crop = image[CT:CB, CL:CR]
            cv2.imwrite(dst_path, image_crop)

            filename = os.path.basename(dst_path)
            dirname = os.path.dirname(dst_path).rsplit('/', 1)[1]
            wgt.write("%s\n" % (os.path.join(dirname, filename)))

            if CROP_FOUND:
                wgt.write("%d\n" % (len(CROP_ANNOS)))
                CH, CW = image_crop.shape[0:2]
                for a in CROP_ANNOS:
                    bl, bt = max(0, a['x'] - CROP_SX), max(0, a['y'] - CROP_SY)
                    bw = a['w'] if bl + a['w'] < CW else CW - bl
                    bh = a['h'] if bt + a['h'] < CH else CH - bt

                    wgt.write("%d %d %d %d 0 0 0 0 0 0\n" % (bl, bt, bw, bh))

                    if DEBUG:
                        cv2.rectangle(image_crop, (bl, bt), (bl+bw, bt+bh), (0, 0, 255), 2)
            else:
                wgt.write('0\n')
            key = cv2.waitKey(2)

            if DEBUG:
                cv2.imshow('cropped', image_crop)
                key = cv2.waitKey(DEBUG_DISPLAY_TIME)

            if key in [113, 120, 99]:
                break

    print('done!')


def main(_):
    IMAGE_DIR = FLAGS.image_dir
    GT_PATH = FLAGS.gt_path
    OUTPUT_IMAGE_DIR = FLAGS.output_image_dir
    OUTPUT_GT_PATH = FLAGS.output_gt_path
    RELTH = FLAGS.min_rel_size

    if not OUTPUT_IMAGE_DIR.endswith('/'):
        OUTPUT_IMAGE_DIR = OUTPUT_IMAGE_DIR + '/'

    refine_widerface_db(IMAGE_DIR, GT_PATH, OUTPUT_IMAGE_DIR, OUTPUT_GT_PATH, RELTH)

if __name__ == '__main__':
    tf.app.run()

# python refine_widerface_2.py --image_dir=/Users/gglee/Data/WiderFace/WIDER_train/images/ --gt_path=/Users/gglee/Data/WiderFace/wider_face_split/wider_face_train_bbx_gt.txt --output_image_dir=/Users/gglee/Data/WiderRefine/wider_train_0521 --output_gt_path=/Users/gglee/Data/WiderRefine/wider_train_0521.txt --min_rel_size=0.08
# python refine_widerface_2.py --image_dir=/Users/gglee/Data/face_ours/ --gt_path=/Users/gglee/Data/face_ours/face_ours.txt --output_image_dir=/Users/gglee/Data/face_train --output_gt_path=/Users/gglee/Data/face_train/gt.txt --min_rel_size=0.08
