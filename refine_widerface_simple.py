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

DEBUG = False
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
tf.app.flags.DEFINE_float('min_rel_size', 0.1, 'ignore faces small than min_rel_size * image_size')

FLAGS = tf.app.flags.FLAGS

MIN_FRAME_SIZE = 240


def find_small_and_large_faces(annos, threshold):
    '''
    return faces smaller than and larger than a given threshold
    :param annos:
    :param threshold: because we crop from the original image, threshold is in pixels
    :return:
    '''

    larges = filter(lambda x: x['w'] >= threshold, annos)
    smalls = filter(lambda x: 10 < x['w'] < threshold, annos)
    drop = filter(lambda x: x['w'] <= 10, annos)

    return smalls, larges


def find_smallest_and_largest_faces(annos):
    if len(annos) > 1:
        sorted_ = sorted(annos, key=lambda k: k['w'])
        return sorted_[0], sorted_[-1]
    elif len(annos) == 1:
        return annos[0], annos[0]
    else:
        return None, None


def refine_annos(annos):
    '''remove annoations that not to be used: labeld as invalid or hard occlusion'''

    survive = []
    drop = []

    for a in annos:
        if a['invalid'] > 0 or a['occlusion'] > 1:
            drop.append(a)
        else:
            survive.append(a)
    return survive, drop


def refine_widerface_db(db_path, gt_path, write_db_path, write_gt_path, ABSTH, RELTH):

    wdb = widerface_explorer.wider_face_db(db_path, gt_path)
    tmp_path = os.path.join(os.path.dirname(db_path), 'tmp')

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
            crop_saved = 0
            MAX_OUTPUT_PER_IMAGE = 6

            annos, invannos = refine_annos(data['annos'])               # remove invalid and heavily occluded faces
            smallest, largest = find_smallest_and_largest_faces(annos)

            AS_ORIGINAL = True if smallest and float(smallest['w']) / W >= RELTH and max(H/float(W), W/float(H)) <= 1.5 else False      # use the original image too

            if AS_ORIGINAL:
                dst_path = image_path.replace(db_path, write_db_path)
                dir_path = os.path.dirname(dst_path)

                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)

                cv2.imwrite(dst_path, image)

                filename = os.path.basename(dst_path)
                dirname = os.path.dirname(dst_path).rsplit('/', 1)[1]

                if AS_ORIGINAL:
                    wgt.write("%s\n" % (os.path.join(dirname, filename)))

                # if AS_BACKGROUND:
                #     wgt.write("%d\n" % 0)
                # else:
                    wgt.write("%d\n" % (len(annos)))
                    for a in annos:
                        wgt.write(("%d %d %d %d %d %d %d %d %d %d\n") % ( \
                            a['x'], a['y'], a['w'], a['h'], \
                            a['blur'], a['expression'], a['illumination'], \
                            a['invalid'], a['occlusion'], a['pose']))

            key = cv2.waitKey(2)
            if DEBUG:
                key = cv2.waitKey(DEBUG_DISPLAY_TIME)

            if key in [113, 120, 99]:
                break

    print('done!')

def main(_):
    IMAGE_DIR = FLAGS.image_dir
    GT_PATH = FLAGS.gt_path
    OUTPUT_IMAGE_DIR = FLAGS.output_image_dir
    OUTPUT_GT_PATH = FLAGS.output_gt_path
    ABSTH = FLAGS.min_abs_size
    RELTH = FLAGS.min_rel_size

    refine_widerface_db(IMAGE_DIR, GT_PATH, OUTPUT_IMAGE_DIR, OUTPUT_GT_PATH, ABSTH, RELTH)

if __name__ == '__main__':
    tf.app.run()

 # python refine_widerface.py --image_dir=/Users/gglee/Data/WiderFace/WIDER_train/images/ --gt_path=/Users/gglee/Data/WiderFace/wider_face_split/wider_face_train_bbx_gt.txt --output_image_dir=/Users/gglee/Data/WiderRefine/train/ --output_gt_path=/Users/gglee/Data/WiderRefine/train/wider_refine_train_gt.txt