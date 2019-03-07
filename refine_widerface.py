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

import numpy as np
import tensorflow as tf
import widerface_explorer
from random import randrange
from copy import deepcopy

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
tf.app.flags.DEFINE_float('min_size', 0.99, 'ignore faces smaller than min_size (0.0~1.0).'
                           'faces smaller than min_size will also be masked out'
                           'in the image.')
tf.app.flags.DEFINE_float('aspect_ratio', -1.0, 'Aspect ratio (WIDTH/HEIGHT) of output images.'
                        'If negatives, ARs are not changed.')

FLAGS = tf.app.flags.FLAGS

def is_small_face(face_width, face_height, image_width, image_height, min_size):
    '''
    return true if a face is small compared to the given image size
    :param face_width:
    :param face_height:
    :param image_width:
    :param image_height:
    :param min_size: 0.0 ~ 1.0
    :return:
    '''
    if face_width >= image_width * min_size and face_height >= image_height * min_size:
        return False
    else:
        return True


def mask_and_copy(data, min_size, image_write_path, write_file, header):
    """If small face exists, mask those out and then copy the image and write modified ground truth
    """
    image = cv2.imread(data['image_path'])
    assert image is not None, "Failed to load image: %s"%(data['image_path'])

    height, width = image.shape[0:2]

    anno2write = []
    negative = False
    if len(data['annos']) == 0:
        negative = True
    else:
        for a in data['annos']:
            x, y, w, h = a['x'], a['y'], a['w'], a['h']

            if is_small_face(w, h, width, height, min_size):
                assert w > 0 and h > 0, '%s\n(x, y, w, h)=(%.2f, %.2f, %.2f, %.2f)' % (data['image_path'], x, y, w, h)
                rrect = ((int(x + w / 2), int(y + h / 2)), (int(w), int(h)), 0.0)
                cv2.ellipse(image, rrect, (0, 0, 0), -1)
            else:
                anno2write.append(a)

    write_file.write("%s\n" % (header))

    if negative:
        write_file.write("0\n")
    else:
        write_file.write("%d\n" % (len(anno2write)))

        for a in anno2write:
            write_file.write(("%d %d %d %d %d %d %d %d %d %d\n") % ( \
                a['x'], a['y'], a['w'], a['h'], \
                a['blur'], a['expression'], a['illumination'], \
                a['invalid'], a['occlusion'], a['pose']))

    #if not os.path.exists(image_write_dir):
    #    os.makedirs(image_write_dir)
    #image_write_path = os.path.join(image_write_dir, os.path.splitext(os.path.basename(data['image_path']))[0])
    cv2.imwrite(image_write_path, image)


def ignore_or_copy(data, min_size, image_write_path, write_file, header):
    image = cv2.imread(data['image_path'])
    assert image is not None, "Failed to load image: %s" % (data['image_path'])

    height, width = image.shape[0:2]

    if len(data['annos']) == 0:
        negative = True
        min_w = width
        min_h = height
    else:
        negative = False
        min_w = min([a['w'] for a in data['annos']])
        min_h = min([a['h'] for a in data['annos']])

    if min_size == 0.0 or not is_small_face(min_w, min_h, width, height, min_size):
                # cond 1) threshold = 0.0 --> copy. no need to check annotation sizes
                # cond 2) if min_face > threshold: valid image -> copy image to a new directory and write annotation
        # (1) copy file
        copyfile(data['image_path'], image_write_path)

        # (2) write corresponding ground truth
        write_file.write("%s\n" % header)
        if negative:
            write_file.write("0\n")
            # write_file.write("1\n0 0 0 0 0 0 0 0 0 0\n")
        else:
            write_file.write("%d\n" % (len(data['annos'])))

            for a in data['annos']:
                write_file.write(("%d %d %d %d %d %d %d %d %d %d\n") % ( \
                    a['x'], a['y'], a['w'], a['h'], \
                    a['blur'], a['expression'], a['illumination'], \
                    a['invalid'], a['occlusion'], a['pose']))

def crop_to_aspect_ratio(data, aspect_ratio, buff_dir='.'):
    data_new = deepcopy(data)
    annos = data_new['annos']

    if len(annos) == 0: # negative image
        return data, aspect_ratio

    image = cv2.imread(data_new['image_path'])

    HEIGHT, WIDTH = image.shape[0:2]

    # (1) no need to crop if AR is already almost the same
    ar_ori = WIDTH/float(HEIGHT)
    if aspect_ratio*0.99 < ar_ori < aspect_ratio*1.01:
        return data_new, ar_ori

    # (2) if not, crop it to the desired aspect ratio

    # (2-1) get bounding box that includes all faces: (min_x, min_y, max_x, max_y)
    min_x = max(0, np.amin([a['x'] for a in annos]))   # face bbox coords
    min_y = max(0, np.amin([a['y'] for a in annos]))
    max_x = min(np.amax([a['x']+a['w'] for a in annos]), WIDTH)
    max_y = min(np.amax([a['y']+a['h'] for a in annos]), HEIGHT)

    w0 = int(max_x-min_x)   # w.r.t bbx_w
    h0 = int(w0/aspect_ratio)

    h1 = int(max_y-min_y)   # w.r.t bbx_h
    w1 = int(h1*aspect_ratio)

    if w0 > w1:     # bw, bh are target image size which (1) includes all faces, (2) of the proper aspect ratio
        bw, bh = w0, h0
    else:
        bw, bh = w1, h1

    image_cropped = image
    cx, cy = 0, 0

    # (2) find best fit (largest box with given ar)
    if bw <= WIDTH and bh <= HEIGHT:    # no problem, just crop
        # l: [0, min_x], t: [0:min_y], r: [max_x:WIDTH], b: [max_y:HEIGHT]
        cx, cy = min_x, min_y
        cw, ch = bw, bh

        min_try = 100
        max_try = 1000
        found_valid = False
        num_try = 0

        do_more = True

        #for r in range(100):    # try random pick for 100 times
        while do_more:
            if num_try > min_try and found_valid:
                do_more = False
            if num_try > max_try:
                do_more = False

            num_try += 1
            x0 = randrange(0, min_x+1)
            y0 = randrange(0, min_y+1)
            x1 = randrange(max_x, WIDTH+1)

            w = x1-x0
            h = int(w/aspect_ratio)
            y1 = y0+h

            if y1 < max_y or y1 >= HEIGHT:  # invalid rect
                continue
            else:
                found_valid = True

            _cx, _cy = x0, y0
            _cw, _ch = (x1-x0), (y1-y0)

            if _cw*_ch > cw*ch:     # keep the larger one
                cx, cy, cw, ch = _cx, _cy, _cw, _ch

        image_cropped = image[cy:cy+ch, cx:cx+cw, :]

    # check aspect ratio of refined image
    th, tw = image_cropped.shape[0:2]
    new_ar = tw / float(th)

    if not (aspect_ratio*0.95 < new_ar < aspect_ratio*1.05):
        print("output image size = (%dx%d). ar=%.2f != %.2f => letter box expansion" % (tw, th, new_ar, aspect_ratio))
        image_cropped, cxe, cye = expand_letter_box(image_cropped, aspect_ratio)
        cx += cxe
        cy += cye

    filename = os.path.basename(data_new['image_path'])     # image is changed: so we neet to save it somewhere (to create tfrecord)
    savepath = os.path.join(buff_dir, '%.2f_'%aspect_ratio+filename)
    cv2.imwrite(savepath, image_cropped)
    data_new['image_path'] = savepath

    # need to modify annotations
    for a in annos:
        a['x'] -= cx
        a['y'] -= cy

    return data_new, new_ar


def expand_letter_box(image, aspect_ratio):

    HEIGHT, WIDTH = image.shape[0:2]
    ar = WIDTH/float(HEIGHT)

    if ar > aspect_ratio:   # wider image -> expand height
        exp_height = int(abs(WIDTH / aspect_ratio - HEIGHT) / 2)
        cx, cy = 0, -exp_height
        image_cropped = cv2.copyMakeBorder(image, exp_height, exp_height, 0, 0, cv2.BORDER_REPLICATE)
    else:
        exp_width = int(abs(HEIGHT * aspect_ratio - WIDTH) / 2)
        cx, cy = -exp_width, 0
        image_cropped = cv2.copyMakeBorder(image, 0, 0, exp_width, exp_width, cv2.BORDER_REPLICATE)

    return image_cropped, cx, cy


def bound_image_size(data, max_width, buff_dir='.'):
    data_new = deepcopy(data)
    annos = data_new['annos']

    if len(annos) == 0: # negative image
        return data

    image = cv2.imread(data_new['image_path'])
    assert image is not None, 'bound_image_size failes loading image: %s'%data_new['image_path']
    HEIGHT, WIDTH = image.shape[0:2]

    if WIDTH > max_width:
        scale = WIDTH/max_width
        resized = cv2.resize(image, (int(WIDTH*scale), int(HEIGHT*scale)))

        for a in annos:
            a['x'] *= scale
            a['y'] *= scale
            a['w'] *= scale
            a['h'] *= scale

        filename = os.path.basename(data_new['image_path'])  # image is changed: so we neet to save it somewhere (to create tfrecord)
        savepath = os.path.join(buff_dir, '%d_'%max_width + filename)
        cv2.imwrite(savepath, resized)
        data_new['image_path'] = savepath

    return data_new


def find_small_and_large_faces(annos, threshold):
    larges = filter(lambda x: x['w'] >= threshold, annos)
    smalls = filter(lambda x: x['w'] < threshold, annos)

    return smalls, larges


def find_bounding_box(annos):
    l = min([a['x'] for a in annos])
    t = min([a['y'] for a in annos])
    r = max([a['x'] + a['w'] for a in annos])
    b = max([a['y'] + a['h'] for a in annos])
    return {'l':l, 't': t, 'r':r, 'b':b}


def check_overlap(bb1, bb2):
    if bb1['l'] > bb2['r'] or bb1['r'] < bb2['l']:
        return False
    if bb1['t'] > bb2['b'] or bb1['b'] < bb2['t']:
        return False

    return True

def refine_widerface_db(db_path, gt_path, write_db_path, write_gt_path, ABSTH, RELTH, aspect_ratio):

    wdb = widerface_explorer.wider_face_db(db_path, gt_path)
    tmp_path = os.path.join(os.path.dirname(db_path), 'tmp')

    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)
        assert os.path.exists(tmp_path)

    if not os.path.exists(os.path.dirname(write_gt_path)):
        os.makedirs(os.path.dirname(write_gt_path))

    with open(write_gt_path, 'w') as wgt:
        total_images = wdb.get_image_count()

        cnt_small_ignore = 0
        cnt_small_use = 0
        cnt_no_face = 0
        cnt_large_use = 0

        cv2.namedWindow('image')
        cv2.moveWindow("image", 0, 0);

        ### we have two tresholds: absth, relth

        ### ignore if largest face < absth (=48 pix)

        for idx in range(total_images):
            data = wdb.get_annos_by_image_index(idx)

            src_path = data['image_path']
            annos = data['annos']
            widths = [a['w'] for a in annos]

            if not widths:      # no annotations -> pure negative image
                cnt_no_face += 1
            elif max(widths) < ABSTH:      # larges < absth, ignore
                # print('max is too small: %d < 48' % max(widths))
                image = cv2.imread(src_path)
                W, H = image.shape[0:2]

                # cv2.line(image, (0, 0), (W, H), (0, 0, 255), 3)
                # cv2.line(image, (W, 0), (0, H), (0, 0, 255), 3)
                # cv2.imshow('image', image)
                # cv2.waitKey(10)
                cnt_small_ignore += 1
            else:
                image = cv2.imread(src_path)
                W, H = image.shape[0:2]

                relmin = min(widths) / float(W)

                if relmin >= RELTH:  # no problem
                    cnt_large_use += 1
                else:
                    cnt_small_use += 1

            # else:
            #     smalls, larges = find_small_and_large_faces(annos, 48)
            #
            #     if len(smalls) == 0:
            #         cnt_large_use += 1
            #     elif len(larges) == 0:
            #         cnt_small_ignore += 1
            #     else:
            #         small_bb = find_bounding_box(smalls)
            #         large_bb = find_bounding_box(larges)
            #
            #         color_small = None
            #         color_large = None
            #
            #         if not check_overlap(small_bb, large_bb):
            #             cnt_small_use += 1
            #             color_small = (0, 255, 0)
            #             color_large = (0, 255, 0)
            #         else:
            #             cnt_small_ignore += 1
            #             color_small = (0, 0, 255)
            #             color_large = (255, 0, 0)
            #
            #         image = cv2.imread(src_path)
            #         W, H = image.shape[0:2]
            #         cv2.rectangle(image, (small_bb['l'], small_bb['t']), (small_bb['r'], small_bb['b']), color_small, 2)
            #         cv2.rectangle(image, (large_bb['l'], large_bb['t']), (large_bb['r'], large_bb['b']), color_large, 2)
            #
            #         for s in smalls:
            #             cv2.rectangle(image, (s['x'], s['y']), (s['x']+s['w'], s['y']+s['h']), (80, 80, 255), 1)
            #         for l in larges:
            #             cv2.rectangle(image, (l['x'], l['y']), (l['x'] + l['w'], l['y'] + l['h']), (255, 80, 80), 1)
            #         cv2.imshow('image', image)
            #         cv2.waitKey(-1)

        print('small to ignore: %d / %d (%.2f%%)' % (
        cnt_small_ignore, total_images, (cnt_small_ignore * 100.0 / total_images)))
        print('small to use: %d / %d (%.2f%%)' % (
        cnt_small_use, total_images, (cnt_small_use * 100.0 / total_images)))
        print('large to use: %d / %d (%.2f%%)' % (
        cnt_large_use, total_images, (cnt_large_use * 100.0 / total_images)))

            # # crop image to meet the aspect ratio constraint (if any)
            # if aspect_ratio > 0.0:
            #     data, ar2 = crop_to_aspect_ratio(data, aspect_ratio, tmp_path)
            #
            # data = bound_image_size(data, 640, tmp_path)    # bound image width to 1600, maintaining the aspect ratio
            #
            # # no need now
            # #if aspect_ratio > 0.0 and max([aspect_ratio, ar2]) / float(min([aspect_ratio, ar2])) > 1.5:    # ignore if diff too much
            # #    skipped += 1
            # #    print("skipped %d images"%skipped)
            # #    continue
            #
            # # prepare paths to write results
            # image_path = data['image_path']
            #
            # dst_path = src_path.replace(os.path.normpath(db_path), os.path.normpath(write_db_path))
            # dir_path = os.path.dirname(dst_path)
            #
            # if not os.path.exists(dir_path):
            #     os.makedirs(dir_path)
            #
            # filename = os.path.basename(dst_path)
            # dirname = os.path.dirname(dst_path).rsplit('/', 1)[1]
            #
            # header = os.path.join(dirname, filename)
            #
            # # crop or split, where actual image createion happens
            # if operation =='mask':
            #     mask_and_copy(data, min_size, dst_path, wgt, header)
            # else:   # split
            #     ignore_or_copy(data, min_size, dst_path, wgt, header)
            #
            # if idx % 100 == 0:
            #     print("image %d: %s" % (idx, image_path))


def split_wider_face_db(db_path, gt_path, write_db_path, write_gt_path, min_size):
    """Make a subset of wider face db that contains face larger than min_size only

    Args:
        image_dir: database folder path (~~~ for example)
        gt_path: ground truth text file
        output_image_dir: where subset of image_path will be stored
        output_gt_path: ground truth text file for images in 'write_image_path'
        min_size: threshold level
    """       
    wdb = widerface_explorer.wider_face_db(db_path, gt_path)

    if not os.path.exists(os.path.dirname(write_gt_path)):
        os.makedirs(os.path.dirname(write_gt_path))
    
    with open(write_gt_path, 'w') as wgt:
        total_images = wdb.get_image_count()
        
        for idx in range(total_images):        
            data = wdb.get_annos_by_image_index(idx)

            image_path = data['image_path']
            image = cv2.imread(image_path)

            if image is None:
                print('Unable to load image: %s'%image_path)
                break

            height, width = image.shape[0:2]

            #assert len(data['annos']) > 0, "no annotation found in %s"%image_path
            if len(data['annos']) == 0: 
                negative = True
                min_w = width
                min_h = height
            else:
                negative = False
                min_w = min([a['w'] for a in data['annos']])
                min_h = min([a['h'] for a in data['annos']])

            if min_w >= width*min_size and min_h >= height*min_size:  # valid image -> copy image to a new directory and write annotation
                # (1) copy file
                dst_path = image_path.replace(db_path, write_db_path)
                dir_path = os.path.dirname(dst_path)

                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)

                copyfile(image_path, dst_path)

                # (2) write corresponding ground truth                
                filename = os.path.basename(dst_path)
                dirname = os.path.dirname(dst_path).rsplit('/', 1)[1]

                wgt.write("%s\n"%(os.path.join(dirname, filename)))

                print(filename, width, height, min_w, min_h, min_size*width, min_size*height)

                # if negative:
                #     wgt.write("1\n0 0 0 0 0 0 0 0 0 0\n")
                # else:
                wgt.write("%d\n"%(len(data['annos'])))

                for a in data['annos']:
                    wgt.write(("%d %d %d %d %d %d %d %d %d %d\n")%( \
                                a['x'], a['y'], a['w'], a['h'], \
                                a['blur'], a['expression'], a['illumination'], \
                                a['invalid'], a['occlusion'], a['pose']))

            if idx%100 == 0:
                print("image %d: %s"%(idx, image_path))

        #wgt.close()

def main(_):
    IMAGE_DIR = FLAGS.image_dir
    GT_PATH = FLAGS.gt_path
    OUTPUT_IMAGE_DIR = FLAGS.output_image_dir
    OUTPUT_GT_PATH = FLAGS.output_gt_path
    MIN_SIZE = FLAGS.min_size
    ASPECT_RATIO = FLAGS.aspect_ratio
    ABSTH = 48
    RELTH = 0.1

    refine_widerface_db(IMAGE_DIR, GT_PATH, OUTPUT_IMAGE_DIR, OUTPUT_GT_PATH, ABSTH, RELTH, ASPECT_RATIO)

if __name__ == '__main__':
    tf.app.run()

# python refine_widerface.py --type=split --image_dir=/Volumes/Data/FaceDetectionDB/WiderFace/WIDER_val/images/ --gt_path=/Volumes/Data/FaceDetectionDB/WiderFace/wider_face_split/wider_face_val_bbx_gt.txt --output_image_dir=/Volumes/Data/WIDER_SPLIT_TEST/images --output_gt_path=/Volumes/Data/WIDER_SPLIT_TEST/gt_split.txt --min_size=0.1
# python refine_widerface.py --type=mask --image_dir=/Volumes/Data/FaceDetectionDB/WiderFace/WIDER_val/images/ --gt_path=/Volumes/Data/FaceDetectionDB/WiderFace/wider_face_split/wider_face_val_bbx_gt.txt --output_image_dir=/Volumes/Data/WIDER_MASK_TEST/images --output_gt_path=/Volumes/Data/WIDER_MASK_TEST/gt_mask.txt --min_size=0.1