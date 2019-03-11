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
tf.app.flags.DEFINE_float('min_size', 0.99, 'ignore faces smaller than min_size (0.0~1.0).'
                           'faces smaller than min_size will also be masked out'
                           'in the image.')
tf.app.flags.DEFINE_float('aspect_ratio', -1.0, 'Aspect ratio (WIDTH/HEIGHT) of output images.'
                        'If negatives, ARs are not changed.')

FLAGS = tf.app.flags.FLAGS


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


def find_small_and_large_faces(annos, threshold):
    larges = filter(lambda x: x['w'] >= threshold, annos)
    smalls = filter(lambda x: x['w'] < threshold, annos)

    return smalls, larges


def find_bounding_box(annos):
    l = min([a['x'] for a in annos])
    t = min([a['y'] for a in annos])
    r = max([a['x'] + a['w'] for a in annos])
    b = max([a['y'] + a['h'] for a in annos])
    return {'l': l, 't': t, 'r': r, 'b': b}


def check_overlap(bb1, bb2):
    if bb1['l'] > bb2['r'] or bb1['r'] < bb2['l']:
        return False
    if bb1['t'] > bb2['b'] or bb1['b'] < bb2['t']:
        return False

    return True


def get_iou(bb1, bb2):
    if bb1['l'] > bb2['r'] or bb1['r'] < bb2['l']:
        return 0.0
    if bb1['t'] > bb2['b'] or bb1['b'] < bb2['t']:
        return 0.0

    ib = [max(bb1['l'], bb2['l']), max(bb1['t'], bb1['t']), min(bb1['r'], bb2['r']), min(bb1['b'], bb2['b'])]
    ai = (ib[2] - ib[0]) * (ib[3] - ib[1])
    a1 = (bb1['r'] - bb1['l']) * (bb1['b'] - bb1['t'])
    a2 = (bb2['r'] - bb2['l']) * (bb2['b'] - bb2['t'])

    return ai / (a1 + a2 - ai)


def refine_annos(annos):
    '''
    remove annoations that not to be used: invalid and hard occlusion
    :param annos:
    :return:
    '''
    survive = []
    drop = []

    for a in annos:
        if a['invalid'] > 0 or a['occlusion'] > 1:
            drop.append(a)
        else:
            survive.append(a)
    return survive, drop


def draw_annos(image, annos, color, line_width):
    for a in annos:
        cv2.rectangle(image, (a['x'], a['y']), (a['x']+a['w'], a['y']+a['h']), color, line_width)


def pick_valid_crop(frame_size, lb, sb):
    '''
    large boxes 는 절대크기가 충분히 큰 box들. small boxes는 절대크기가 작아 원본 crop해도 사용이 불가
    large boxes 를 감싸면서 small box는 포함하지 않는 영역의 list를 리턴한다

    how it works? large boxes 가운데 두 개를 고른다. 두 box의 bounding box 내에 small box가 없으면 사용할 수 있는 crop 임

    :param frame_size: (W, H)
    :param lb:
    :param sb:
    :return:
    '''
    res = []

    num_lb = len(lb)
    MULTIPLE = 3 if num_lb < 4 else 1

    for i in range(num_lb):
        for j in range(i, num_lb):
            a0, a1 = lb[i], lb[j]
            if a0['l'] >= a0['r'] or a0['t'] >= a0['b'] or a1['l'] >= a1['r'] or a1['t'] >= a1['b']:
                print('skip invalid box')
                continue

            bbox = {'l': min(a0['l'], a1['l']), 't': min(a0['t'], a1['t']),         # this is the candidate
                    'r': max(a0['r'], a1['r']), 'b': max(a0['b'], a1['b'])}
            frame = {'l': frame_size[0], 't': frame_size[1], 'r': 0, 'b': 0}

            overlap = False
            for s in sb:
                if check_overlap(bbox, s):
                    overlap = True
                    break

            if not overlap:     # bbox does not overlap with sb -> safe to use -> expand it until it meet any s in sb
                # expand width first
                lmost = sorted(filter(lambda s: s['r'] < bbox['l'] and s['b'] > bbox['t'] and s['t'] < bbox['b'], sb),
                               key=itemgetter('r'), reverse=True)
                rmost = sorted(filter(lambda s: s['l'] > bbox['r'] and s['b'] > bbox['t'] and s['t'] < bbox['b'], sb),
                               key=itemgetter('l'))
                lmost = lmost[0] if lmost else frame
                rmost = rmost[0] if rmost else frame

                sbr = filter(lambda s: s['l'] > lmost['r'] and s['r'] < rmost['l'], sb)     ## up and down directions

                tmost = sorted(filter(lambda s: s['b'] < bbox['t'], sbr), key=itemgetter('b'), reverse=True)
                bmost = sorted(filter(lambda s: s['t'] > bbox['b'], sbr), key=itemgetter('t'))

                tmost = tmost[0] if tmost else frame
                bmost = bmost[0] if bmost else frame

                out_box = {'l': lmost['r'], 't': tmost['b'], 'r': rmost['l'], 'b': bmost['t']}

                obh, obw = out_box['b'] - out_box['t'], out_box['r'] - out_box['l']

                if obh > obw:
                    cy = int((out_box['b'] + out_box['t']) / 2)
                    out_box['t'] = int(cy - obw / 2) if int(cy - obw / 2) < bbox['t'] else int(
                        (bbox['t'] + out_box['t']) / 2)
                    out_box['b'] = int(cy + obw / 2) if int(cy + obw / 2) > bbox['b'] else int(
                        (bbox['b'] + out_box['b']) / 2)

                MAX_TRY = 30
                ARTH = 1.5

                # out_box와 bbox 사이의 box를 random search
                for m in range(MULTIPLE):
                    max_score = 0.0
                    max_box = out_box
                    max_box['score'] = 0.0

                    for t in range(MAX_TRY):
                        rl = out_box['l'] if out_box['l'] == bbox['l'] else randrange(out_box['l'], bbox['l'])
                        rt = out_box['t'] if out_box['t'] == bbox['t'] else randrange(out_box['t'], bbox['t'])
                        rr = out_box['r'] if out_box['r'] == bbox['r'] else randrange(bbox['r'], out_box['r'])
                        rb = out_box['b'] if out_box['b'] == bbox['b'] else randrange(bbox['b'], out_box['b'])

                        rw = rr - rl
                        rh = rb - rt
                        curar = rw / rh

                        if 1/ARTH < curar < ARTH:       # AR은 가로, 세로 모두 ARTH 이하이어야 함
                            score = min(curar, 4.0/3.0) / max(curar, 4.0/3.0) * rw*rh       # 4:3에 가까우면서 큰 것을 찾는다

                            if score > max_score:
                                max_box = {'l': rl, 't': rt, 'r': rr, 'b': rb, 'score': score}

                    # validity check for the last
                    h, w = max_box['b'] - max_box['t'], max_box['r'] - max_box['l']
                    if min(w, h) > 300 and not on_the_border(max_box, lb) and not is_redundant(max_box, res):
                        res.append(max_box)

    if len(res) > 0:
        res = sorted(res, key = itemgetter('score'), reverse=True)

    return res


def is_redundant(target, pool):
    if len(pool) == 0:
        return False

    for b in pool:
        if get_iou(target, b) > 0.6:
            return True
    return False


def on_the_border(border, boxes):
    for b in boxes:
        if (b['l'] < border['l'] < b['r'] or b['l'] < border['r'] < b['r']) and \
            (b['b'] > border['t'] and b['t'] < border['b']):
            return True

        if (b['t'] < border['t'] < b['b'] or b['t'] < border['b'] < b['b']) and \
            (b['r'] > border['l'] and b['l'] < border['r']):
            return True

    return False


def check_inside(inbox, outbox):
    if inbox['l'] >= outbox['l'] and inbox['t'] >= outbox['t'] and \
        inbox['r'] <= outbox['r'] and inbox['b'] <= outbox['b']:
        return True
    else:
        return False


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

        cv2.namedWindow('image')
        cv2.moveWindow('image', 0, 0);

        ### we have two tresholds: absth, relth
        ### ignore if largest face < absth (=48 pix)

        for idx in range(total_images):
            data = wdb.get_annos_by_image_index(idx)
            image_path = data['image_path']
            image = cv2.imread(image_path)
            H, W = image.shape[0:2]
            crop_saved = 0
            MAX_OUTPUT_PER_IMAGE = 5

            annos, invannos = refine_annos(data['annos'])       # remove invalid and heavily occluded faces
            small, large = find_small_and_large_faces(annos, ABSTH)     # find faces smaller than and larger than ABSTH

            def draw_box(image, anno, color, line_width=1):
                cv2.rectangle(image, (anno['l'], anno['t']), (anno['r'], anno['b']), color, line_width)

            if DEBUG:       # draw all annotations
                image_boxes = deepcopy(image)
                for ia in invannos:
                    draw_box(image_boxes, ia, (0, 0, 0), 2)
                for l in large:
                    draw_box(image_boxes, l, (0, 255, 0), 1)
                for s in small:
                    draw_box(image_boxes, s, (0, 0, 255), 1)
                cv2.imshow('invalid_small_large', image_boxes)

            if len(large) > 0:
                crops = pick_valid_crop((W, H), large, small)

                if DEBUG:   # draw crop candidates with valid boxes
                    image_crops = deepcopy(image)
                    for c in crops:
                        draw_box(image_crops, c, (0, 255, 255), 2)
                    for s in small:
                        draw_box(image_crops, s, (0, 0, 255), 1)
                    for l in large:
                        draw_box(image_crops, l, (0, 255, 0), 1)
                    cv2.imshow('crop_boxes', image_crops)

                for n, c in enumerate(crops):
                    # make new annotations w.r.t crops
                    cropped_anno = []
                    REL_COND = True

                    x0, y0 = c['l'], c['t']

                    for a in large:
                        if check_inside(a, c):  # if annotation is in crop
                            newa = deepcopy(a)
                            newa['x'] -= x0
                            newa['y'] -= y0
                            newa['l'] -= x0
                            newa['t'] -= y0
                            newa['r'] -= x0
                            newa['b'] -= y0
                            assert newa['r'] <= c['r'] - c['l'] and newa['b'] <= c['b'] - c['t'], print(newa, c)
                            cropped_anno.append(newa)

                            if newa['w'] < RELTH*(c['r']-c['l']):       # if face is too small
                                REL_COND = False
                                break

                    if REL_COND:
                        if crop_saved < MAX_OUTPUT_PER_IMAGE:
                            cropped_image = deepcopy(image[c['t']:c['b'], c['l']:c['r'], :])

                            # save the annotation and image finally!
                            dst_path = image_path.replace(db_path, write_db_path)
                            paths = dst_path.split('.')
                            dst_path = paths[0] + '_%d.' % n + paths[1]
                            dir_path = os.path.dirname(dst_path)

                            if not os.path.exists(dir_path):
                                os.makedirs(dir_path)

                            cv2.imwrite(dst_path, cropped_image)
                            H, W = cropped_image.shape[0:2]

                            # (2) write corresponding ground truth
                            filename = os.path.basename(dst_path)
                            dirname = os.path.dirname(dst_path).rsplit('/', 1)[1]

                            wgt.write("%s\n" % (os.path.join(dirname, filename)))

                            # if negative:
                            #     wgt.write("1\n0 0 0 0 0 0 0 0 0 0\n")
                            # else:
                            wgt.write("%d\n" % (len(cropped_anno)))

                            def _display_debug_info(img, crops, lbs, sbs):
                                image = deepcopy(img)
                                for c in crops:
                                    draw_box(image, c, (255, 0, 255), 2)

                                for s in sbs:
                                    draw_box(image, s, (0, 0, 255), 1)

                                for l in lbs:
                                    draw_box(image, l, (0, 255, 0), 1)
                                cv2.imshow('whatiswrong', image)
                                cv2.waitKey(-1)


                            def _is_valid_anno(image, anno):
                                left = sorted(anno, key=itemgetter('l'))[0]['l']
                                top = sorted(anno, key=itemgetter('t'))[0]['t']
                                right = sorted(anno, key=itemgetter('r'), reverse=True)[0]['r']
                                bottom = sorted(anno, key=itemgetter('b'), reverse=True)[0]['b']
                                H, W = image.shape[0: 2]

                                if left < 0 or top < 0 or right > W or bottom > H:
                                    return False
                                else:
                                    return True

                            if DEBUG:
                                image_crop = deepcopy(cropped_image)
                                for a in cropped_anno:
                                    draw_box(image_crop, a, (0, 255, 0))
                                cv2.imshow("cropped_%d" % n, image_crop)

                            for a in cropped_anno:
                                assert a['x'] >= 0.0 and a['y'] >= 0.0, print("Valid Anno:" , _is_valid_anno(image, data['annos']), "ImageSize: ", image.shape, _display_debug_info(image, crops, large, small))
                                assert a['x'] + a['w'] <= W, print("Valid Anno:" , _is_valid_anno(image, data['annos']), "ImageSize: ", image.shape, _display_debug_info(image, crops, large, small))   #_display_debug_info(image, crops, large, small)
                                assert a['y'] + a['h'] <= H, print("Valid Anno:" , _is_valid_anno(image, data['annos']), "ImageSize: ", image.shape, _display_debug_info(image, crops, large, small))   #_display_debug_info(image, crops, large, small)
                                wgt.write(("%d %d %d %d %d %d %d %d %d %d\n") % ( \
                                    a['x'], a['y'], a['w'], a['h'], \
                                    a['blur'], a['expression'], a['illumination'], \
                                    a['invalid'], a['occlusion'], a['pose']))
                            crop_saved += 1

                        # draw_annos(cropped_image, cropped_anno, (0, 0, 255), 1)
                        # cv2.imshow('image_%d' % n, cropped_image)

            #     for c in crops:
            #         cv2.rectangle(image, (c['l'], c['t']), (c['r'], c['b']),
            #                       (0, 50 + n * 200 / len(crops), 255 - n * 200 / len(crops)), 1)
            #
            # draw_annos(image, annos, (0, 255, 0), 2)
            # draw_annos(image, invannos, (0, 0, 255), 2)
            # draw_annos(image, small, (255, 0, 0), 1)
            #
            # cv2.imshow('image', image)
            key = cv2.waitKey(2)
            if DEBUG:
                key = cv2.waitKey(DEBUG_DISPLAY_TIME)

            if key in [113, 120, 99]:
                break

    print('done!')


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
    ABSTH = 36
    RELTH = 0.1

    refine_widerface_db(IMAGE_DIR, GT_PATH, OUTPUT_IMAGE_DIR, OUTPUT_GT_PATH, ABSTH, RELTH, ASPECT_RATIO)

if __name__ == '__main__':
    tf.app.run()

# python refine_widerface.py --type=split --image_dir=/Volumes/Data/FaceDetectionDB/WiderFace/WIDER_val/images/ --gt_path=/Volumes/Data/FaceDetectionDB/WiderFace/wider_face_split/wider_face_val_bbx_gt.txt --output_image_dir=/Volumes/Data/WIDER_SPLIT_TEST/images --output_gt_path=/Volumes/Data/WIDER_SPLIT_TEST/gt_split.txt --min_size=0.1
# python refine_widerface.py --type=mask --image_dir=/Volumes/Data/FaceDetectionDB/WiderFace/WIDER_val/images/ --gt_path=/Volumes/Data/FaceDetectionDB/WiderFace/wider_face_split/wider_face_val_bbx_gt.txt --output_image_dir=/Volumes/Data/WIDER_MASK_TEST/images --output_gt_path=/Volumes/Data/WIDER_MASK_TEST/gt_mask.txt --min_size=0.1