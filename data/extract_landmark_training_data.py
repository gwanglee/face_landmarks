#-*- coding: utf-8 -*-

'''
Landmark dataset 에서 training data (face patch, landmark points)를 추출하기 위한 script.
Landmark dataset (300W, Menpo, Multi-PIE)는 원 영상에 대한 68 landmark points가 annotate 되어 있다. 따라서, 얼굴 검출기를 적용 하여
얼굴 영역만을 잘라내고, 잘라낸 얼굴 영역에 대해 landmark 좌표를 normalize 해야 한다.
'''

import os
import cv2
import numpy as np
from copy import deepcopy
from model.face_detector import Detector
from random import random

DEBUG = True

MIN_CONF = 0.5      # 얼굴 검출 시 Confidence
MIN_OVERLAP = 0.5   # 얼굴 검출 시 IOU threshold: face의 ground truth가 없기 때문에, landmark points의 bounding box와 비교
EXTEND = True       # 검출된 얼굴 영역 영상 저장 시 영역을 확장할 것인지 결정
EXTEND_RATIO = 0.1  # 검출된 얼굴 영역의 확장 시 확장 크기를 결정 (0.1 이면 10% 확장하여 저장)

ROTATE = False       # rotation augmentation 을 사용할 것인지
MAX_ROTATE = 30     # rotation augmentation 의 범위
DETECTOR_INPUT_SIZE = 128  # 검출기의 입력 크기

# todo: add jittering?

WRITE_PATH = '/Users/gglee/Data/Landmark/export/0612'  # Crop된 patch와 landmark annotation을 저장할 폴더
# RAND_FACTOR = 16.0        # Multi-PIE의 경우 양이 많기 때문에 RAND_FACTOR의 비율로 sampling 하여 이용

IMAGES_DIR_PATHS = [  # 학습 data를 추출하기 위한 원본 학습 data가 저장되어 있는 위치
    '/Users/gglee/Data/Landmark/300W/01_Indoor',
    '/Users/gglee/Data/Landmark/300W/02_Outdoor',
    '/Users/gglee/Data/Landmark/menpo_train_release',
    '/Users/gglee/Data/Landmark/mpie'
]

FACE_DETECTOR_PATH = '/Users/gglee/Data/TFModels/0610/ssd_face_128_v28/freeze/frozen_inference_graph.pb'  # 사용할 얼굴 검출기

def get_files(folder_path):
    '''
    Return list of filename pairs of [image, annotation]
    :param folder_path:
    :return: list of [/path/filename.png, /path/filename/pts]
    '''
    assert os.path.exists(folder_path) and os.path.isdir(folder_path)

    files = []
    for f in os.listdir(folder_path):
        if not f.startswith('.') and (f.endswith('png') or f.endswith('jpg')):
            basename = os.path.splitext(f)[0]
            pts_path = os.path.join(folder_path, basename + '.pts')
            if os.path.exists(pts_path):
                files.append([os.path.join(folder_path, f), pts_path])

    return files


def read_pts(filepath):
    '''
    Read landmark points (.pts)
    :param filepath: path to the pts text file
    :return: decoded .pts ([x, y] x num_pts)
    '''

    assert os.path.exists(filepath) and os.path.isfile(filepath)

    with open(filepath) as rf:
        l1 = rf.readline()
        l2 = rf.readline()

        n_points = int(l2.split(' ')[1])

        pts = []

        rf.readline()
        for i in range(n_points):
            x, y = rf.readline().split(' ')
            x, y, = float(x), float(y)
            pts.append([x, y])

        assert len(pts) == n_points

        return pts


def get_bounding_box(pts):
    '''
    Get bounding box for a given set of landmark points
    :param pts:
    :return: bbox in [l, t, r, b]
    '''

    x = [p[0] for p in pts]
    y = [p[1] for p in pts]
    return {'x': min(x), 'y': min(y), 'w': max(x)-min(x), 'h': max(y)-min(y)}


def normalize_points_with_rect(pts, faceRect):
    '''
    normalize point coordiates in range of [0, 1] w.r.t. the face rectangle
    :param pts: landmark points
    :param faceRect:  [l, t, r, b]
    :return: normalized points
    '''

    l, t = faceRect['x'], faceRect['y']
    w, h = faceRect['w'], faceRect['h']

    normed = [[(p[0] - l) / w, (p[1] - t) / h] for p in pts]
    return normed


def center_normalize_points_with_rect(pts, faceRect):
    '''
    normalize point coordiates between [-1, 1] w.r.t. the face rectangle
    :param pts: landmark points
    :param faceRect:  [l, t, r, b]
    :return: normalized points
    '''

    l, t = faceRect['x'], faceRect['y']
    w, h = faceRect['w'], faceRect['h']
    cx, cy = (l+w)/2.0, (t+h)/2.0

    normed = [[(p[0] - cx) / (w/2), (p[1] - cy) / (h/2)] for p in pts]
    return normed


def load_detection(image_path, det_name, th=0.1):
    '''
    load detection data having confidence larger than a given threshold
    :param image_path:
    :param det_name:
    :param th:
    :return:
    '''
    dir_name = os.path.dirname(image_path)
    basename = os.path.basename(image_path).split('.')[0]
    det_dir = os.path.join(dir_name, det_name)
    dets = []

    det_path = os.path.join(det_dir, basename + '.txt')
    if not os.path.exists(det_path):
        return dets
    else:
        with open(det_path) as rf:
            lines = rf.readlines()
            num_dets = int(lines[1].strip())
            assert num_dets == len(lines)-2

            for i in range(2, num_dets):
                x, y, w, h, c = lines[i].split()
                if float(c) > th:
                    dets.append({'x': float(x), 'y': float(y), 'w': float(w), 'h': float(h), 'conf': float(c)})

        return dets


def get_overlap(box1, box2):
    '''
    compute iou between two boxes
    :param box1:
    :param box2:
    :return:
    '''

    l0, t0, r0, b0 = box1['x'], box1['y'], box1['x']+box1['w'], box1['y']+box1['h']
    l1, t1, r1, b1 = box2['x'], box2['y'], box2['x']+box2['w'], box2['y']+box2['h']

    if l0 > r1 or r0 < l1:
        return 0.0
    if t0 > b1 or b0 < t1:
        return 0.0

    li = max(l0, l1)        # intersection
    ri = min(r0, r1)
    ti = max(t0, t1)
    bi = min(b0, b1)

    ai = (ri-li)*(bi-ti)    # area of the intersection
    a0 = (r0-l0)*(b0-t0)    # area of box_0
    a1 = (r1-l1)*(b1-t1)    # area of box_1

    return ai / (a0+a1-ai)


def find_best_matching_box(target, detections, MIN_CONF=0.2, MIN_OVERLAP=0.2):
    '''
    find the best matching detection box with a given box (= target)
    :param target: [l, t, r, b]
    :param detections: [{'x', 'y', 'w', 'h', 'conf'}]
    :return: {'x', 'y', 'w', 'h', 'conf'}
    '''

    if len(detections) == 0:
        return None

    max_overlap = 0.0
    max_index = -1

    # boxes = filter(lambda b: get_overlap(target, b) > MIN_OVERLAP and b['conf'] > MIN_CONF, detections)
    #
    # if not boxes:
    #     return None
    #
    # bottom = sorted(boxes, key=itemgetter(''), reverse=True)[0]['b']
    #
    # sorted()

    for i, b in enumerate(detections):
        if b['conf'] > MIN_CONF:
            cur_overlap = get_overlap(target, b)
            if cur_overlap > max_overlap and cur_overlap > MIN_OVERLAP:
                max_index = i
                max_overlap = cur_overlap

    if max_index >= 0:
        return detections[max_index]
    else:
        return None


def square_and_expand(box, r=0.0, frame=None):
    '''
    make box to square and expand its border
    :param box: target to extend
    :param r: extend ratio = 1+r
    :param frame: if not None, clip extended box to the frame (W, H)
    :return:
    '''

    w = box['w']
    h = box['h']
    cx = box['x'] + w/2.0
    cy = box['y'] + h/2.0

    tw = th = max(w, h)*(1+r)
    tx = cx - tw/2.0
    ty = cy - th/2.0

    box['x'] = int(tx + 0.5)
    box['y'] = int(ty + 0.5)
    box['w'] = int(tw + 0.5)
    box['h'] = int(th + 0.5)
    r = box['x'] + box['w']
    b = box['y'] + box['h']

    if frame:
        box['x'] = 0 if box['x'] < 0 else box['x']
        box['y'] = 0 if box['y'] < 0 else box['y']
        r = frame[0]-1 if r >= frame[0] else r
        b = frame[1]-1 if b >= frame[1] else b
        box['w'] = r - box['x']
        box['h'] = b - box['y']

    return box


def is_inside(in_box, out_box):
    '''
    check if a box is inside of another
    :param in_box:
    :param out_box:
    :return:
    '''

    l0, t0, r0, b0 = in_box['x'], in_box['y'], in_box['x']+in_box['w'], in_box['y']+in_box['h']
    l1, t1, r1, b1 = out_box['x'], out_box['y'], out_box['x']+out_box['w'], out_box['y']+out_box['h']

    if l0 >= l1 and t0 >= t1 and r0 <= r1 and b0 <= b1:
        return True
    else:
        return False


def get_rotated_image_and_points(image, points, angle):
    '''
    학습 데이터를 in-plane rotation 시키기 위한 함수 (for augmentation)
    :param image: 원본 영상
    :param points: landmark 좌표
    :param angle: 회전할 각도
    :return: 회전된 영상 (rotated)와 해당 영상에서 landmark 점의 좌표 (mat_rot_pts)
    '''
    H, W, _ = image.shape
    mat_rot = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    rotated = cv2.warpAffine(image, mat_rot, (W, H))

    mat_arr = np.asarray(points, dtype=np.float).reshape((68, 2))
    mat_arr3 = np.concatenate([mat_arr, np.ones((68, 1), dtype=np.float)], axis=1)
    mat_rot_pts = np.transpose(np.matmul(mat_rot, np.transpose(mat_arr3)))

    # if DEBUG:
    #     for p in mat_rot_pts:
    #         cv2.circle(rotated, (int(p[0]), int(p[1])), 2, (0, 255, 0))
    #     cv2.imshow('rotated', rotated)

    return rotated, mat_rot_pts


def detect_face(detector, image, threshold=0.1):
    '''
    영상에서 얼굴을 검출한다.
    :param detector: face detector to use
    :param image: image to detect face
    :param threshold: confidence threshold for face detection
    :return: list of dict of faces {'x', 'y', 'w', 'h', 'conf'}
    '''

    H, W, _ = image.shape
    resize2detect = cv2.cvtColor(cv2.resize(image, (DETECTOR_INPUT_SIZE, DETECTOR_INPUT_SIZE)), cv2.COLOR_BGR2RGB)
    imtensor = np.expand_dims(resize2detect, 0)
    boxes, scores = detector.detect(imtensor)
    detected = []

    for i, b in enumerate(boxes):
        if scores[i] > threshold:
            detected.append(
                {'x': b[1] * W, 'y': b[0] * H, 'w': (b[3] - b[1]) * W, 'h': (b[2] - b[0]) * H, 'conf': scores[i]})

    return detected


def get_box_to_crop(image, points):
    '''
    영상에서 학습에 사용할 얼굴 영역 좌표를 추출한다. 우선 얼굴을 검출하고, 검출된 얼굴 중에서 landmark points의 bounding box 와 가장 overlap 이 큰 검출 box 를 찾는다.
    검출된 face box 가 충분히 confident 하면 해당 box 의 좌표를 return 한다. Return 되는 face box 의 좌표는 square_and_expand 함수를 통해 확장된 영역이다.
    :param image: 입력 영상
    :param points: image 에 대한 landmark annotation
    :return: 얼굴 영역의 box 좌표 {'x', 'y', 'w', 'h', 'conf'}
    '''
    if DEBUG:
        image_debug = deepcopy(image)

    H, W, _ = image.shape
    frame = {'x': 0, 'y': 0, 'w': W, 'h': H}

    pbbox = get_bounding_box(points)  # point bounding box

    if DEBUG:
        cv2.rectangle(image_debug, (int(pbbox['x']), int(pbbox['y'])),
                      (int(pbbox['x'] + pbbox['w']), int(pbbox['y'] + pbbox['h'])), (0, 255, 255))
        for p in points:
            cv2.circle(image_debug, (int(p[0] + 0.5), int(p[1] + 0.5)), 2, (0, 255, 0))

    detections = detect_face(detector, image, 0.5)
    best = find_best_matching_box(pbbox, detections, MIN_CONF=MIN_CONF,
                                  MIN_OVERLAP=MIN_OVERLAP)  # best matching detection

    if DEBUG:
        for d in detections:
            cv2.rectangle(image_debug, (int(d['x']), int(d['y'])), (int(d['x'] + d['w']), int(d['y'] + d['h'])),
                          (0, int(d['conf'] * 255), 0), 2)

    if not best:
        if DEBUG:
            save_path = os.path.join(os.path.dirname(s[0]), path_no_match)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            cv2.imwrite(os.path.join(save_path, os.path.basename(s[0])), image_debug)
        return None

    if DEBUG:  # draw best match in RED
        cv2.rectangle(image_debug, (int(best['x']), int(best['y'])),
                      (int(best['x'] + best['w']), int(best['y'] + best['h'])), (0, 0, 255))

    if EXTEND:
        best = square_and_expand(best, EXTEND_RATIO, (W, H))

    if DEBUG:  # draw best match extended in yellow
        cv2.rectangle(image_debug, (int(best['x']), int(best['y'])),
                      (int(best['x'] + best['w']), int(best['y'] + best['h'])), (255, 255, 0))

    # if not is_inside(pbbox, best):
    #     cnt_on_the_border += 1
    #
    #     if DEBUG:
    #         cv2.imshow('point_out_of_box', image_debug)
    #         cv2.waitKey(5)
    #         if DEBUG:
    #             save_path = os.path.join(os.path.dirname(s[0]), path_border)
    #             if not os.path.exists(save_path):
    #                 os.mkdir(save_path)
    #             cv2.imwrite(os.path.join(save_path, os.path.basename(s[0])), image_debug)
    #     continue

    # if not is_inside(best, frame):      # or just clip it?
    #     cnt_on_the_border += 1
    #
    #     if DEBUG:
    #         cv2.imshow('detect_out_of_frame', image_debug)
    #         cv2.waitKey(5)
    #         if DEBUG:
    #             save_path = os.path.join(os.path.dirname(s[0]), path_border)
    #             if not os.path.exists(save_path):
    #                 os.mkdir(save_path)
    #             cv2.imwrite(os.path.join(save_path, os.path.basename(s[0])), image_debug)
    #     continue

    return best


def save_training_data(image, points, crop_box, save_path, basename):
    '''
    원본 영상에서 얼굴 영역의 box가 결정되었을 때, 이를 학습용 별도 파일로 저장한다. 저장된 파일은 이후 .tfrecord 를 만드는 데 이용된다.
    :param image: input image
    :param points: landmark point annoation
    :param crop_box: face box to crop (to use as the training data)
    :param save_path: where to save the patch
    :param basename: image 파일의 basename
    :return: basename.png - 얼굴 영상, basename.npts - [0, +1]로 정규화된 landmark 좌표, basename.cpts - [-1, +1]로 정규화된 landmark 좌표
    '''

    cropped = image[int(crop_box['y']):int(crop_box['y'] + crop_box['h']),
              int(crop_box['x']):int(crop_box['x'] + crop_box['w']), :]

    cv2.imwrite(os.path.join(save_path, basename + '.png'), cropped)


    normed = normalize_points_with_rect(points, crop_box)
    pts = np.array(normed)
    pts.tofile(os.path.join(save_path, basename + '.npts'))

    cnormed = center_normalize_points_with_rect(points, crop_box)
    cpts = np.array(cnormed)
    cpts.tofile(os.path.join(save_path, basename + '.cpts'))


    if DEBUG and random() < 0.01:
        cv2.imshow('cropped', cropped)
        cv2.waitKey(1)


if __name__ == '__main__':
    detector = Detector(FACE_DETECTOR_PATH, 160, 1)

    if not os.path.exists(WRITE_PATH):
        os.makedirs(WRITE_PATH)

    path_less_point = 'less_point'
    path_border = 'border'
    path_no_match = 'no_match'
    path_crop = 'crop'

    cnt_less_point = 0
    cnt_no_overlap = 0
    cnt_on_the_border = 0
    cnt_generated = 0
    cnt_total = 0

    for P in IMAGES_DIR_PATHS:
        samples = get_files(P)
        cnt_total += len(samples)

        for i, s in enumerate(samples):
            # if random() > 1/RAND_FACTOR:
            #     continue

            print('%d: %s, %s' % (i, s[0], s[1]))
            image = cv2.imread(s[0], cv2.IMREAD_COLOR)
            points = np.reshape(read_pts(s[1]), (-1, 2))
            basename = os.path.splitext(os.path.basename(s[0]))[0]

            if len(points) != 68:
                cnt_less_point += 1
                if DEBUG:
                    save_path = os.path.join(os.path.dirname(s[0]), path_less_point)
                    if not os.path.exists(save_path):
                        os.mkdir(save_path)
                    cv2.imwrite(os.path.join(save_path, os.path.basename(s[0])), image)
                continue

            crop_box = get_box_to_crop(image, points)       # 원본 영상에서 얼굴 검출 후 data 저장
            if not crop_box:
                cnt_no_overlap += 1
                continue
            else:
                save_training_data(image, points, crop_box, WRITE_PATH, basename)
                cnt_generated += 1

            if ROTATE:                                      # Rotation augmentation 사용 시 한번 더 수행
                angle = random()*MAX_ROTATE*2 - MAX_ROTATE
                image, points = get_rotated_image_and_points(image, points, angle)

                crop_box = get_box_to_crop(image, points)
                if not crop_box:
                    cnt_no_overlap += 1
                    continue
                else:
                    save_training_data(image, points, crop_box, WRITE_PATH, basename + '_r')
                    cnt_generated += 1


    print('generated: %d, border_out: %d, no_overlap: %d, less_point: %d, total: %d'%
          (cnt_generated, cnt_on_the_border, cnt_no_overlap, cnt_less_point, cnt_total))