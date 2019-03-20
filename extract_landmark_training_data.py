import os
# from PIL import Image, ImageDraw
import cv2
from copy import deepcopy

# todo:
# 1) support menpo dataset
# 2) work with face detection info

DEBUG = False
EXTEND = 0.1
PATCH_SIZE = 56

def getFiles(folder_path):
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


def loadLandmarkData(filepath):
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

        if n_points != 68:
            return pts

        rf.readline()
        for i in range(n_points):
            x, y = rf.readline().split(' ')
            x, y, = float(x), float(y)
            pts.append([x, y])

        assert len(pts) == n_points

        return pts


def getBoundingBox(pts):
    '''
    Get bounding box for given facial landmarks
    :param pts:
    :return: bbox in [l, t, r, b]
    '''
    x = [p[0] for p in pts]
    y = [p[1] for p in pts]
    return {'x': min(x), 'y': min(y), 'w': max(x)-min(x), 'h': max(y)-min(y)}


def normalizePointsWithRect(pts, faceRect):
    '''
    normalize point coordiates w.r.t. the face rectanble
    :param pts: landmark points
    :param faceRect:  [l, t, r, b]
    :return:
    '''
    l, t = faceRect['x'], faceRect['y']
    w, h = faceRect['w'], faceRect['h']

    normed = [[(p[0] - l) / w, (p[1] - t) / h] for p in pts]
    return normed


def normalizePoints(pts):
    '''
    normalize point coordinates between 0 ~ 1
    :param pts:
    :return:
    '''
    return normalizePointsWithRect(pts, getBoundingBox(pts))


def loadDetection(image_path, det_name, th=0.1):
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
                    dets.append({'x':float(x), 'y':float(y), 'w':float(w), 'h':float(h), 'conf':float(c)})

        return dets

def getOverlap(box1, box2):
    l0, t0, r0, b0 = box1['x'], box1['y'], box1['x']+box1['w'], box1['y']+box1['h']
    l1, t1, r1, b1 = box2['x'], box2['y'], box2['x']+box2['w'], box2['y']+box2['h']

    if l0 > r1 or r0 < l1:
        return 0.0
    if t0 > b1 or b0 < t1:
        return 0.0

    li = max(l0, l1)
    ri = max(r0, r1)
    ti = max(t0, t1)
    bi = max(b0, b1)

    ai = (ri-li)*(bi-ti)
    a0 = (r0-l0)*(b0-t0)
    a1 = (r1-l1)*(b1-t1)

    return ai / (a0+a1-ai)


def findBestMatchingBox(target, detections):
    '''
    :param target: [l, t, r, b]
    :param detections: [{'x', 'y', 'w', 'h']
    :return:
    '''

    if len(detections) == 0:
        return None

    max_overlap = 0.0
    max_index = -1

    for i, b in enumerate(detections):
        if b['conf'] > 0.3:
            cur_overlap = getOverlap(target, b)
            if cur_overlap > max_overlap and cur_overlap > 0.1:
                max_index = i
                max_overlap = cur_overlap

    if max_index >= 0:
        return detections[max_index]
    else:
        return None


def square_and_expand(box, r=0.0):
    '''
    make box to square and expand by given ratio
    :param box:
    :return:
    '''

    w = box['w']
    h = box['h']
    cx = box['x'] + w/2.0
    cy = box['y'] + h/2.0

    tw = th = max(w, h)*(1+r)
    tx = cx - tw/2.0
    ty = cy - th/2.0

    # offset['x'] = box['x'] - int(tx + 0.5)
    # offset['y'] = box['y'] - int(ty + 0.5)

    box['x'] = int(tx + 0.5)
    box['y'] = int(ty + 0.5)
    box['w'] = int(tw + 0.5)
    box['h'] = int(th + 0.5)

    return box#, offset


def is_inside(in_box, out_box):
    l0, t0, r0, b0 = in_box['x'], in_box['y'], in_box['x']+in_box['w'], in_box['y']+in_box['h']
    l1, t1, r1, b1 = out_box['x'], out_box['y'], out_box['x']+out_box['w'], out_box['y']+out_box['h']

    if l0 >= l1 and t0 >= t1 and r0 <= r1 and b0 <= b1:
        return True
    else:
        return False


if __name__ == '__main__':
    PATH = [
        '/Users/gglee/Data/Landmark/300W/01_Indoor',
        '/Users/gglee/Data/Landmark/300W/02_Outdoor',
        '/Users/gglee/Data/Landmark/menpo_train_release'
    ]

    WRITE_PATH = '/Users/gglee/Data/Landmark/export'
    DET_NAME = '160v5'

    WRITE_PATH = os.path.join(WRITE_PATH, DET_NAME)

    if not os.path.exists(WRITE_PATH):
        os.makedirs(WRITE_PATH)

    import numpy as np

    cnt_less_point = 0
    cnt_on_the_border = 0
    cnt_generated = 0
    cnt_no_overlap = 0
    cnt_total = 0;

    for P in PATH:
        samples = getFiles(P)
        cnt_total += len(samples)

        for i, s in enumerate(samples):
            image = cv2.imread(s[0], cv2.IMREAD_COLOR)

            if DEBUG:
                image_debug = deepcopy(image)

            H, W, _ = image.shape
            points = loadLandmarkData(s[1])
            frame = {'x': 0, 'y': 0, 'w': W, 'h': H}

            if not points or len(points) != 68:
                cnt_less_point += 1
                continue

            pbbox = getBoundingBox(points)          # point bounding box
            cbox = pbbox

            if DEBUG:
                cv2.rectangle(image_debug, (int(pbbox['x']), int(pbbox['y'])), (int(pbbox['x']+pbbox['w']), int(pbbox['y']+pbbox['h'])), (0, 255, 255))
                for p in points:
                    cv2.circle(image_debug, (int(p[0]+0.5), int(p[1]+0.5)), 2, (0, 255, 0))

            if DET_NAME:
                dets = loadDetection(s[0], DET_NAME)
                best = findBestMatchingBox(pbbox, dets)     # best matching detection

                if not best:
                    cnt_no_overlap += 1

                    if DEBUG:
                        for d in dets:
                            if d['conf'] > 0.1:
                                cv2.rectangle(image_debug, (int(d['x']), int(d['y'])), (int(d['x']+d['w']), int(d['y']+d['h'])),
                                              (int(d['conf']*255, 0, int(d['conf']*255))))
                        cv2.imshow('no_best', image_debug)
                        cv2.waitKey(-1)
                    continue

                best = square_and_expand(best, EXTEND)

                if DEBUG:
                    cv2.rectangle(image_debug, (int(best['x']), int(best['y'])),
                                  (int(best['x'] + best['w']), int(best['y'] + best['h'])), (0, 0, 255))

                if not is_inside(pbbox, best):
                    cnt_on_the_border += 1

                    if DEBUG:
                        cv2.imshow('point_out_of_box', image_debug)
                        cv2.waitKey(-1)
                    continue

                if not is_inside(best, frame):      # or just clip it?
                    cnt_on_the_border += 1

                    if DEBUG:
                        cv2.imshow('point_out_of_frame', image_debug)
                        cv2.waitKey(-1)
                    continue

                cbox = best
            else:
                cbox = square_and_expand(pbbox, EXTEND)  # crop box

            # if DEBUG:
            #     cv2.rectangle(image_debug, (int(pbbox['x']), int(pbbox['y'])),
            #                   (int(pbbox['x'] + pbbox['w']), int(pbbox['y'] + pbbox['h'])),
            #                   (255, 0, 255), 1)
            #     cv2.rectangle(image_debug, (int(cbox['x']), int(cbox['y'])), (int(cbox['x'] + cbox['w']),
            #                                                                   int(cbox['y'] + cbox['h'])),
            #                   (0, 255, 0), 1)
            #     cv2.imshow('debug', image_debug)

            cropped = image[int(cbox['y']):int(cbox['y']+cbox['h']), int(cbox['x']):int(cbox['x']+cbox['w']), :]
            cropped = cv2.resize(cropped, (PATCH_SIZE, PATCH_SIZE))

            bname = os.path.splitext(os.path.basename(s[0]))[0]

            arr = np.array(cropped).astype(dtype=np.uint8)
            arr.tofile(os.path.join(WRITE_PATH, bname + '.img'))

            normed = normalizePointsWithRect(points, cbox)
            pts = np.array(normed)
            pts.tofile(os.path.join(WRITE_PATH, bname + '.pts'))

            w, h = PATCH_SIZE, PATCH_SIZE
            for p in normed:
                l, t, r, b = int(p[0]*w) - 1, int(p[1]*h) - 1, int(p[0]*w) + 1, int(p[1]*h) + 1
                cv2.rectangle(cropped, (l, t), (r, b), (0, 0, 255))

            cv2.imwrite(os.path.join(WRITE_PATH, bname + '.jpg'), cropped)
            cnt_generated += 1

            if DEBUG:
                cv2.imshow('cropped', cropped)
                cv2.waitKey(500)

            if i % 10 == 0:
                print('[%d / %d]: %s' %(i, len(samples), s[0]))
                cv2.imshow('patch', cropped)
                key = cv2.waitKey(10)

    print('generated: %d, border_out: %d, no_overlap: %d, less_point: %d, total: %d'%
          (cnt_generated, cnt_on_the_border, cnt_no_overlap, cnt_less_point, cnt_total))