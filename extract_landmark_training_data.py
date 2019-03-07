import os
# from PIL import Image, ImageDraw
import cv2

# todo:
# 1) support menpo dataset
# 2) work with face detection info

def getFiles(folder_path):
    '''
    Return list of filename pairs of [image, annotation]
    :param folder_path:
    :return: list of [/path/filename.png, /path/filename/pts]
    '''
    assert os.path.exists(folder_path) and os.path.isdir(folder_path)

    files = []
    for f in os.listdir(folder_path):
        if not f.startswith('.') and f.endswith('png'):
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
    return [min(x), min(y), max(x), max(y)]


def normalizePointsWithRect(pts, faceRect):
    '''
    normalize point coordiates w.r.t. the face rectanble
    :param pts: landmark points
    :param faceRect:  [l, t, r, b]
    :return:
    '''
    bbox = faceRect
    l, t = bbox[0], bbox[1]
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

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
    l0, t0, r0, b0 = box1[0], box1[1], box1[2], box1[3]
    l1, t1, r1, b1 = box2[0], box2[1], box2[2], box2[3]

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

    max_overlap = 0.0
    max_index = 0

    for i, b in enumerate(detections):
        cur_overlap = getOverlap(target, [b['x'], b['y'], b['x']+b['w'], b['y']+b['h']])
        if cur_overlap > max_overlap:
            max_index = i
            max_overlap = cur_overlap

    return detections[max_index]


def square_and_expand(box, r=0.0):
    '''
    make box to square and expand by given ratio
    :param box:
    :return:
    '''
    W = box[2] - box[0]
    H = box[3] - box[1]

    if W > H:
        diff = (W - H)/2
        box[1] -= diff
        box[3] += diff
    else:
        diff = (H - W)/2
        box[0] -= diff
        box[2] += diff

    W = box[2] - box[0]
    H = box[3] - box[1]

    box[0] -= W*r/2
    box[1] -= H*r/2
    box[2] += W*r/2
    box[3] += W*r/2

    return box


if __name__ == '__main__':
    PATH = ['/Users/gglee/Data/Landmark/300W/01_Indoor', '/Users/gglee/Data/Landmark/300W/02_Outdoor',
            '/Users/gglee/Data/Landmark/menpo_train_release']
    WRITE_PATH = '/Users/gglee/Data/Landmark/export'
    DET_NAME = '1207'
    PATCH_SIZE = 64

    if not os.path.exists(WRITE_PATH):
        os.makedirs(WRITE_PATH)

    import numpy as np

    for P in PATH:
        samples = getFiles(P)

        for i, s in enumerate(samples):
            image = cv2.imread(s[0], cv2.IMREAD_COLOR)
            points = loadLandmarkData(s[1])

            if not points or len(points) != 68:
                continue

            bbox = getBoundingBox(points)
            cx, cy = (bbox[0] + bbox[2])/2.0, (bbox[1] + bbox[3])/2.0
            w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]

            ibbox = [cx-w/2, cy-h/2, cx+w/2, cy+h/2]

            if DET_NAME:
                dets = loadDetection(s[0], DET_NAME)
                best = findBestMatchingBox(ibbox, dets)
                ibbox = [best['x'], best['y'], best['x']+best['w'], best['y']+best['h']]

            ibbox = square_and_expand(ibbox, 0.1)

            W, H = image.shape[0:2]
            if ibbox[0] < 0 or ibbox[1] < 0 or ibbox[2] > W or ibbox[3] > H:
                continue

            cropped = image[int(ibbox[1]):int(ibbox[3]), int(ibbox[0]):int(ibbox[2]), :]
            cropped = cv2.resize(cropped, (PATCH_SIZE, PATCH_SIZE))

            bname = os.path.splitext(os.path.basename(s[0]))[0]

            arr = np.array(cropped).astype(dtype=float)
            arr.tofile(os.path.join(WRITE_PATH, bname + '.img'))

            normed = normalizePointsWithRect(points, [int(ibbox[0]), int(ibbox[1]), int(ibbox[2]), int(ibbox[3])])
            pts = np.array(normed)
            pts.tofile(os.path.join(WRITE_PATH, bname + '.pts'))

            if i % 10 == 0:
                w, h = PATCH_SIZE, PATCH_SIZE
                for p in normed:
                    l, t, r, b = int(p[0]*w) - 1, int(p[1]*h) - 1, int(p[0]*w) + 1, int(p[1]*h) + 1
                    cv2.rectangle(cropped, (l, t), (r, b), (0, 0, 255))

                cv2.imshow('patch', cropped)
                key = cv2.waitKey(10)