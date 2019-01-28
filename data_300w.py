import os
from PIL import Image, ImageDraw

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

        version = int(l1.split(' ')[1])
        n_points = int(l2.split(' ')[1])

        assert version == 1 and n_points == 68
        pts = []

        rf.readline()
        for i in range(n_points):
            x, y = rf.readline().split(' ')
            x, y, = float(x), float(y)
            pts.append([x, y])

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


def normalizePoints(pts):
    bbox = getBoundingBox(pts)
    l, t = bbox[0], bbox[1]
    w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]

    normed = [[(p[0]-l)/w, (p[1]-t)/h] for p in pts]
    return normed

if __name__ == '__main__':
    PATH = ['/home/gglee/Data/300W/01_Indoor', '/home/gglee/Data/300W/02_Outdoor']
    WRITE_PATH = '/home/gglee/Data/300W/export'
    if not os.path.exists(WRITE_PATH):
        os.makedirs(WRITE_PATH)

    import numpy as np
    import psutil

    for P in PATH:
        samples = getFiles(P)

        for i, s in enumerate(samples):
            image = Image.open(s[0])
            points = loadLandmarkData(s[1])

            bbox = getBoundingBox(points)
            cx, cy = (bbox[0] + bbox[2])/2.0, (bbox[1] + bbox[3])/2.0
            w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]

            ibbox = [int(cx-w/2), int(cy-h/2), int(cx+w/2), int(cy+h/2)]
            cropped = image.crop(ibbox)

            bname = os.path.splitext(os.path.basename(s[0]))[0]
            arr = np.array(cropped)
            normed = normalizePoints(points)
            pts = np.array(normed)
            arr.tofile(os.path.join(WRITE_PATH, bname + '.img'))
            pts.tofile(os.path.join(WRITE_PATH, bname + '.pts'))

            if i % 10 == 0:
                draw = ImageDraw.Draw(cropped)
                w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
                for p in normed:
                    l, t, r, b = int(p[0]*w) - 1, int(p[1]*h) - 1, int(p[0]*w) + 1, int(p[1]*h) + 1
                    draw.ellipse((l, t, r, b))

                del draw

                for proc in psutil.process_iter():
                    if proc.name() == 'display':
                        proc.kill()

                cropped.show()

    for proc in psutil.process_iter():
        if proc.name() == 'display':
            proc.kill()