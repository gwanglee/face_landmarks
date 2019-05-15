import os
from xml.etree import ElementTree
from random import shuffle
import cv2
from operator import itemgetter
import tensorflow as tf
import PIL.Image
import io
import hashlib
import sys
sys.path.append('/Users/gglee/Develop/models/research')
sys.path.append('/Users/gglee/Develop/models/research/slim')

from object_detection.utils import dataset_util

# def prepare_example(image_path, boxes):
#     with tf.gfile.GFile(image_path, 'rb') as fid:
#         encoded_jpg = fid.read()
#         encoded_jpg_io = io.BytesIO(encoded_jpg)
#         image = PIL.Image.open(encoded_jpg_io)
#         if image.format != 'JPEG':
#             print (image.format)
#             return None
#         key = hashlib.sha256(encoded_jpg).hexdigest()
#         width, height = image.size
#
#         xmin, ymin, xmax, ymax = [], [], [], []
#         classes, classes_text = [], []
#         difficult_obj = []
#
#         for b in boxes:
#             xmin.append(float(b[0]) / width)
#             ymin.append(float(b[1]) / height)
#             xmax.append(float(b[2]) / width)
#             ymax.append(float(b[3]) / height)
#             label = 'face'
#             classes.append(1)
#             classes_text.append(label)
#             difficult_obj.append(0)
#
#         example = tf.train.Example(features=tf.train.Features
#             (feature={
#             'image/height': dataset_util.int64_feature(height),
#             'image/width': dataset_util.int64_feature(width),
#             'image/filename': dataset_util.bytes_feature(image_path.encode('utf8')),
#             'image/source_id': dataset_util.bytes_feature(image_path.encode('utf8')),
#             'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
#             'image/encoded': dataset_util.bytes_feature(encoded_jpg),
#             'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
#
#             'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
#             'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
#             'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
#             'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
#             'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
#             'image/object/class/label': dataset_util.int64_list_feature(classes),
#             'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
#         }))
#         return example


def parse_xml(xml_path):
    root = ElementTree.parse(xml_path).getroot()

    width, height = 0, 0

    for s in root.findall('size'):
        for w in s.findall('width'):
            width = int(w.text)
        for h in s.findall('height'):
            height = int(h.text)

    assert width != 0 and height != 0, 'width or height is zero in %s' % xml_path
    faces = []

    for obj in root.findall('object'):
        for name in obj.findall('name'):
            if name.text == 'face':
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text) #/ float(width)
                ymin = int(bbox.find('ymin').text) #/ float(height)
                xmax = int(bbox.find('xmax').text) #/ float(width)
                ymax = int(bbox.find('ymax').text) #/ float(height)

                faces.append({'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax,
                              'width': (xmax-xmin), 'height': (ymax-ymin)})

    return faces


def get_bbox(objs):
    if objs is not None and len(objs) > 0:
        left = sorted(objs, key=itemgetter('xmin'))[0]['xmin']
        right = sorted(objs, key=itemgetter('xmax'), reverse=True)[0]['xmax']
        top = sorted(objs, key=itemgetter('ymin'))[0]['ymin']
        bottom = sorted(objs, key=itemgetter('ymax'), reverse=True)[0]['ymax']

        return {'xmin': left, 'ymin': top, 'xmax': right, 'ymax': bottom}
    else:
        return None


if __name__=='__main__':
    IMAGES_ROOT = '/Users/gglee/Data/face_ours'
    image_lists = []
    WRITE_FILE = '/Users/gglee/Data/face_ours/face_ours.txt'

    for dir, subs, files in os.walk(IMAGES_ROOT):
        for f in files:
            if not f.startswith('.') and f.split('.')[-1].lower() in ['jpg', 'png']:
                # print(os.path.join(dir, f))
                image_lists.append(os.path.join(dir, f))

    print('%d files found from %s' % (len(image_lists), IMAGES_ROOT))

    # shuffle(image_lists)

    with open(WRITE_FILE, 'w') as wf:

        for i, ip in enumerate(image_lists):
            print('%d: %s' % (i, ip))
            xml_path = os.path.splitext(ip)[0] + '.xml'

            if os.path.exists(xml_path):
                image = cv2.imread(ip)
                height, width = image.shape[0:2]

                bbox = parse_xml(xml_path)
                dirname = os.path.dirname(ip).split('/')[-1]
                filename = os.path.basename(ip)
                p2write = os.path.join(dirname, filename)
                wf.write('%s\r\n' % p2write)

                if bbox is not None and len(bbox) > 0:
                    wf.write('%d\r\n' % len(bbox))

                    for b in bbox:
                        cv2.rectangle(image, (int(b['xmin']), int(b['ymin'])), (int(b['xmax']), int(b['ymax'])),
                                      (0, 0, 255), 2)
                        wf.write('%d %d %d %d 0 0 0 0 0 0\r\n' % (b['xmin'], b['ymin'], b['xmax']-b['xmin'], b['ymax']-b['ymin']))
                else:
                    wf.write('%d\r\n' % 0)

                    # min_size = 0.08
                    # small = filter(lambda x: x['width'] < min_size, bbox)
                    # large = filter(lambda x: x['width'] >= min_size, bbox)
                    #
                    # abb = get_bbox(bbox)
                    # sbb = get_bbox(small)
                    # lbb = get_bbox(large)

                    # if both bbox are None -> background
                    # if large_bbox only --> happy
                    # if small bbox only --> see if wd can use it after crop
                    # if small bbox & large bbox does not overlap --> crop to include large bbox only
                    # is small & large overlap -> drop it

                    # if abb is not None:
                    #     cv2.rectangle(image, (int(abb['xmin']*width), int(abb['ymin']*height)),
                    #                   (int(abb['xmax']*width), int(abb['ymax']*height)), (255, 255, 255), 1)
                    #
                    # if sbb is not None:
                    #     cv2.rectangle(image, (int(sbb['xmin']*width), int(sbb['ymin']*height)),
                    #                   (int(sbb['xmax']*width), int(sbb['ymax']*height)), (0, 255, 0), 1)
                    # if lbb is not None:
                    #     cv2.rectangle(image, (int(lbb['xmin'] * width), int(lbb['ymin'] * height)),
                    #               (int(lbb['xmax'] * width), int(lbb['ymax'] * height)), (255, 0, 0), 1)


                # if width > 1024:
                #     image = cv2.resize(image, (1024, 768))
                # cv2.imshow('image', image)
                # cv2.waitKey(10)