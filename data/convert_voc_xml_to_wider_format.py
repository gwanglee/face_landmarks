#-*- coding: utf-8 -*-
'''
VOC 포맷으로 되어있는 annotation (.xml)을 WiderFace 형태로 변환한다.
'''

import os
from xml.etree import ElementTree
from random import shuffle
import cv2


def parse_xml(xml_path):
    '''
    VOC annotation을 읽어들인 후 해당 내용을 dictionary 에 담아 반환한다. XML 파일 내에 face class 만 있다고 가정한다 (다른 class 는 무시된다).
    :param xml_path: VOC annotation 경로
    :return: list of dict of {'xmin', 'ymin', 'xmax', 'ymax', 'width', 'height' }
    '''
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


if __name__=='__main__':
    IMAGES_ROOT = '/Users/gglee/Data/face_ours'                 # VOC format data (image, xml) 경로
    WRITE_FILE = '/Users/gglee/Data/face_ours/face_ours.txt'    # WiderFace format 으로 저장할 gt 파일의 경로

    image_lists = []

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

                cv2.imshow('image', image)
                cv2.waitKey(5)