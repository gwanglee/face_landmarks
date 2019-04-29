from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
from random import shuffle
import tensorflow as tf
import train
import os
import numpy as np


def prepare_example(data, use_gray=False, size=56):
    assert os.path.exists(data[0])
    assert os.path.exists(data[1])

    SIZE = size

    img_path = data[0]
    pts_path = data[1]

    img = cv2.imread(img_path)
    img = cv2.resize(img, (SIZE, SIZE))
    pts = np.fromfile(pts_path, dtype=np.float64).astype(np.float32)
    pts = np.reshape(pts, (len(pts), 1))

    if use_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = np.reshape(img, (1, -1))

    example = tf.train.Example(features = tf.train.Features(feature={
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=img.tobytes())),
        'points': tf.train.Feature(float_list=tf.train.FloatList(value=pts))
    }))

    # print(example)

    return example


def make_tfrecord(tfr_path, lists, use_gray=False, size=56):
    writer = tf.python_io.TFRecordWriter(tfr_path)

    for i, l in enumerate(lists):
        example = prepare_example(l, use_gray=use_gray, size=size)
        if example:
            writer.write(example.SerializeToString())

        if i % 50:
            print(('%d: %s') % (i, l[0]))

    writer.close()


def main(_):
    DATA_PATH = '/Users/gglee/Data/Landmark/export/0424'
    TRAIN_RATIO = 0.9
    USE_GRAY=True
    SIZE = 56

    TRAIN_TFR_PATH = '/Users/gglee/Data/Landmark/export/0424.%d.gray.train.tfrecord' % SIZE
    VAL_TFR_PATH = '/Users/gglee/Data/Landmark/export/0424.%d.gray.val.tfrecord' % SIZE

    list_files = train.prepare_data_list(DATA_PATH, '.npts')
    shuffle(list_files)

    pos_split = int(len(list_files)*TRAIN_RATIO)
    train_list = list_files[:pos_split]
    val_list = list_files[pos_split:]

    print(len(train_list), len(val_list))

    make_tfrecord(TRAIN_TFR_PATH, train_list, use_gray=USE_GRAY, size=SIZE)
    make_tfrecord(VAL_TFR_PATH, val_list, use_gray=USE_GRAY, size=SIZE)

if __name__=='__main__':
    tf.app.run()