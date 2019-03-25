from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
from random import shuffle
import tensorflow as tf
import train
import os
import numpy as np


def prepare_example(data):
    assert os.path.exists(data[0])
    assert os.path.exists(data[1])

    img_path = data[0]
    pts_path = data[1]

    img = np.fromfile(img_path, dtype=np.uint8)
    pts = np.fromfile(pts_path, dtype=np.float64).astype(np.float32)
    pts = np.reshape(pts, (len(pts), 1))
    # print(pts.shape)

    # cv2.imshow('loaded', np.reshape(img, (56, 56, 3)))
    # cv2.waitKey(-1)

    example = tf.train.Example(features = tf.train.Features(feature={
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=img.tobytes())),
        'points': tf.train.Feature(float_list=tf.train.FloatList(value=pts))
    }))

    # print(example)

    return example


def make_tfrecord(tfr_path, lists):
    writer = tf.python_io.TFRecordWriter(tfr_path)

    for i, l in enumerate(lists):
        example = prepare_example(l)
        if example:
            writer.write(example.SerializeToString())

        if i % 50:
            print(('%d: %s') % (i, l[0]))

    writer.close()


def main(_):
    DATA_PATH = '/Users/gglee/Data/Landmark/export/160v5'
    TRAIN_RATIO = 0.9

    TRAIN_TFR_PATH = '/Users/gglee/Data/Landmark/export/160v5.0322.train.tfrecord'
    VAL_TFR_PATH = '/Users/gglee/Data/Landmark/export/160v5.0322.val.tfrecord'

    list_files = train.prepare_data_list(DATA_PATH)
    shuffle(list_files)

    pos_split = int(len(list_files)*TRAIN_RATIO)
    train_list = list_files[:pos_split]
    val_list = list_files[pos_split:]

    print(len(train_list), len(val_list))

    make_tfrecord(TRAIN_TFR_PATH, train_list)
    make_tfrecord(VAL_TFR_PATH, val_list)


if __name__=='__main__':
    tf.app.run()