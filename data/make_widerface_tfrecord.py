"""Convert Wider Face Dataset to TFRecord for object detection

Converts wider face detection dataset to TFRecords with a standard format allowing
  to use this dataset to train object detectors. The raw dataset can be
  downloaded from:
  http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/

  Wider face dataset contains 393,703 face in 32,203 images. 40%, 10% and 50% of them
  are training, validation and test sets.

  Running this script will create .tfrecord files for training and validation sets:
    wider_face_train.tfrecord and wider_face_val.tfrecord

Example usage:
    python object_detection/dataset_tools/make_widerface_tfrecord.py \
        --data_dir=/dataset/root/wider_face \
        --output_path=/where/to/write/tfrecords \
        --label_map_path=data/face_detection_label_map.pbtxt

Todo:
    * add directory check in main func.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import sys
import cv2
from random import shuffle
import tensorflow as tf
import PIL.Image
import os

import widerface_explorer

# sys.path.append('./models/research')
# sys.path.append('./models/research/slim')

from object_detection.utils import dataset_util

tf.app.flags.DEFINE_string('image_dir', '', 'Location of directory directory that contains:'
                           'test images subfolders')
tf.app.flags.DEFINE_string('gt_path', '', 'Location of ground truth text file')
tf.app.flags.DEFINE_string('output_path', '', 'Filepath where resulting .tfrecord will be saved')
tf.app.flags.DEFINE_string('negative_path', None, 'Negative images to add')
tf.app.flags.DEFINE_integer('negative_sample', 1, 'Sampling rate for negative data')

FLAGS = tf.app.flags.FLAGS

def prepare_example(data):
    """Make a data input to a tensorflow input example.

    Args:
      data:
      label_map_dict:

    Returns:
      None

    Todos:
      - Add all face attributes so we can ignore some faces for training
      - Black out small faces (black mask is sufficient?)
    """

    image_path = data['image_path']

    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()

    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width, height = image.size

    xmin, ymin, xmax, ymax = [], [], [], []
    classes, classes_text = [], []
    difficult_obj = []

    for anno in data['annos']:
        if not anno['invalid']:     # ignore invalid faces
            x0, y0 = float(anno['x']) / width, float(anno['y']) / height
            x1, y1 = float(anno['x']+anno['w']) / width, float(anno['y']+anno['h']) / height

            if not (x0 >= 0.0 and y0 >= 0.0 and x1 <= 1.0 and y1 <= 1.0):
                img = cv2.imread(image_path)
                print(img.shape)
                print('[%d x %d]' % (width, height), anno)
                cv2.rectangle(img, (anno['x'], anno['y']), (anno['x']+anno['w'], anno['y']+anno['h']), (0, 0, 255), 2)
                cv2.imshow('inval', img)
                cv2.waitKey(-1)

            # assert x0 >= 0.0 and y0 >= 0.0 and x1 <= 1.0 and y1 <= 1.0, print('%f, %f, %f, %f' % (x0, y0, x1, y1))

            xmin.append(x0)
            ymin.append(y0)
            xmax.append(x1)
            ymax.append(y1)

            classes.append(int(1))  # we have face only
            classes_text.append("face")
            difficult_obj.append(anno['invalid'])

    if len(xmin) == 0:
        return None

    # do we need filename / source id (actually use them?)
    example = tf.train.Example(features = tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(image_path.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(image_path.encode(''
                                                                        'utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),

        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),

    }))
    return example


def prepare_negative_example(image_path):

    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()

    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width, height = image.size

    xmin, ymin, xmax, ymax = [], [], [], []
    classes, classes_text = [], []
    difficult_obj = []

    # do we need filename / source id (actually use them?)
    example = tf.train.Example(features = tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(image_path.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(image_path.encode(''
                                                                        'utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),

        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),

    }))
    return example


def write_tfrecord(image_path, gt_path, tfrecord_path, negative_path=None, negative_sample=1):
    writer = tf.python_io.TFRecordWriter(tfrecord_path)

    wdb = widerface_explorer.wider_face_db(image_path, gt_path)
    num_positives = wdb.get_image_count()

    negative_images = []
    if negative_path:
        for root, dirs, files in os.walk(negative_path):
            for f in files:
                if f.split('.')[-1].lower() in ['jpg', 'png']:
                    negative_images.append(os.path.join(root, f))
    num_negatives = len(negative_images)

    if num_negatives > 0 and negative_sample > 1:
        shuffle(negative_images)
        num_to_sample = max(1, int(len(negative_images) / negative_sample))
        negative_images = negative_images[0:num_to_sample]
        num_negatives = num_to_sample

    idx = range(num_positives+num_negatives)
    if negative_path:
        for i in range(num_positives, num_positives+num_negatives):
            idx[i] *= -1
    shuffle(idx)

    for i, cur in enumerate(idx):
        if cur >= 0:
            data = wdb.get_annos_by_image_index(cur)
            example = prepare_example(data)
            image_path = data['image_path']
        else:
            cur *= -1
            image_path = negative_images[cur-num_positives]
            example = prepare_negative_example(image_path)

        if example is not None:
            writer.write(example.SerializeToString())

        if i % 100 == 0:
            print("image %d: %s" % (i, image_path))

    writer.close()


def integrity_check(tfr_path):
    itr = tf.python_io.tf_record_iterator(path=tfr_path)
    cnt = 0
    with tf.Session() as sess:
        for r in itr:
            example = tf.train.Example()
            example.ParseFromString(r)

            jpg = tf.decode_raw(example.features.feature['image/encoded'].bytes_list.value[0], tf.uint8)
            jpg_ = sess.run(jpg)

            print('[%d] %s' % (cnt, example.features.feature['image/filename'].bytes_list.value[0]))
            decoded = cv2.imdecode(jpg_, -1)
            # cv2.imshow('image', decoded)
            # cv2.waitKey(1)
            cnt += 1


def main(_):
    IMAGE_DIR = FLAGS.image_dir
    GT_PATH = FLAGS.gt_path
    OUTPUT_PATH = FLAGS.output_path
    NEGATIVE_PATH = FLAGS.negative_path
    NEGATIVE_SAMPLE = FLAGS.negative_sample

    write_tfrecord(IMAGE_DIR, GT_PATH, OUTPUT_PATH, NEGATIVE_PATH, NEGATIVE_SAMPLE)
    # integrity_check(OUTPUT_PATH)
    
if __name__ == '__main__':
    tf.app.run()

# run example
#python make_widerface_tfrecord.py --image_dir=/Users/gglee/Data/WiderRefine/train_random_150 --output_path=/Users/gglee/Data/WiderRefine/train_random_150/train_random_150.tfrecord --gt_path=/Users/gglee/Data/WiderRefine/train_random_150/wider_refine_train_gt.txt

#  python make_widerface_tfrecord.py --image_dir=/home/gglee/Data/WiderFace/WIDER_refine_0422/ --output_path=/home/gglee/Data/WiderFace/WIDER_refine_0422/wider_train_0422.tfrecord --gt_path=/home/gglee/Data/WiderFace/WIDER_refine_0422/refine_train_0422.txt --negative_path=/home/gglee/Data/WiderFace/negatives/

# python ./data/make_widerface_tfrecord.py --image_dir=/Users/gglee/Data/face_train --output_path=/Users/gglee/Data/face_train/face_0521.tfrecord --gt_path=/Users/gglee/Data/face_train/gt.txt --negative_path=/Users/gglee/Data/face_negative/
