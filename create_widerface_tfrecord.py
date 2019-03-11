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
    python object_detection/dataset_tools/create_widerface_tfrecord.py \
        --data_dir=/dataset/root/wider_face \
        --output_path=/where/to/write/tfrecords \
        --label_map_path=data/face_detection_label_map.pbtxt

Todo:
    * add directory check in main func.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from shutil import copyfile

import hashlib
import io
import sys
import cv2
from random import shuffle
import tensorflow as tf
import PIL.Image

import widerface_explorer

sys.path.append('./models/research')
sys.path.append('./models/research/slim')

from object_detection.utils import dataset_util

tf.app.flags.DEFINE_string('image_dir', '', 'Location of directory directory that contains:'
                           'test images subfolders')
tf.app.flags.DEFINE_string('gt_path', '', 'Location of ground truth text file')
tf.app.flags.DEFINE_string('output_path', '', 'Filepath where resulting .tfrecord will be saved')
#tf.app.flags.DEFINE_string('label_map_path', 'data/face_detection_label_map.pbtxt',    #assume we have only one class (1:face)
#                           'Path to label map proto.')

FLAGS = tf.app.flags.FLAGS

def prepare_example(data, label_map_dict):
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

    # if min_size < 1.0:  # filter out small faces
    #
    #     assert image is not None, image_path
    #     height, width = image.shape[:2]
    #
    #     for anno in data['annos']:
    #         # check box integrity
    #         if anno['w'] < 0:
    #             anno['w'] = anno['w']*-1
    #             anno['x'] = anno['x']-anno['w']
    #
    #         if anno['h'] < 0:
    #             anno['h'] = anno['h']*-1
    #             anno['y'] = anno['y']-anno['h']
    #
    #         x, y = anno['x'], anno['y']
    #         w, h = anno['w'], anno['h']
    #
    #         if x == 0.0 and y == 0.0 and w == 0.0 and h == 0.0:
    #             continue
    #
    #         if is_smaller(anno, width, height, min_size):   # mask out face if too small
    #             # some data are actually all zeros (to provide negative images?)
    #             # TODO: need to check all-zero annotations are okay with training code
    #             assert w>0 and h>0, '%s\n(x, y, w, h)=(%.2f, %.2f, %.2f, %.2f)'%(image_path, x, y, w, h)
    #             rrect = ((int(x+w/2), int(y+h/2)), (int(w), int(h)), 0.0)
    #             cv2.ellipse(image, rrect, (0, 0, 0), -1)
    #
    #     # where to save? need to save?
    #     basename = os.path.basename(image_path)
    #     parent_dir = image_path.rsplit('/', 2)[1]
    #     tmp_path = os.path.join(os.path.join(FLAGS.data_dir, 'temp_%f'%(min_size)), parent_dir)
    #
    #     if os.path.exists(tmp_path) is False:
    #         os.makedirs(tmp_path)
    #
    #     write_path = os.path.join(tmp_path, basename)
    #     cv2.imwrite(write_path, image)
    #     image_path = write_path
    #
    #     cv2.imshow('masked', image)
    #     cv2.waitKey(1)

def write_tfrecord(image_path, gt_path, tfrecord_path):
    writer = tf.python_io.TFRecordWriter(tfrecord_path)

    wdb = widerface_explorer.wider_face_db(image_path, gt_path)
    total_images = wdb.get_image_count()
    idx = range(total_images)
    shuffle(idx)
    
    for i in idx:
        data = wdb.get_annos_by_image_index(i)

        example = prepare_example(data, None)
        if example is not None:
            writer.write(example.SerializeToString())

        if i % 100 == 0:
            print("image %d: %s" % (i, data['image_path']))

    writer.close()

def main(_):
    IMAGE_DIR = FLAGS.image_dir
    GT_PATH = FLAGS.gt_path
    OUTPUT_PATH = FLAGS.output_path

    write_tfrecord(IMAGE_DIR, GT_PATH, OUTPUT_PATH)
    
if __name__ == '__main__':
    tf.app.run()

# run example
# python create_widerface_tfrecord.py --data_dir=/home/gglee/Data/WiderFace/ --output_path=/home/gglee/Data/WiderFace/tfrecords/wider_mask_0.03.tfrecord --min_size=0.03