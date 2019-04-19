import tensorflow as tf
import data_landmark as data
import net
import os
import numpy as np
import cv2
from operator import itemgetter

slim = tf.contrib.slim

tf.app.flags.DEFINE_string('models_dir', '/home/gglee/Data/Landmark/train/bests', 'where trained models are stored')

FLAGS = tf.app.flags.FLAGS

def load_settings(ckpt_path):
    path_setting = os.path.join(os.path.dirname(ckpt_path), 'train_setting.txt')
    assert os.path.exists(path_setting) and os.path.isfile(
        path_setting), 'train_setting.txt not exist for [%s]' % ckpt_path

    normalizer_fn = None
    normalizer_params = {}
    depth_multiplier = 1.0
    depth_gamma = 1.0
    is_color = True

    with open(path_setting, 'r') as rf:
        for l in rf:
            if 'is_color' in l:
                _, is_color = l.split(':')
            elif 'use_batch_norm' in l:
                _, bn_val = l.split(':')
                if bn_val.strip() == 'True':
                    normalizer_fn = slim.batch_norm
                    normalizer_params = {'is_training': False}
            elif 'depth_multiplier' in l:
                _, dm_val = l.split(':')
                depth_multiplier = float(dm_val.strip())
            elif 'depth_gamma' in l:
                _, dg_val = l.split(':')
                depth_gamma = float(dg_val.strip())

    return {'normalizer_fn': normalizer_fn, 'normalizer_params': normalizer_params, 'depth_multiplier': depth_multiplier,
            'depth_gamma': depth_gamma, 'is_color': is_color }


def restore_and_save(ckpt_path):
    # ckpt to pb & tflite
    # load train_setting
    settings = load_settings(ckpt_path)
    normalizer_fn = settings['normalizer_fn']
    normalizer_params = settings['normalizer_params']
    depth_multiplier = settings['depth_multiplier']
    depth_gamma = settings['depth_gamma']
    is_color = settings['is_color']

    # count_records = data.get_tfr_record_count(tfr_path)
    # dataset = data.load_tfrecord(tfr_path, batch_size=64, num_parallel_calls=16, is_color=is_color)
    # iterator = dataset.make_initializable_iterator()

    # BATCH_WIDTH = 8
    # BATCH_SIZE = BATCH_WIDTH*BATCH_WIDTH
    # NUM_ITER = int(count_records/BATCH_SIZE)
    dir_path = os.path.dirname(ckpt_path)
    pb_path = os.path.join(dir_path, 'frozen_model.pb')
    tflite_float_path = os.path.join(dir_path, 'landmark.float.tflite')
    tflite_qint8_path = os.path.join(dir_path, 'landmark.qint8.tflite')

    with tf.Session() as sess:

        # image, points = iterator.get_next()
        dph = tf.placeholder(tf.float32, (1, 56, 56, 3), 'input')

        with tf.variable_scope('model') as scope:
            predicts, _ = net.lannet(dph, is_training=False,
                                     depth_mul=depth_multiplier, depth_gamma=depth_gamma,
                                     normalizer_fn=normalizer_fn, normalizer_params=normalizer_params)
            saver = tf.train.Saver(tf.global_variables())

            # Load weights
            saver.restore(sess, ckpt_path)

            # Freeze the graph
            frozen_graph_def = tf.graph_util.convert_variables_to_constants(
                sess,
                sess.graph_def,
                ['model/lannet/fc7/Relu']) #output_node_names)

            # Save the frozen graph
            with open(pb_path, 'wb') as f:
                f.write(frozen_graph_def.SerializeToString())

    # if tf.__version__[:4] == "1.13":
    converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph(
        pb_path, ['input'], ['model/lannet/fc7/Relu'],
        input_shapes={'input': [1, 56, 56, 3]})
        # pb_path, input_node_names, output_node_names,
        # input_shapes=input_shapes)
    # else:
    #     converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph(
    #         pb_path, input_node_names, output_node_names,
    #         input_shapes=input_shapes)

    tflite_model = converter.convert()
    open(tflite_float_path, "wb").write(tflite_model)
    print('>> %s' % tflite_float_path)

    # converter.optimizations = [tf.contrib.lite.Optimize.OPTIMIZE_FOR_SIZE]
    # tflite_quant_model = converter.convert()
    # open(tflite_qint8_path, "wb").write(tflite_quant_model)
    # print('>> %s' % tflite_qint8_path)

    ## NOT WROKING
    converter.inference_type = tf.contrib.lite.constants.QUANTIZED_UINT8
    input_arrays = converter.get_input_arrays()
    converter.quantized_input_stats = {input_arrays[0]: (128.0, 128.0)}

    tflite_model = converter.convert()
    open(tflite_qint8_path, "wb").write(tflite_model)
    print('>> %s' % tflite_qint8_path)


# ckpt to pb & tflite
# with tf.Session() as sess:
#     build_model(is_training=False)
#     saver = tf.train.Saver(tf.global_variables())
#
#     # Load weights
#     saver.restore(sess, weight_path)
#
#     # Freeze the graph
#     frozen_graph_def = tf.graph_util.convert_variables_to_constants(
#         sess,
#         sess.graph_def,
#         output_node_names)
#
#     # Save the frozen graph
#     with open(pb_path, 'wb') as f:
#       f.write(frozen_graph_def.SerializeToString())
#
# if tf.__version__[:4] == "1.13":
#   converter = tf.lite.TFLiteConverter.from_frozen_graph(
#     pb_path, input_node_names, output_node_names,
#       input_shapes=input_shapes)
# else:
#   converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph(
#     pb_path, input_node_names, output_node_names,
#       input_shapes=input_shapes)
# tflite_model = converter.convert()
# open(tflite_path, "wb").write(tflite_model)

if __name__=='__main__':

    if not os.path.exists(FLAGS.models_dir) or not os.path.isdir(FLAGS.models_dir):
        print('check models_dir (not a dir or not exist): %s' % FLAGS.models_dir)
        exit()

    dirs = []
    for d in os.listdir(FLAGS.models_dir):
        path = os.path.join(FLAGS.models_dir, d)
        if os.path.isdir(path):
            dirs.append(path)

    dirs = sorted(dirs)

    for path in dirs:
        files = []
        for f in os.listdir(path):
            if f.endswith('.index'):
                step_num = int(f.split('-')[1].split('.')[0])
                files.append({'name': os.path.splitext(f)[0], 'steps': step_num})

        if len(files) == 0:
            continue

        largest = sorted(files, key=itemgetter('steps'), reverse=True)[0]['name']

        with tf.Graph().as_default():
            ckpt2use = os.path.join(path, largest)
            err = restore_and_save(ckpt2use)
