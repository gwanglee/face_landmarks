from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim

import sys
sys.path.append('/Users/gglee/Develop/models/research/slim/nets')

from mobilenet import mobilenet_v2

def mnet2(input_tensor, activation_fn=tf.nn.relu6, depth_multiplier=0.35):
    with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope()):
        logits, endpoints = mobilenet_v2.mobilenet(input_tensor, num_classes=68*2, depth_multiplier=depth_multiplier,
                                                   activation_fn=activation_fn)
        return logits, endpoints

        # num_classes = 1001,
        # depth_multiplier = 1.0,
        # scope = 'MobilenetV2',
        # conv_defs = None,
        # finegrain_classification_mode = False,
        # min_depth = None,
        # divisible_by = None,
        # activation_fn = None,
        # ** kwargs