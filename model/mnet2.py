from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim

import sys
sys.path.append('/Users/gglee/Develop/models/research/slim/nets')

from mobilenet import mobilenet_v2

def mnet2(input_tensor, activation_fn=tf.nn.relu6, depth_multiplier=0.35):
    end_points = {}

    with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope()):
        feature_map = end_points['ftr_map'] = mobilenet_v2.mobilenet_base(input_tensor, depth_multiplier=depth_multiplier)

        net = slim.flatten(feature_map)
        net = slim.fully_connected(net, (256 * depth_multiplier), scope='fc6')  # none -> fc6
        net = slim.fully_connected(net, 68 * 2, scope='fc7')

    return net, end_points
        # logits, endpoints = mobilenet_v2.mobilenet(input_tensor, num_classes=68*2, depth_multiplier=depth_multiplier,
        #                                            activation_fn=activation_fn)
        # return logits, endpoints