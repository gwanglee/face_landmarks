# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains a variant of the LeNet model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim

def lannet(inputs, is_training=False, deopout_keep_prob=0.5, scope=None, depth_mul=1, normalizer_fn=None, normalizer_params={},
           regularizer=None):
  end_points = {}

  with tf.variable_scope('lannet', [inputs], reuse=tf.AUTO_REUSE):
    with slim.arg_scope([slim.conv2d, slim.max_pool2d], padding='SAME'):
        with slim.arg_scope([slim.conv2d], normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,
                            weights_regularizer=regularizer):
            net = end_points['conv1'] = slim.conv2d(inputs, 8*depth_mul, [3, 3], scope='conv1')
            net = end_points['pool1'] = slim.max_pool2d(net, [2, 2], stride=2, scope='pool1')
            net = end_points['conv2'] = slim.conv2d(net, 16*depth_mul, [3, 3], scope='conv2')
            net = end_points['pool2'] = slim.max_pool2d(net, [2, 2], stride=2, scope='pool2')
            net = end_points['conv3'] = slim.conv2d(net, 32*depth_mul, [3, 3], scope='conv3')
            net = end_points['pool3'] = slim.max_pool2d(net, [2, 2], stride=2, scope='pool3')
            net = end_points['conv4'] = slim.conv2d(net, 64*depth_mul, [3, 3], scope='conv4')
            net = end_points['pool4'] = slim.max_pool2d(net, [2, 2], stride=2, scope='pool4')
            net = end_points['conv6'] = slim.conv2d(net, 128*depth_mul, [3, 3], scope='conv5')
            net = slim.flatten(net)
            net = slim.fully_connected(net, 256*depth_mul, scope='fc6')  # none -> fc6
            net = slim.fully_connected(net, 68*2, activation_fn=None, scope='fc7')

        return net, end_points

def arg_scope(weight_decay=0.0005):
  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
      activation_fn=tf.nn.relu6) as sc:
    return sc