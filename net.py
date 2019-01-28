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
# import nets.inception_v3 as inception_v3

slim = tf.contrib.slim


def lannet(inputs,
        scope=None):
  """Creates a variant of the LeNet model.

  Note that since the output is a set of 'logits', the values fall in the
  interval of (-infinity, infinity). Consequently, to convert the outputs to a
  probability distribution over the characters, one will need to convert them
  using the softmax function:

        logits = lenet.lenet(images, is_training=False)
        probabilities = tf.nn.softmax(logits)
        predictions = tf.argmax(logits, 1)

  Args:
    images: A batch of `Tensors` of size [batch_size, height, width, channels].
    num_classes: the number of classes in the dataset. If 0 or None, the logits
      layer is omitted and the input features to the logits layer are returned
      instead.
    is_training: specifies whether or not we're currently training the model.
      This variable will determine the behaviour of the dropout layer.
    dropout_keep_prob: the percentage of activation values that are retained.
    prediction_fn: a function to get predictions out of logits.
    scope: Optional variable_scope.

  Returns:
     net: a 2D Tensor with the logits (pre-softmax activations) if num_classes
      is a non-zero integer, or the inon-dropped-out nput to the logits layer
      if num_classes is 0 or None.
    end_points: a dictionary from components of the network to the corresponding
      activation.
  """
  end_points = {}

  with tf.variable_scope(scope, 'Darknet', [inputs]):
    with slim.arg_scope([slim.conv2d, slim.max_pool2d], padding='SAME'):
        net = end_points['conv1'] = slim.conv2d(inputs, 8, [3, 3], scope='conv1')
        net = end_points['pool1'] = slim.max_pool2d(net, [2, 2], stride=2, scope='pool1')
        net = end_points['conv2'] = slim.conv2d(net, 16, [3, 3], scope='conv2')
        net = end_points['pool2'] = slim.max_pool2d(net, [2, 2], stride=2, scope='pool2')
        net = end_points['conv3'] = slim.conv2d(net, 32, [3, 3], scope='conv3')
        net = end_points['pool3'] = slim.max_pool2d(net, [2, 2], stride=2, scope='pool3')
        net = end_points['conv4'] = slim.conv2d(net, 64, [3, 3], scope='conv4')
        net = end_points['pool4'] = slim.max_pool2d(net, [2, 2], stride=2, scope='pool4')
        net = end_points['conv5'] = slim.conv2d(net, 128, [3, 3], scope='conv5')
        net = end_points['pool5'] = slim.max_pool2d(net, [2, 2], stride=2, scope='pool5')
        net = end_points['conv6'] = slim.conv2d(net, [2, 2], padding='VALID', scope='conv6')
        net = slim.flatten(net)
        slim.fully_connected(net, 64*2)
        return net, end_points

        net = slim.flatten(net)
        end_points['Flatten'] = net

def darknet(inputs,
             num_classes=1000,
             is_training=True,
             dropout_keep_prob=0.8,
             min_depth=16,
             depth_multiplier=1.0,
             prediction_fn=slim.softmax,
             spatial_squeeze=True,
             reuse=None,
             create_aux_logits=True,
             scope='Darknet',
             global_pool=False):
    with tf.variable_scope(scope, 'Darknet', [inputs], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            net, end_points = darknet_base(inputs, scope=scope)

            net = end_points['pool7'] = slim.max_pool2d(net, [2, 2], stride=2, scope='pool7')
            net = end_points['conv8'] = slim.conv2d(net, 1024, [3, 3], scope='conv8')

            with tf.variable_scope('Logits'):
                if global_pool:
                # Global average pooling.
                    net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='GlobalPool')
                    end_points['global_pool'] = net
                else:
                # Pooling with a fixed kernel size.
                    kernel_size = inception_v3._reduced_kernel_size_for_small_input(net, [8, 8])
                    net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                                          scope='AvgPool_1a_{}x{}'.format(*kernel_size))
                    end_points['AvgPool_1a'] = net
                if not num_classes:
                    return net, end_points
                # 1 x 1 x 2048
                net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
                end_points['PreLogits'] = net
                # 2048
                logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                     normalizer_fn=None, scope='Conv2d_1c_1x1')
                if spatial_squeeze:
                    logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
                    # 1000
            end_points['Logits'] = logits
            end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
        return logits, end_points

darknet.default_image_size = 224


def darknet_arg_scope(weight_decay=0.0):
  """Defines the default lenet argument scope.

  Args:
    weight_decay: The weight decay to use for regularizing the model.

  Returns:
    An `arg_scope` to use for the inception v3 model.
  """
  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected], \
      weights_regularizer=slim.l2_regularizer(weight_decay), \
      weights_initializer=tf.truncated_normal_initializer(stddev=0.1), \
      activation_fn=tf.nn.relu) as sc:
    return sc