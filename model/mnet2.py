from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim

import sys
sys.path.append('/Users/gglee/Develop/models/research/nets')

from mobilenet import mobilenet_v2

def mnet2(input_tensor):
    mobilenet_v2.mobilenet(input_tensor, )