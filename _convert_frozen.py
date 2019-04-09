import tensorflow as tf
import os

from tensorflow.python.platform import gfile
dir = '/Users/gglee/Data/Landmark/train/bests/0408_gpu1_x101'
name = 'frozen_2'
model_path="/tmp/frozen/dcgan.pb"

# read graph definition
f = gfile.FastGFile(os.path.join(dir, name), "rb")
gd = graph_def = tf.GraphDef()
graph_def.ParseFromString(f.read())

# fix nodes
for node in graph_def.node:
    if node.op == 'RefSwitch':
        node.op = 'Switch'
        for index in xrange(len(node.input)):
            if 'moving_' in node.input[index]:
                node.input[index] = node.input[index] + '/read'
    elif node.op == 'AssignSub':
        node.op = 'Sub'
        if 'use_locking' in node.attr: del node.attr['use_locking']

# import graph into session
tf.import_graph_def(graph_def, name='')
tf.train.write_graph(graph_def, dir, 'good_frozen_2.pb', as_text=False)
tf.train.write_graph(graph_def, dir, 'good_frozen_2.pbtxt', as_text=True)