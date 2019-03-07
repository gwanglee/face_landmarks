import tensorflow as tf
import numpy as np
import os
from PIL import Image, ImageDraw
import psutil
import shutil

import net
from random import shuffle

CH = 3

slim = tf.contrib.slim

def prepare_data_list(inpath):
    assert os.path.exists(inpath) and os.path.isdir(inpath)

    lists = []
    for l in os.listdir(inpath):
        if l.endswith('img'):
            pts_path = os.path.join(inpath, os.path.splitext(l)[0] + '.pts')
            if os.path.exists(pts_path):
                lists.append([os.path.join(inpath, l), pts_path])

    assert len(lists) > 0

    return lists

def read_data(list2train, cur_pos, batch_size=10):
    data = np.zeros((batch_size, 48, 48, CH), dtype=np.float32)
    ans = np.zeros((batch_size, 68*2), dtype=np.float32)

    for i in range(batch_size):
        cur_pos += 1
        p = cur_pos % len(list2train)
        d = np.reshape(np.fromfile(list2train[p][0], dtype=float), (48, 48, CH))
        a = np.fromfile(list2train[p][1], dtype=float)
        data[i, :, :, :] = d * 2 / 255.0 - 1.0       # norm between -1 to 1
        ans[i, :] = a
    return data, ans, p


def train(list2train, max_epoch=16, batch_size=32, num_threads=4, save_path='./train/model.ckpt'):

    num_samples = len(list2train)

    with slim.arg_scope(net.arg_scope()):

        data_ph = tf.placeholder(tf.float32, [None, 48, 48, CH], name='input')
        ans_ph = tf.placeholder(tf.float32, [None, 68*2])

        estims, _ = net.lannet(data_ph, is_training=True)

        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.01

        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, len(list2train), #len(list2train)/batch_size*4,
                                                   0.998, staircase=True)
        # loss = tf.losses.mean_squared_error(ans_ph, estims, scope='mse')
        loss = tf.losses.huber_loss(ans_ph, estims, delta=0.01, scope='mse')
        # train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
        train_op = tf.train.AdamOptimizer().minimize(loss)
        ema = tf.train.ExponentialMovingAverage(decay=0.999)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        sess = tf.Session()
        sess.run(init_op)

        num_iter = 0
        epoch = 0
        pos = 0

        while epoch < max_epoch:

            input_batch, ans_batch, pos = read_data(list2train, pos, batch_size=batch_size)

            steps, lr, val_loss, ans_pred, _ = sess.run([global_step, learning_rate, loss, estims, train_op], feed_dict={data_ph: input_batch, ans_ph: ans_batch})
            num_iter += 1

            steps = num_iter * batch_size
            epoch = int(steps/num_samples)
            print('(pos %4d) Epoch %d, iter %d : loss=%f. lr=%f' % (pos, epoch, num_iter, val_loss, lr))

            # if epoch % int(max_epoch/48) == 0:
            #     patch = np.asarray((input_batch[0, :, :, :]+1.0)*255.0).astype('uint8').reshape((48, 48))
            #     img = Image.fromarray(patch)
            #     pts = np.asarray(ans_batch[0, :]).reshape((68, 2))
            #
            #     draw = ImageDraw.Draw(img)
            #     w, h = img.width, img.height
            #     for p in pts:
            #         l, t, r, b = int(p[0] * w) - 1, int(p[1] * h) - 1, int(p[0] * w) + 1, int(p[1] * h) + 1
            #         draw.ellipse((l, t, r, b))
            #
            #     del draw
            #
            #     for proc in psutil.process_iter():
            #         if proc.name() == 'display':
            #             proc.kill()
            #
            #     img.show()

        # write save code here
        if save_path:
            path_to_save = save_path
            if os.path.exists(path_to_save):
                shutil.rmtree(path_to_save)

            os.makedirs(path_to_save)

            saver = tf.train.Saver()
            save_path = saver.save(sess, os.path.join(path_to_save, 'trained.ckpt'))
            tf.train.write_graph(sess.graph.as_graph_def(), path_to_save, 'trained.pbtxt', as_text=True)
            tf.train.write_graph(sess.graph.as_graph_def(), path_to_save, 'trained.pb', as_text=False)
            print('model saved: %s'%save_path)

        sess.close()

if __name__ == '__main__':
    train_list = prepare_data_list('/Users/gglee/Data/Landmark/300W/export')
    train(train_list, max_epoch=29, save_path='/Users/gglee/Data/Landmark/300W/model')