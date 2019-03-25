import tensorflow as tf
import numpy as np

import sys
sys.path.append('..')

import net

slim = tf.contrib.slim

class Classifier(object):
    def __init__(self, input_size, model_path):
        self.input_size = input_size

        with slim.arg_scope(net.arg_scope(weight_decay=0.0005)):
            with tf.Graph().as_default():
                self.sess = tf.Session()
                # Build a Graph that computes the logits predictions from the
                # inference model.
                self.images_pholder = tf.placeholder(tf.float32, [8, input_size, input_size, 3])

                with tf.variable_scope('model') as scope:
                    self.landmarks, _ = net.lannet(self.images_pholder, is_training=False)

                saver = tf.train.Saver()
                saver.restore(self.sess, model_path)

    def predict(self, in_batch):
        # output = np.zeros((len(in_batch), 2), dtype=np.float32)      # fix me: output gets only one result.
        # for i in xrange(len(in_batch)):
        #     predict = self.sess.run([self.cls_prob], feed_dict={self.images_pholder: in_batch[i:i + 1]})
        #     output[i] = np.reshape(predict, (2))
        #     print(output)
        #
        # return output
        pred_out = self.sess.run([self.landmarks], feed_dict={self.images_pholder: in_batch})
        # print(pred_out, self.landmarks)
        # return self.landmarks
        return np.asarray(pred_out)


if __name__ == '__main__':
    PATH = '/Users/gglee/Data/300W/export'
    landmark_estimator = Classifier(48, '/Users/gglee/Data/300W/model/model')

    in_tensor = np.zeros((8, 48, 48, 1), dtype=np.float32)

    import os
    import cv2

    for f in os.listdir(PATH):
        if not f.startswith('.') and f.endswith('img'):
            data = np.fromfile(os.path.join(PATH, f))
            in_tensor[0, :, :, 0] = ((np.asarray(data).reshape((48, 48)).astype(np.float32))/255.0-1.0)

            landmarks = np.reshape(np.squeeze(landmark_estimator.predict(in_tensor)), (-1, 68, 2))

            image = ((np.asarray(in_tensor[0, :, :, 0]).squeeze() + 1.0) * 255.0).astype(np.uint8)

            for p in landmarks[0]:
                cv2.circle(image, (int(p[0] * 48), int(p[1] * 48)), 1, (255, 255, 255))

            cv2.imshow("patch", image)

            key = cv2.waitKey(-1)
            if key == 113 or key == 120:
                break