import tensorflow as tf
import numpy as np
import cv2

class Detector(object):
    def __init__(self, mode_path, input_size, batch_size):
        self.input_size = input_size
        self.batch_size = batch_size
        self.detection_graph = tf.Graph()
        self.session = tf.Session(graph=self.detection_graph)

        with self.detection_graph.as_default():
            self.input_pholder = tf.placeholder(tf.float32, [self.batch_size, input_size, input_size, 3])

            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(mode_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            with self.detection_graph.as_default():
                self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')


    def detect(self, image_batches):
        '''
        note that returned box cooridnates are normalized between [0, 1]
        :param image_batches: [batch_size, input_size, input_size, 3]
        :return: boxes=[t, l, b, r], scores=[conf]
        '''
        boxes, scores = self.session.run([self.detection_boxes, self.detection_scores],
                                 feed_dict={self.image_tensor: image_batches})

        return np.squeeze(boxes), np.squeeze(scores)


if __name__=='__main__':
    MODEL_PATH = '/Users/gglee/Data/TFModels/ssd_mobilenet_v2_quantized_160_v5/freeze/frozen_inference_graph.pb'
    detector = Detector(MODEL_PATH, 160, 1)
    image = cv2.imread('/Users/gglee/Data/Landmark/300W/01_Indoor/indoor_001.png')

    resized = cv2.cvtColor(cv2.resize(image, (160, 160)), cv2.COLOR_BGR2RGB)
    extended = np.expand_dims(resized, axis=0)

    boxes, scores = detector.detect(extended)

    H, W, _ = image.shape
    for i, b in enumerate(boxes):
        if scores[i] > 0.5:
            cv2.rectangle(image, (int(b[1]*W), int(b[0]*H)), (int(W*b[3]), int(H*b[2])), (0, 255, 0), 2)

    print(boxes, scores)
    cv2.imshow('face', image)
    cv2.waitKey(-1)