'''Test code to check the detection performance of a SSD face detector.
A face patch image is scaled-up and slided over the whole image frame to see if \
the detector misses faces at certain scales or locations.
'''

import os
import cv2
import numpy as np
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('face_checkpoint_dir', '/Users/gglee/Data/TFModels/128/ssd_face_128_v8/freeze',
                    'Directory containing checkpoints to evaluate, typically '
                    'set to `train_dir` used in the training job.')

flags.DEFINE_string('label_map_path', './face_label.pbtxt',
                    'File path of the label_map file. It can be omitted for one class detection (face)')

SAVE = True
SAVE_WIDTH = 640
SAVE_HEIGHT = 480

FLAGS = flags.FLAGS

def check_overlap(bb1, bb2):
    if bb1[0] > bb2[2] or bb1[2] < bb2[0]:
        return 0.0
    if bb1[1] > bb2[3] or bb1[3] < bb2[1]:
        return 0.0

    a1 = (bb1[2] - bb1[0] + 1) * (bb1[3] - bb1[1] + 1)
    a2 = (bb2[2] - bb2[0] + 1) * (bb2[3] - bb2[1] + 1)

    # print(bb1, bb2, a1, a2)
    ai = (min(bb1[2], bb2[2]) - max(bb1[0], bb2[0]) + 1) * (min(bb1[3], bb2[3]) - max(bb1[1], bb2[1]) + 1)

    overlap = ai / float(a1 + a2 - ai)
    # print(overlap)
    return overlap


if __name__=='__main__':
    MODEL_PATH = FLAGS.face_checkpoint_dir
    FACE_CKPT_PATH = os.path.join(MODEL_PATH, 'frozen_inference_graph.pb')
    LABEL_MAP_PATH = FLAGS.label_map_path

    video_writer = cv2.VideoWriter("/Users/gglee/Data/verify_face_ssd.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                   30.0, (SAVE_WIDTH, SAVE_HEIGHT)) if SAVE else None

    # import detection graph
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(FACE_CKPT_PATH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    BG_PATH = '/Users/gglee/Pictures/background.jpg'
    FACE_PATH = '/Users/gglee/Pictures/face.png'

    image_face = cv2.imread(FACE_PATH)
    image_bg = cv2.imread(BG_PATH)
    image_bg = cv2.resize(image_bg, (1280, 960))

    min_scale = 0.2
    max_scale = 2.8

    scales = []
    while min_scale < max_scale:
        scales.append(min_scale)
        min_scale *= 1.2
    # scales = [0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]

    FRAME_HEIGHT, FRAME_WIDTH = image_bg.shape[0:2]
    print('input_size: %d x %d' % (FRAME_WIDTH, FRAME_HEIGHT))

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:

            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            with open(os.path.join(os.path.dirname(MODEL_PATH), 'face_eval.txt'), 'w') as wf:
                for s in scales:
                    h, w = image_face.shape[0:2]
                    face = cv2.resize(image_face, (int(w*s), int(h*s)))

                    PATCH_HEIGHT, PATCH_WIDTH = face.shape[0:2]
                    STEP_X, STEP_Y = max(15, PATCH_WIDTH/10), max(15, PATCH_HEIGHT/10)

                    offset_y = 0

                    wf.write('scale: %.2f (%d x %d) \n' % (s, PATCH_WIDTH, PATCH_HEIGHT))

                    dw, dh = [], []

                    while offset_y + PATCH_HEIGHT < FRAME_HEIGHT + PATCH_HEIGHT/2:
                        offset_x = 0

                        while offset_x + PATCH_WIDTH < FRAME_WIDTH + PATCH_WIDTH/2:
                            bg_copy = image_bg.copy()
                            copy_width = min(PATCH_WIDTH, FRAME_WIDTH - offset_x - 1)
                            copy_height = min(PATCH_HEIGHT, FRAME_HEIGHT - offset_y - 1)
                            bg_copy[offset_y:offset_y+copy_height, offset_x:offset_x+copy_width] = face[:copy_height, :copy_width]

                            image_np = cv2.cvtColor(bg_copy, cv2.COLOR_RGB2BGR)
                            image_np_expanded = np.expand_dims(image_np, axis=0)

                            (boxes, scores, classes, num) = sess.run(
                                [detection_boxes, detection_scores, detection_classes, num_detections],
                                feed_dict={image_tensor: image_np_expanded})

                            boxes = np.squeeze(boxes)
                            scores = np.squeeze(scores)
                            classes = np.squeeze(classes).astype(np.int32)

                            hit = False

                            cv2.rectangle(bg_copy, (offset_x, offset_y),
                                          (offset_x + PATCH_WIDTH, offset_y + PATCH_HEIGHT), (255, 0, 0), 2)

                            for i, box in enumerate(boxes):
                                if scores[i] > 0.1:
                                    l, t, r, b, = int(box[1]*FRAME_WIDTH), int(box[0]*FRAME_HEIGHT), int(box[3]*FRAME_WIDTH), int(box[2]*FRAME_HEIGHT)
                                    hit = True if check_overlap([l, t, r, b], [offset_x, offset_y, offset_x + PATCH_WIDTH,
                                                                       offset_y + PATCH_HEIGHT]) > 0.0 else False

                                    if hit:
                                        cv2.rectangle(bg_copy, (l, t), (r, b), (0, 255, 0), 2)
                                        dw.append(r-l)
                                        dh.append(b-t)
                                        break
                                    else:
                                        cv2.rectangle(bg_copy, (l, t), (r, b), (0, 0, 255), 2)

                            cv2.imshow('image', bg_copy)
                            offset_x += STEP_X

                            wf.write('%d ' % (1 if hit else 0))

                            if SAVE:
                                video_writer.write(cv2.resize(bg_copy, (SAVE_WIDTH, SAVE_HEIGHT)))

                            cv2.waitKey(1)

                        wf.write('\n')
                        offset_y += STEP_Y

                    wf.write('\n')
                    try:
                        wf.write('%2.f, %2.f\n' % (sum(dw) / float(len(dw)), sum(dh) / float(len(dh))))
                    except ZeroDivisionError:
                        wf.write('0.0, 0.0\n')

    if video_writer:
        video_writer.release()