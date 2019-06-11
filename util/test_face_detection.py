'''
run face detection on eval sets to see when it fails
'''

import os
import cv2
import tensorflow as tf
import numpy as np
import random
from data.convert_voc_xml_to_wider_format import parse_xml
from data.refine_widerface import get_iou

flags = tf.app.flags
flags.DEFINE_string('images_dir', None,
                    'Directory containing sub-dirs of images')

flags.DEFINE_string('model', None, 'trained face detector')
flags.DEFINE_string('out', None, 'path of the text file to write the results')

FLAGS = flags.FLAGS

COLOR_MATCHED_DET = (55, 55, 255)
COLOR_UNMATCHED_DET = (0, 0, 130)
COLOR_MATCHED_GT = (55, 255, 55)
COLOR_UNMATCHED_GT = (0, 130, 0)

IOU_TH = 0.003
CONF_TH = 0.5

def draw_readable_text(image, text, pos, size):
    cv2.putText(image, text, (pos[0]-1, pos[1]), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0))
    cv2.putText(image, text, (pos[0]+1, pos[1]), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0))
    cv2.putText(image, text, (pos[0], pos[1]-1), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0))
    cv2.putText(image, text, (pos[0], pos[1]+1), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0))
    cv2.putText(image, text, (pos[0], pos[1]), cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255))


def draw_grids(image, num_cell):
    '''
    num_cell: (h, w)
    '''
    H, W = image.shape[0:2]

    xs = [int(float(W) * i / num_cell[1]) for i in range(num_cell[1])]
    ys = [int(float(H) * i / num_cell[0]) for i in range(num_cell[0])]

    for x in xs:
        cv2.line(image, (x, 0), (x, H), (0, 0, 0), 2)

    for y in ys:
        cv2.line(image, (0, y), (W, y), (0, 0, 0))

    return image


def find_best_matching_gtbox(detection, answers):
    '''

    :param detection:
    :param answers:
    :return:
    '''
    ious = map(lambda a: get_iou(detection, a), answers)
    max_index = np.argmax(np.asarray(ious))
    # print(ious, max_index)
    return ious[max_index], max_index


if __name__=='__main__':
    if not FLAGS.images_dir or not FLAGS.model:
        print('check images_dir or model')
        exit()

    IMAGES_DIR = FLAGS.images_dir
    MODEL_PATH = os.path.join(FLAGS.model, 'frozen_inference_graph.pb')
    OUT_PATH = FLAGS.out

    wf = open(OUT_PATH, 'w') if OUT_PATH else None
    wf.write('class, iou, cx, cy, w, h\n')

    if not os.path.exists(IMAGES_DIR) or not os.path.isdir(IMAGES_DIR):
        exit()

    images = []
    for root, dirs, files in os.walk(IMAGES_DIR):
        for f in files:
            if f.split('.')[-1].lower() in ['jpg', 'png']:
                images.append(os.path.join(root, f))

    print('%d images found' % len(images))

    random.shuffle(images)

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(MODEL_PATH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    if not detection_graph:
        print('Failed to load detection graph: %s' % MODEL_PATH)
        exit()


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

            for ip in images:
                image = cv2.imread(ip)

                if image is None:
                    print('Failed to read image: %s' % ip)
                    continue

                print(ip)
                H, W = image.shape[0:2]

                WIDTH = 480
                I2D = float(WIDTH)/W
                HEIGHT = int(I2D*H)
                image_draw = cv2.resize(image, (WIDTH, HEIGHT))

                xmp_path = os.path.splitext(ip)[0] + '.xml'
                gts = parse_xml(xmp_path)

                image_np = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                boxes = np.squeeze(boxes)
                scores = np.squeeze(scores)
                classes = np.squeeze(classes).astype(np.int32)

                # draw_grids(image_draw, (128/16, 128/16))

                HEIGHT, WIDTH, _ = image.shape
                crop_boxes = []

                out_class = ''


                dets = []
                for i, box in enumerate(boxes):
                    if scores[i] < CONF_TH:
                        continue
                    dets.append({'l': box[1]*W, 't': box[0]*H, 'r': box[3]*W, 'b': box[2]*H, 'c': scores[i]})

                ans = []
                for g in gts:
                    ans.append({'l': g['xmin'], 't': g['ymin'], 'r': g['xmax'], 'b': g['ymax']})

                def write_result(f, cls, iou, bb, size):
                    cx = (bb['l'] + bb['r']) / (2.0 * size[0])
                    cy = (bb['t'] + bb['b']) / (2.0 * size[1])
                    w = (bb['r'] - bb['l']) / float(size[0])
                    h = (bb['b'] - bb['t']) / float(size[1])

                    f.write('%s, %.3f, %.3f, %.3f, %.3f, %.3f\n' % (cls, iou, cx, cy, w, h))

                for d in dets:
                    iou, idx = 0.0, -1
                    matched = False

                    if len(ans) > 0:
                        iou, idx = find_best_matching_gtbox(d, ans)

                    if iou > IOU_TH:
                        COLOR_DET = COLOR_MATCHED_DET
                        matched = True
                    else:
                        COLOR_DET = COLOR_UNMATCHED_DET

                    cv2.rectangle(image_draw, (int(I2D * d['l']), int(I2D * d['t'])),
                                  (int(I2D * d['r']), int(I2D * d['b'])), COLOR_DET, 2)

                    if matched:
                        match = ans[idx]
                        cv2.rectangle(image_draw, (int(I2D*match['l']), int(I2D*match['t'])),
                                      (int(I2D*match['r']), int(I2D*match['b'])), COLOR_MATCHED_GT, 2)
                        draw_readable_text(image_draw, '%.2f' % iou, (int(I2D*(match['l']*3+match['r'])/4.0),
                                                                      int(I2D*(match['t']+match['b'])/2.0)), 0.5)
                        ans.pop(idx)
                        write_result(wf, 'match', iou, match, (WIDTH, HEIGHT))
                    else:
                        write_result(wf, 'fa', -1.0, d, (WIDTH, HEIGHT))

                for a in ans:
                    cv2.rectangle(image_draw, (int(I2D * a['l']), int(I2D * a['t'])),
                                  (int(I2D * a['r']), int(I2D * a['b'])), COLOR_UNMATCHED_GT, 2)
                    write_result(wf, 'miss', -1.0, a, (WIDTH, HEIGHT))

                cv2.imshow("image", image_draw)
                key = cv2.waitKey(10)
                if key == ord('q'):
                    if wf:
                        wf.close()
                    break

    wf.close()
    print('Done!')