import numpy as np
import os
import sys
import tensorflow as tf
from time import time
import cv2
from shutil import rmtree

flags = tf.app.flags
flags.DEFINE_string('checkpoint_dir', '',
                    'Directory containing checkpoints to evaluate, typically '
                    'set to `train_dir` used in the training job.')

flags.DEFINE_string('label_map_path', '',
                    'Directory containing checkpoints to evaluate, typically '
                    'set to `train_dir` used in the training job.')

flags.DEFINE_string('video_path', '',
                    'Video file path. if both video_path ans images_dirs are empty, camera is used for input.')

flags.DEFINE_string('images_dir', '',
                    'Directory containing images for evaluation. For WiderFaceDB,'
                    'has_child_dirs should be set to True. If empty, camera stream is used for input')

flags.DEFINE_string('write_dir', '',
                    'Directory to write detection results. For WiderFaceDB,'
                    'has_child_dirs should be set to True')

flags.DEFINE_bool('has_child_dirs', 'True', 'If true, images_dir is considered as'
                    'a root folder which contains sub-directories of images')

flags.DEFINE_bool('save', 'True',
                  'If true, save resulting images or video with detection boxes.')

flags.DEFINE_bool('disp', 'False',
                  'If true, show detection images (only for file inpu)')

FLAGS = flags.FLAGS

def prepare_filelist(val_path, is_root):
    images_to_test = []

    if is_root:
        for folder in os.listdir(val_path):
            if os.path.isdir(os.path.join(val_path, folder)):
                sub = os.path.join(val_path, folder)
                for name in os.listdir(sub):
                    filepath = os.path.join(sub, name)
                    if filepath.endswith('jpg'):
                        base, ext = name.split('.')
                        cur = {'folder': folder, 'basename': base, 'ext': ext}
                        images_to_test.append(cur)
    else:
        for f in os.listdir(val_path):
            filepath = os.path.join(val_path, f)
            if os.path.isfile(filepath):
                if filepath.endswith('jpg') or filepath.endswith('png'):
                    base, ext = f.split('.')
                    cur = {'folder': val_path, 'basename': base, 'ext': ext}
                    images_to_test.append(cur)


    return images_to_test


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def write_detection(write_dir, entry, boxes, scores, th_score, is_root=True):
    if is_root:
        write_subdir = os.path.join(write_dir, entry['folder'])
    else:
        write_subdir = write_dir

    if os.path.exists(write_subdir) is False:
        os.makedirs(write_subdir)

    with open(os.path.join(write_subdir, entry['basename'] + '.txt'), 'w') as write_file:
        write_file.write(entry['basename'] + '\n')

        str_to_write = []

        HEIGHT, WIDTH = image_np.shape[0:2]

        for i in range(boxes.shape[0]):
            ymin, xmin, ymax, xmax = boxes[i]
            x, y = xmin * WIDTH, ymin * HEIGHT
            h, w = (ymax - ymin) * HEIGHT, (xmax - xmin) * WIDTH

            if scores[i] > th_score:
                str_to_write.append(('%f %f %f %f %f\n' % (x, y, w, h, scores[i])))

        write_file.write('%d\n' % len(str_to_write))
        for s in str_to_write:
            write_file.write(s)

if __name__ == '__main__':

    MODEL_NAME = FLAGS.checkpoint_dir
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    LABEL_MAP_PATH = FLAGS.label_map_path
    IMAGES_DIR = FLAGS.images_dir
    WRITE_DET_DIR = FLAGS.write_dir
    VIDEO_PATH = FLAGS.video_path

    video_writer = None

    if IMAGES_DIR == '':
        use_camera = True

        if VIDEO_PATH == '':
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(VIDEO_PATH)
            if cap is None:
                print('cannot open video: %s'%VIDEO_PATH)
                exit()

        if FLAGS.save and VIDEO_PATH != '':
            if FLAGS.write_dir == '':
                write_dir = os.path.dirname(VIDEO_PATH)
            else:
                write_dir = FLAGS.write_dir

            basename = os.path.splitext(os.path.dirname(VIDEO_PATH))[0]
            vid_write_path = os.path.join(write_dir, basename+'_result.avi')
            video_writer = cv2.VideoWriter(vid_write_path, cv2.cv.FOURCC('Y', 'U', 'V', '2'), 15.0, (624, 352))
    else:
        use_camera = False
        IS_ROOT = FLAGS.has_child_dirs
        images_to_test = prepare_filelist(IMAGES_DIR, IS_ROOT)

    if WRITE_DET_DIR != '':
        if os.path.exists(WRITE_DET_DIR):
            rmtree(WRITE_DET_DIR)
        os.makedirs(WRITE_DET_DIR)

    # List of the strings that is used to add correct label for each box.
    # PATH_TO_LABELS = os.path.join('data', 'wider_label_map.pbtxt')
    NUM_CLASSES = 1

    # import detection graph
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    start_time = time()
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

            more = True
            pos = 0

            while more:
                if use_camera:
                    more, image = cap.read()
                else:
                    entry = images_to_test[pos]
                    # for image_path in TEST_IMAGE_PATHS:
                    image_path = os.path.join(os.path.join(IMAGES_DIR, entry['folder']), entry['basename'] + '.' + entry['ext'])
                    print(image_path)

                    pos += 1
                    if pos == len(images_to_test):
                        more = False

                    # the array based representation of the image will be used later in order to prepare the
                    # result image with boxes and labels on it.
                    #image_np = load_image_into_numpy_array(image)
                    image = cv2.imread(image_path)

                    if image is None:
                        print('image not exist: %s'%image_path)
                        continue

                image_np = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)    # need this? check performance

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # if WRITE_DET_DIR != '':
                #     write_detection(WRITE_DET_DIR, entry, np.squeeze(boxes), np.squeeze(scores), 0.005, IS_ROOT)

                # boxes = np.reshape(boxes, (len(boxes)/4, 4))

                boxes = np.squeeze(boxes)
                scores = np.squeeze(scores)
                classes = np.squeeze(classes).astype(np.int32)

                for i, b in enumerate(boxes):
                    if scores[i] < 0.5:
                        break;
                    H, W, _ = image.shape
                    cv2.rectangle(image, (int(b[1]*W), int(b[0]*H)), (int(b[3]*W), int(b[2]*H)), (0, 0, 255), 2)

                cv2.imshow("image", image)
                key = cv2.waitKey(20)
                if key == 113 or key == 120:
                    break

                # if FLAGS.save or FLAGS.disp or use_camera:
                #     # Visualization of the results of a detection.
                #     vis_util.visualize_boxes_and_labels_on_image_array(
                #         image,
                #         np.squeeze(boxes),
                #         np.squeeze(classes).astype(np.int32),
                #         np.squeeze(scores),
                #         category_index,
                #         use_normalized_coordinates=True,
                #         line_thickness=2,
                #         min_score_thresh=0.5)

                    # if FLAGS.save:
                    #     if not use_camera:
                    #         image_to_save = image
                    #         path2save = os.path.join(WRITE_DET_DIR, entry['folder'])
                    #         path2save = os.path.join(path2save, entry['basename'] + '.' + entry['ext'])
                    #         cv2.imwrite(path2save, image_to_save)
                    #     else:
                    #         if video_writer:
                    #             image_to_save = cv2.resize(image, (624, 352))
                    #             video_writer.write(image_to_save)

                    # if FLAGS.disp or use_camera:
                    #     cv2.imshow("image", image)
                    #     keyIn = cv2.waitKey(1)
                    #     if keyIn == ord('c') or keyIn == ord('x') or keyIn == ord('q'):
                    #         more = False
                    #         if video_writer:
                    #             video_writer.Release()
    end_time = time()

    print('===== task finished: %s seconds ======', end_time-start_time)
#python detect_faces.py --checkpoint_dir=/Volumes/Data/Trained/WiderFaceSSD/freeze/300_small_valid/output_inference_graph.pb --label_map_path=/Users/gglee/Develop/FaceDetectionEmbedded/utils/data/face_detection_label_map.pbtxt --images_dir=/Volumes/Data/FaceDetectionDB/WiderFace/WIDER_val/images/ --write_dir=/Volumes/Data/FaceDetectionDB/WiderFace/test_wider_val --save_image=True --disp=True

#python detect_faces.py --checkpoint_dir=/Users/gglee/Develop/emdfd/utils/train/ssd_darknet/ --label_map_path=/Users/gglee/Develop/emdfd/utils/data/face_detection_label_map.pbtxt

# python detect_face.py --checkpoint_dir=/Users/gglee/Data/1207_face/