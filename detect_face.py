import numpy as np
import os
import sys
import tensorflow as tf
from time import time
import cv2
from shutil import rmtree
import inference as infer

'''
Usage:

(1) to detect from camera:
python detect_faces.py \
        --face_checkpoint_dir=/must/specified [--landmark_checkpoint_dir=/can/be/omitted] \ 
        [--write_dir_name=folder_name_to_save] [--dist=True]
        
(2) to detect from a local video file:
python detect_faces.py --video_path='/video/file/to/run/detection.avi' \
        --face_checkpoint_dir=/must/specified [--landmark_checkpoint_dir=/can/be/omitted] \ 
        [--write_dir_name=folder_name_to_save] [--dist=True]
        
(3) to detect from images in a folder:
python detect_faces.py --images_dir='/folder/name/to/load/images' \
        --face_checkpoint_dir=/must/specified [--landmark_checkpoint_dir=/can/be/omitted] \ 
        [--write_dir_name=folder_name_to_save] [--dist=True]
        
(4) to detect from series of folders containing images:
python detect_faces.py --folder_list='/some/file/containing/folder/names.txt' \
        --face_checkpoint_dir=/must/specified [--landmark_checkpoint_dir=/can/be/omitted] \ 
        [--write_dir_name=folder_name_to_save] [--dist=True]
'''

flags = tf.app.flags
flags.DEFINE_string('face_checkpoint_dir', '',
                    'Directory containing checkpoints to evaluate, typically '
                    'set to `train_dir` used in the training job.')

flags.DEFINE_string('landmark_checkpoint_path', '',
                    'Directory containing checkpoints to evaluate, typically '
                    'set to `train_dir` used in the training job.')

flags.DEFINE_string('label_map_path', './face_label.pbtxt',
                    'File path of the label_map file. It can be omitted for one class detection (face)')

flags.DEFINE_string('video_path', '',
                    'Video file path. if both video_path ans images_dirs are empty, camera is used for input.')

flags.DEFINE_string('images_dir', '',
                    'Directory containing images for evaluation. If empty, camera stream is used for input')

flags.DEFINE_string('folder_list', '',
                    'Text file that contains directories to scaen')

flags.DEFINE_string('write_dir_name', '',
                    'If specified, a folder will be created by the given name and detection result will be saved as .txt'
                    'Only work for image sources, not video')

flags.DEFINE_bool('save_images', 'True',
                  'If true, save resulting images or video with detection boxes.')

flags.DEFINE_bool('disp', 'True',
                  'If true, show detection images (only for file input)')

FLAGS = flags.FLAGS


def prepare_filelist(folder_path):
    '''
    given list of folder names, return list of fliepaths for images to evaluate
    :param folder_paths:
    :return:
    '''
    assert os.path.isdir(folder_path)
    assert os.path.exists(folder_path)

    images_to_test = []

    for f in os.listdir(folder_path):
        filepath = os.path.join(folder_path, f)

        if os.path.isfile(filepath):
            if filepath.endswith('jpg') or filepath.endswith('png'):
                base, ext = f.split('.')
                cur = {'folder': folder_path, 'basename': base, 'ext': ext}
                images_to_test.append(cur)

    return images_to_test


def write_detection(write_dir, entry, boxes, scores, th_score):

    assert os.path.exists(write_dir), 'Folder not exist: %s' % write_dir

    with open(os.path.join(write_dir, entry['basename'] + '.txt'), 'w') as write_file:
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

    # mode related
    MODEL_NAME = FLAGS.face_checkpoint_dir
    FACE_CKPT_PATH = os.path.join(MODEL_NAME, 'frozen_inference_graph.pb')
    LANDMARK_CKPT_PATH = FLAGS.landmark_checkpoint_path
    LABEL_MAP_PATH = FLAGS.label_map_path

    # source related
    IMAGES_DIR = FLAGS.images_dir
    VIDEO_PATH = FLAGS.video_path
    FOLDER_LISTS_TXT = FLAGS.folder_list

    # actions rlated
    WRITE_DET_DIR = FLAGS.write_dir_name
    DISP = FLAGS.disp

    video_writer = None
    cap = None
    use_camera = False
    landmark_estimator = None

    if LANDMARK_CKPT_PATH != '':
        assert os.path.exists(LANDMARK_CKPT_PATH), 'Landmark checkpoint not exist: %s' % LANDMARK_CKPT_PATH
        landmark_estimator = infer.Classifier(48, LANDMARK_CKPT_PATH)

    # set sources
    image_to_test = []
    if VIDEO_PATH:
        cap = cv2.VideoCapture(VIDEO_PATH)
        if cap is None:
            print('cannot open video: %s'%VIDEO_PATH)
            exit()
    elif IMAGES_DIR:
        image_to_test = prepare_filelist(IMAGES_DIR)
    elif FOLDER_LISTS_TXT:
        assert os.path.exists(FOLDER_LISTS_TXT), 'Folder list file not exist: %s' % FOLDER_LISTS_TXT
        with open(FOLDER_LISTS_TXT) as rf:
            for l in rf.readlines():
                l = l.strip()
                print('  > Reading file list from: %s' % l)
                assert os.path.exists(l) and os.path.isdir(l), 'Folder exist: %d or dir: %d, %s' % (os.path.exists(l), os.path.isdir(l), l)
                image_to_test += prepare_filelist(l)
    else:
        use_camera = True
        cap = cv2.VideoCapture(0)

    # if FLAGS.save_images:
    #     if FLAGS.write_dir == '':
    #         write_dir = os.path.dirname(VIDEO_PATH)
    #     else:
    #         write_dir = FLAGS.write_dir
    #
    #     basename = os.path.splitext(os.path.dirname(VIDEO_PATH))[0]
    #     vid_write_path = os.path.join(write_dir, basename+'_result.avi')
    #     video_writer = cv2.VideoWriter(vid_write_path, cv2.cv.FOURCC('Y', 'U', 'V', '2'), 15.0, (624, 352))
    # else:
    #     use_camera = False
    #     IS_ROOT = FLAGS.has_child_dirs
    #     images_to_test = prepare_filelist(IMAGES_DIR, IS_ROOT)

    # if WRITE_DET_DIR:
    #     if os.path.exists(WRITE_DET_DIR):
    #         rmtree(WRITE_DET_DIR)
    #     os.makedirs(WRITE_DET_DIR)

    # List of the strings that is used to add correct label for each box.
    # PATH_TO_LABELS = os.path.join('data', 'wider_label_map.pbtxt')
    NUM_CLASSES = 1

    # import detection graph
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(FACE_CKPT_PATH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    start_time = time()
    counter = 0

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
            image = None
            entry = None

            while more:
                if cap:
                    more, image = cap.read()
                else:
                    entry = image_to_test[pos]
                    # for image_path in TEST_IMAGE_PATHS:
                    image_path = os.path.join(os.path.join(IMAGES_DIR, entry['folder']), entry['basename'] + '.' + entry['ext'])
                    print('[%d / %d] %s' % (pos, len(image_to_test), image_path))

                    pos += 1
                    if pos == len(image_to_test):
                        more = False

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

                if WRITE_DET_DIR and not cap:
                    # tdoo: create folder
                    save_dir = os.path.join(entry['folder'], WRITE_DET_DIR)

                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir, 0755)

                    write_detection(save_dir, entry, np.squeeze(boxes), np.squeeze(scores), 0.005)

                # boxes = np.reshape(boxes, (len(boxes)/4, 4))

                patches = np.zeros((8, 48, 48, 3), dtype=np.float32)

                boxes = np.squeeze(boxes)
                scores = np.squeeze(scores)
                classes = np.squeeze(classes).astype(np.int32)

                HEIGHT, WIDTH, _ = image.shape
                scale = 0.5

                crop_boxes = []
                SCORE_TH = 0.6

                for i, box in enumerate(boxes):
                    if scores[i] < SCORE_TH:
                        continue

                    # assert box[1] < box[3] and box[0] < box[2], "%f > %f, %f > %f" % (box[1], box[3], box[0], box[1])

                    # classify faces
                    l, t, r, b = int(box[1] * WIDTH), int(box[0] * HEIGHT), int(box[3] * WIDTH), int(box[2] * HEIGHT)
                    w, h = r - l, b - t

                    if w == 0 or h == 0:
                        print(box, '[%d, %d, %d, %d] w/ %.2f' %(l, t, r, b, scores[i]))
                        continue

                    cx, cy = (l + r) / 2.0, (b + t) / 2.0

                    w, h = w * scale, h * scale
                    l, r = max(0, int(cx - w)), min(int(cx + w), WIDTH)
                    t, b = max(0, int(cy - h)), min(int(cy + h), HEIGHT)

                    if l == r:
                        r = l + 1
                    if t == b:
                        b = t + 1
                    t = b - (r - l)

                    crop_boxes.append([l, t, r, b])

                for i, box in enumerate(crop_boxes):
                    # print(image.shape, box)
                    face = image[box[1]:box[3], box[0]:box[2], :]

                    # face = image[min(box[1], box[3]):max(box[1], box[3]), min(box[0], box[2]):max(box[0], box[2])]
                    # print(face.shape)

                    if landmark_estimator:
                        if i < 8:
                            patch = cv2.resize(face, (48, 48))
                            patches[i, :, :, :] = ((np.asarray(patch).astype(np.float32))/255.0-1.0)
                            # verify = ((np.asarray(patches[i, :, :, 0]).squeeze()+1.0)*255.0).astype(np.uint8)
                            # cv2.imshow("verify", verify)

                        landmarks = np.reshape(np.squeeze(landmark_estimator.predict(patches)), (-1, 68, 2))

                for i, box in enumerate(boxes):
                    if scores[i] < SCORE_TH:
                        break

                    cv2.rectangle(image, (int(box[1] * WIDTH), int(box[0] * HEIGHT)),
                                  (int(box[3] * WIDTH), int(box[2] * HEIGHT)), (0, 0, 255), 2)

                    if landmark_estimator:
                        patch = ((np.asarray(patches[i, :, :, 0]).squeeze() + 1.0) * 255.0).astype(np.uint8)

                        for p in landmarks[i]:
                            cv2.circle(patch, (int(p[0]*48), int(p[1]*48)), 1, (255, 255, 255))
                        cv2.imshow("patch", patch)

                        cv2.rectangle(image, (int(crop_boxes[i][0]), int(crop_boxes[i][1])),
                                      (int(crop_boxes[i][2]), int(crop_boxes[i][3])), (0, 255, 0), 2)

                        H, W = (box[2]-box[0])*HEIGHT, (box[3]-box[1])*WIDTH
                        for p in landmarks[i]:
                            cv2.circle(image, (int(box[1]*WIDTH+(p[0]*W)), int(box[0]*HEIGHT+(H-W)+p[1]*W)), 2, (0, 255, 255))

                cv2.imshow("image", image)

                key = cv2.waitKey(1)
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
# python detect_face.py --checkpoint_dir=/Users/gglee/Data/TFModels/ssd_wider_mn2_0.5_192_ar1