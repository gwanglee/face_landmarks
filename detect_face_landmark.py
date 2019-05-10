import numpy as np
import os
import tensorflow as tf
from time import time
import cv2
import inference as infer
from copy import deepcopy
from operator import itemgetter
from evaluate_landmark_model import  load_settings


'''
Usage:

(1) to detect from camera:
python detect_face_landmark.py \
        --face_checkpoint_dir=/must/specified [--landmark_checkpoint_path=/can/be/omitted] \ 
        [--write_txt_dir=folder_name_to_save] [--dist=True]
        
(2) to detect from a local video file:
python detect_face_landmark.py --video_path='/video/file/to/run/detection.avi' \
        --face_checkpoint_dir=/must/specified [--landmark_checkpoint_path=/can/be/omitted] \ 
        [--write_txt_dir=folder_name_to_save] [--dist=True]
        
(3) to detect from images in a folder:
python detect_face_landmark.py --images_dir='/folder/name/to/load/images' \
        --face_checkpoint_dir=/must/specified [--landmark_checkpoint_path=/can/be/omitted] \ 
        [--write_txt_dir=folder_name_to_save] [--dist=True]
        
(4) to detect from series of folders containing images:
python detect_face_landmark.py --folder_list='/some/file/containing/folder/names.txt' \
        --face_checkpoint_dir=/must/specified [--landmark_checkpoint_path=/can/be/omitted] \ 
        [--write_txt_dir=folder_name_to_save] [--dist=True]
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
                    'Text file that contains directories to scan')

flags.DEFINE_string('write_txt_dir', None,
                    'If specified, a folder will be created by the given name and detection result will be saved as .txt'
                    'Only work for image sources, not video')

flags.DEFINE_bool('disp', 'True',
                  'If true, show detection images (only for file input)')

FLAGS = flags.FLAGS

LANDMARK_INPUT_SIZE = 56

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

    images_to_test = sorted(images_to_test, key=itemgetter('basename'))

    return images_to_test


def write_detection(write_dir, basename, face_landmarks):

    assert os.path.exists(write_dir), 'Folder not exist: %s' % write_dir

    with open(os.path.join(write_dir, basename + '.txt'), 'w') as wf:
        wf.write(entry['basename'] + '\n')
        wf.write('%d\n' % len(face_landmarks))

        for res in face_landmarks:
            wf.write('%d %d %d %d\n' % (res['face'][0], res['face'][1], res['face'][2], res['face'][3]))
            for i in range(68):
                wf.write('%.3f %.3f ' % (res['landmark'][i][0], res['landmark'][i][1]))
        wf.write('\n')


def read_detection(filepath):
    assert os.path.exists(filepath), 'file not exist: %s' % filepath
    detection = []

    with open(filepath) as rf:
        header = rf.readline()
        if header.strip() != os.path.splitext(os.path.basename(filepath))[0]:
            raise ValueError('Wrong detection file : %s ' % filepath)

        num = int(rf.readline().strip())

        for i in range(num):
            l, t, r, b = rf.readline().split()
            l, t, r, b = int(l.strip()), int(t.strip()), int(r.strip()), int(b.strip())

            landmark = []
            for p in rf.readline().split():
                landmark.append(float(p.strip()))

            detection.append({'face': [l, t, r, b], 'landmark': landmark})

    return detection


if __name__ == '__main__':

    # models
    MODEL_NAME = FLAGS.face_checkpoint_dir
    FACE_CKPT_PATH = os.path.join(MODEL_NAME, 'frozen_inference_graph.pb')
    LANDMARK_CKPT_PATH = FLAGS.landmark_checkpoint_path
    LABEL_MAP_PATH = FLAGS.label_map_path

    # source
    IMAGES_DIR = FLAGS.images_dir
    VIDEO_PATH = FLAGS.video_path
    FOLDER_LISTS_TXT = FLAGS.folder_list

    # save & display
    WRITE_TXT_DIR = FLAGS.write_txt_dir
    DISP = FLAGS.disp

    FACE_SCORE_THRESHOLD = 0.5

    video_writer = None
    cap = None
    landmark_estimator = None
    LIVE_FEED = False

    landmark_settings = load_settings(LANDMARK_CKPT_PATH)

    if LANDMARK_CKPT_PATH != '':
        # assert os.path.exists(LANDMARK_CKPT_PATH), 'Landmark checkpoint not exist: %s' % LANDMARK_CKPT_PATH
        DEPTH_MULTIPLIER = landmark_settings['depth_multiplier']
        DEPTH_GAMMA = landmark_settings['depth_gamma']
        NORM_FN = landmark_settings['normalizer_fn']
        NORM_PARAM = landmark_settings['normalizer_params']
        LANDMARK_BATCHSIZE = 8

        print(DEPTH_GAMMA, DEPTH_MULTIPLIER)

        landmark_estimator = infer.Classifier(LANDMARK_INPUT_SIZE, LANDMARK_CKPT_PATH,
                                              depth_multiplier=DEPTH_MULTIPLIER, depth_gamma=DEPTH_GAMMA,
                                              normalizer_fn=NORM_FN, normalizer_params=NORM_PARAM,
                                              batch_size=LANDMARK_BATCHSIZE)

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
        cap = cv2.VideoCapture(0)
        LIVE_FEED = True

    WRITE_WIDTH = int(1980/2)
    WRITE_HEIGHT = int(1080/2)
    video_writer = cv2.VideoWriter("/Users/gglee/Data/out.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15.0, (WRITE_WIDTH, WRITE_HEIGHT))

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
                    entry = {'folder': os.path.splitext(VIDEO_PATH)[0], 'basename': '%05d' % int(cap.get(cv2.CAP_PROP_POS_FRAMES)) }
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

                image_draw = deepcopy(image)
                image_np = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                patches = np.zeros((8, LANDMARK_INPUT_SIZE, LANDMARK_INPUT_SIZE, 3), dtype=np.float32)

                boxes = np.squeeze(boxes)
                scores = np.squeeze(scores)
                classes = np.squeeze(classes).astype(np.int32)

                HEIGHT, WIDTH, _ = image.shape
                crop_boxes = []

                for i, box in enumerate(boxes):
                    if scores[i] < FACE_SCORE_THRESHOLD:
                        continue

                    l, t, r, b = int(box[1] * WIDTH), int(box[0] * HEIGHT), int(box[3] * WIDTH), int(box[2] * HEIGHT)

                    cv2.rectangle(image_draw, (l, t), (r, b), (0, 0, 255), 2)   # red
                    cv2.circle(image_draw, (l, t), 3, (255, 255, 0), -1)

                    w, h = r - l, b - t

                    if w == 0 or h == 0:
                        print(box, '[%d, %d, %d, %d] w/ %.2f' %(l, t, r, b, scores[i]))
                        continue

                    # make crop-box centered and expanded
                    cx, cy = (l + r) / 2.0, (b + t) / 2.0
                    w, h = (r - l), (b - t)
                    ts = max(w, h) * 1.1 / 2.0              # expand 10%

                    l = int(min(max(0.0, cx - ts), WIDTH))
                    t = int(min(max(0.0, cy - ts), HEIGHT))
                    r = int(min(max(0.0, cx + ts), WIDTH))
                    b = int(min(max(0.0, cy + ts), HEIGHT))
                    crop_boxes.append([l, t, r, b])

                    cv2.rectangle(image_draw, (l, t), (r, b), (0, 255, 0), 2)       # green: expanded

                for i, box in enumerate(crop_boxes):
                    if landmark_estimator:
                        if i < LANDMARK_BATCHSIZE:
                            l, t, r, b = box[0], box[1], box[2], box[3]
                            face = image[t:b, l:r, :]

                            patch = cv2.resize(face, (LANDMARK_INPUT_SIZE, LANDMARK_INPUT_SIZE))
                            patches[i, :, :, :] = ((np.asarray(patch).astype(np.float32))/255.0-1.0)
                            # verify = ((np.asarray(patches[i, :, :, 0]).squeeze()+1.0)*255.0).astype(np.uint8)
                            # cv2.imshow("verify", verify)

                if LANDMARK_CKPT_PATH:
                    landmarks = np.reshape(np.squeeze(landmark_estimator.predict(patches)), (-1, 68, 2))

                    face_landmarks = []

                    for i, box in enumerate(crop_boxes):
                        face_landmarks.append({'face': box, 'landmark': landmarks[i]})

                        H, W = box[3]-box[1], box[2] - box[0]
                        for j in range(68):
                            if j not in [16, 21, 26, 41, 30, 47, 35, 59, 67]:
                                p0, p1 = landmarks[i][j], landmarks[i][j+1]
                                cv2.line(image_draw, (int(box[0]+(p0[0]*W)+0.5), int(box[1]+(p0[1]*H)+0.5)),
                                         (int(box[0] + (p1[0] * W) + 0.5), int(box[1] + (p1[1] * H) + 0.5)), (0, 0, 255), 1)
                        for p in landmarks[i]:
                            cv2.circle(image_draw, (int(box[0]+(p[0]*W)+0.5), int(box[1]+(p[1]*H)+0.5)), 2, (0, 255, 255))

                    if FLAGS.write_txt_dir and not LIVE_FEED:
                        save_dir = os.path.join(entry['folder'], WRITE_TXT_DIR)
                        if not os.path.exists(save_dir):
                            os.mkdir(save_dir, 0755)

                        write_detection(save_dir, entry['basename'], face_landmarks)

                cv2.imshow("image", cv2.flip(image_draw, 1))

                if video_writer.isOpened():
                    image_write = cv2.resize(image_draw, (WRITE_WIDTH, WRITE_HEIGHT))
                    video_writer.write(image)
                    
                key = cv2.waitKey(1)
                if key == 113 or key == 120:
                    video_writer.release()
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

# python detect_face_landmark.py --checkpoint_dir=/Users/gglee/Data/1207_face/
# python detect_face_landmark.py --face_checkpoint_dir=/Users/gglee/Data/TFModels/ssd_mobilenet_v2_quantized_160x160_v3/freeze
# python detect_face_landmark.py --face_checkpoint_dir=/Users/gglee/Data/TFModels/ssd_mobilenet_v2_quantized_160_v5/freeze/
# python detect_face_landmark.py --face_checkpoint_dir=/Users/gglee/Data/TFModels/ssd_mobilenet_v2_quantized_160_v5/freeze/ --write_dir_name=160v5 --folder_list=./folder.txt
# python detect_face_landmark.py --face_checkpoint_dir=/Users/gglee/Data/TFModels/ssd_mobilenet_v2_quantized_160_v5/freeze/ --landmark_checkpoint_path=/Users/gglee/Data/Landmark/train/0403_gpu1/x103_l1_sgd_0.003_lrd_0.6_200k_bn_l2_0.005/model.ckpt-900000