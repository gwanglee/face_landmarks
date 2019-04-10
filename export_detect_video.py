import os
import cv2
import tensorflow as tf
from detect_face_landmark import read_detection

flags = tf.app.flags
flags.DEFINE_string('images', None,
                    'Directory containing images')

flags.DEFINE_string('detections', None,
                    'directory names that contain detection results. must be subfolders of the images_dir (comma separated)')

# flags.DEFINE_bool('disp', 'True',
#                   'If true, show detection images (only for file input)')

FLAGS = flags.FLAGS

COLOR = [[(75, 25, 230), (0, 0, 128)], # Red
         [(75, 170, 60), (0, 128, 0)], # Green
         [(200, 130, 0), (128, 0, 0)], # Blue
         [(48, 130, 245), (40, 110, 170)], # Orange
         [(240, 240, 70), (128, 128, 0)],  # Cyan
         [(25, 225, 255), (0, 128, 128)],  # Yellow
        ]

FPS = 15

PAIRS = []
for p in range(68):
    if p not in [17 - 1, 22 - 1, 27 - 1, 42 - 1, 31 - 1, 36 - 1, 48 - 1, 60 - 1, 68 - 1]:
        PAIRS.append([p, p + 1])
    PAIRS.append([41, 36])
    PAIRS.append([47, 42])
    PAIRS.append([59, 48])
    PAIRS.append([67, 60])


if __name__=='__main__':
    if not FLAGS.images:
        print('no such folder: %s' % FLAGS.images)
        exit()

    root = FLAGS.images
    filenames = []
    for f in os.listdir(root):
        if os.path.splitext(f)[1].lower() in ['.jpg', '.png']:
            filenames.append(f)

    filenames = sorted(filenames)

    if len(filenames) == 0:
        print('no images')
        exit()

    subs = FLAGS.detections
    subs = [x.strip() for x in subs.split(',')]

    sample = cv2.imread(os.path.join(root, filenames[0]))
    HEIGHT, WIDTH, _ = sample.shape
    writer = cv2.VideoWriter(os.path.join(root, 'result.avi'), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), FPS, (WIDTH, HEIGHT))

    for f in filenames:
        image = cv2.imread(os.path.join(root, f))

        for i, sub in enumerate(subs):
            detections = read_detection(os.path.join(root, sub, os.path.splitext(f)[0] + '.txt'))

            for d in detections:
                l, t, r, b = d['face']
                cv2.rectangle(image, (l, t), (r, b), (255, 255, 255), 2)

                w, h = r - l, b - t

                for p in PAIRS:
                    x0 = int(d['landmark'][p[0]*2] * w + l)
                    y0 = int(d['landmark'][p[0]*2+1] * h + t)
                    x1 = int(d['landmark'][p[1]*2] * w + l)
                    y1 = int(d['landmark'][p[1]*2+1] * h + t)
                    cv2.line(image, (x0, y0), (x1, y1), COLOR[i][0])

                for p in range(68):
                    x = int(d['landmark'][p*2]*w + l)
                    y = int(d['landmark'][p*2+1]*h + t)
                    cv2.circle(image, (x, y), 2, COLOR[i][1])

        cv2.imshow('image', image)
        writer.write(image)

        cv2.waitKey(1)