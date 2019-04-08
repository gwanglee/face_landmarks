import tensorflow as tf
import cv2
import numpy as np

def _parse_function(example_proto, input_size=56, is_color=True, augment=False):
    CH = 3 if is_color else 1
    SIZE = input_size
    NUM_LANDMARKS = 68

    features = {"image": tf.FixedLenFeature([SIZE*SIZE*CH], tf.string),
                "points": tf.FixedLenFeature([NUM_LANDMARKS*2], tf.float32)}
    parsed_features = tf.parse_single_example(example_proto, features)

    img = tf.reshape(tf.decode_raw(parsed_features["image"], tf.uint8), (SIZE, SIZE, CH))

    if augment:
        img = tf.image.random_brightness(img, max_delta=0.5)
        img = tf.image.random_contrast(img, lower=0.2, upper=2.0)
        img = tf.image.random_hue(img, max_delta=0.08)

    normed = tf.subtract(tf.multiply(tf.cast(img, tf.float32), 2.0 / 255.0), 1.0)

    pts = tf.reshape(tf.cast(parsed_features['points'], tf.float32), (NUM_LANDMARKS*2, ))

    return normed, pts


def load_tfrecord(tfr_path, input_size=56, batch_size=64, num_parallel_calls=8, is_color=True, shuffle=False, augment=False):
    dataset = tf.data.TFRecordDataset(tfr_path)
    dataset = dataset.repeat()
    if shuffle:
        dataset = dataset.shuffle(1000)
    dataset = dataset.map(lambda ex: _parse_function(ex, input_size=input_size, is_color=is_color, augment=augment),
                          num_parallel_calls=num_parallel_calls)#.cache()
    dataset = dataset.batch(batch_size)
    dataset.prefetch(buffer_size=batch_size)

    return dataset


def get_tfr_record_count(tfr_path):
    count = 0
    for r in tf.python_io.tf_record_iterator(tfr_path):
        count += 1

    return count


if __name__=='__main__':
    with tf.Graph().as_default():
        TRAIN_TFR_PATH = '/Users/gglee/Data/Landmark/export/160v5.0402.ext.train.tfrecord'

        num_train = get_tfr_record_count(TRAIN_TFR_PATH)
        data_train = load_tfrecord(TRAIN_TFR_PATH, augment=True, batch_size=8*8)
        iter_train = data_train.make_initializable_iterator()
        img_train, pts_train = iter_train.get_next()

        with tf.Session() as sess:
            cnt = 0
            sess.run([iter_train.initializer])

            while cnt < num_train:
                images, pts = sess.run([img_train, pts_train])

                for i in range(64):
                    curi = images[i, :, :, :]
                    curp = pts[i, :]
                    img = np.asarray((curi + 1.0) * 255.0 / 2.0, dtype=np.uint8)

                    for p in range(68):
                        x = int(curp[p*2]*56)
                        y = int(curp[p*2+1]*56)
                        cv2.circle(img, (x, y), 2, (0, 255, 0))

                    cv2.imshow('image', img)
                    if cv2.waitKey(100) in [113, 120, 99]:  # q, x, c to quit
                        exit()
                cnt += 64