import tensorflow as tf
import cv2

def _parse_function(example_proto, augment=False):
    features = {"image": tf.FixedLenFeature([56*56*3], tf.string),
                "points": tf.FixedLenFeature([68*2], tf.float32)}
    parsed_features = tf.parse_single_example(example_proto, features)

    img = tf.reshape(tf.decode_raw(parsed_features["image"], tf.uint8), (56, 56, 3))

    if augment:
        img = tf.image.random_brightness(img, max_delta=0.5)
        img = tf.image.random_contrast(img, lower=0.2, upper=2.0)
        img = tf.image.random_hue(img, max_delta=0.08)

    normed = tf.subtract(tf.multiply(tf.cast(img, tf.float32), 2.0 / 255.0), 1.0)

    pts = tf.reshape(tf.cast(parsed_features['points'], tf.float32), (136, ))

    return normed, pts


def load_tfrecord(tfr_path, batch_size=64, num_parallel_calls=8, augment=False):
    dataset = tf.data.TFRecordDataset(tfr_path)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(1000)
    dataset = dataset.map(lambda ex: _parse_function(ex, augment), num_parallel_calls=num_parallel_calls).cache()
    dataset = dataset.batch(batch_size)
    # if is_training:
    #     dataset = dataset.map(_train_preprocess)
    dataset.prefetch(buffer_size=batch_size)

    return dataset

#
# def _train_preprocess(image, points):
#     image = tf.image.random_brightness(image, max_delta=0.5)
#     image = tf.image.random_contrast(image, lower=0.2, upper=2.0)
#     image = tf.image.random_hue(image, max_delta=0.08)
#
#     return image, points

def get_tfr_record_count(tfr_path):
    count = 0
    for r in tf.python_io.tf_record_iterator(tfr_path):
        count += 1

    return count


if __name__=='__main__':
    with tf.Graph().as_default():
        TRAIN_TFR_PATH = '/Users/gglee/Data/Landmark/export/160v5.0322.train.tfrecord'
        VAL_TFR_PATH = '/Users/gglee/Data/Landmark/export/160v5.0322.val.tfrecord'

        num_train = get_tfr_record_count(TRAIN_TFR_PATH)
        num_val = get_tfr_record_count(VAL_TFR_PATH)

        data_train = load_tfrecord(TRAIN_TFR_PATH, is_training=True)
        data_val = load_tfrecord(VAL_TFR_PATH)

        iter_train = data_train.make_initializable_iterator()
        iter_val = data_val.make_initializable_iterator()

        img_train, pts_train = iter_train.get_next()
        val_imgs, val_pts = iter_val.get_next()

        with tf.Session() as sess:
            # sess.run(iter_train.initializer, iter_val.initializer)
            cnt = 0
            while cnt < num_train:
                images, pts = sess.run(img_train, pts_train)

                for i in range(64):
                    curi = images[i, :, :, :]
                    curp = pts[i, :, :, :]
                    print(curp)
                cnt += 64