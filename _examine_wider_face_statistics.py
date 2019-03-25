import widerface_explorer
import cv2
import numpy as np

if __name__=='__main__':
    DB_PATH='/Users/gglee/Data/WiderRefine/train_fr/'
    GT_PATH='/Users/gglee/Data/WiderRefine/train_fr/wider_refine_train_gt.txt'
    BINS = 50

    wdb = widerface_explorer.wider_face_db(DB_PATH, GT_PATH)

    total_images = wdb.get_image_count()
    hist = np.zeros(BINS, dtype=int)

    for idx in range(total_images):
        data = wdb.get_annos_by_image_index(idx)
        image_path = data['image_path']
        image = cv2.imread(image_path)
        H, W = image.shape[0:2]

        annos = data['annos']

        for a in annos:
            pos = int(((a['w'] / float(W)) - 0.000001) * BINS)
            hist[pos] += 1

    print(hist)