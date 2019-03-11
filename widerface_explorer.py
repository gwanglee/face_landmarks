import os
import cv2
import numpy as np
from random import randint

class wider_face_db(object):
    """A class to load and explore images and annotations in wider face dataset.

    It reads file lists and ground truth annotations from given input paths. The class instance
    binds to a image path and a ground truth text file, which means individual instance should be
    created for training and validation sets.

    Args:
        image_root_path: Directory where wider face images are stored. It contains sub-folders 
        of scene names such as: '0--Parade', ... , '61--StreetBattle;
        gt_txt_path: file path of the ground truth (i.e., wider_face_{train, val}_bbx_gt.txt)
        
    """

    def __init__(self, image_root_path, gt_txt_path):
        self._db_root_path = image_root_path
        self._gt_path = gt_txt_path

        files, annos = self.load_annos(self._gt_path)
        assert (files is not None and annos is not None), \
            "Failed to parse ground truth text file"

        self._files = files
        self._annos = annos

    def get_image_count(self):
        """ Return the number of images in the db """
        return self._num_images

    def get_annotation_count(self):
        """ Return the number of annotations in the db. 
        
        An image would contain one or more faces. So image count and annotation count can be (and are) different
        """
        return self._num_annos

    def get_annos_by_image_index(self, i):
        """ Get annoatation information for a certain face

        It returns a dict consist of 'image_path' and 'annos'. 'annos' has attributes
            return:
                + ['imaage_path']
                + ['annos']
                    + ['x'], ['y'], ['w'], ['h']
                    + ['blur'], ['expression'], ['illumination'], ['occlusion'], ['pose'], ['invalid']
        """
        assert (len(self._files) > 0 and len(self._annos) > 0), \
            "No annotations loaded"

        fpath = self._files['filename'][i]
        idx_annos = np.where(self._annos['file_id'] == i)[0]

        annos = []
        for idx in idx_annos:
            annos.append({'x': self._annos['x'][idx],
                          'y': self._annos['y'][idx],
                          'w': self._annos['w'][idx],
                          'h': self._annos['h'][idx],

                          'l': self._annos['x'][idx],
                          't': self._annos['y'][idx],
                          'r': self._annos['x'][idx] + self._annos['w'][idx],
                          'b': self._annos['y'][idx] + self._annos['h'][idx],

                          'blur': self._annos['blur'][idx],
                          'expression': self._annos['expression'][idx],
                          'illumination': self._annos['illumination'][idx],
                          'occlusion': self._annos['occlusion'][idx],
                          'pose': self._annos['pose'][idx],
                          'invalid': self._annos['invalid'][idx]
                        })

        return {'image_path': os.path.join(self._db_root_path, fpath), 'annos': annos}
    
    def load_annos(self, anno_path):
        """ returns to matrices which contains filename and annotations
        filename_matrix : [ fid, filename ]
        annotation_matrix: [ fid, x, y, w, h, blur, expr, illum, occ, pose, invalid ]

        filename: {'file_id': [], 'filename': [] }  # filename contains event name
        anno: {  'file_id': [], 'anno_id': [], 'x': [], ... }

        f_matrix and a_matrix are created for each event folder
        """

        if os.path.exists(anno_path) is False or os.path.isfile(anno_path) is False or anno_path.endswith('txt') is False:
            print("Wrong path: not exist or not a txt file: %s" % anno_path)
            return None, None

        list_file_id, list_anno_id = [], []
        list_x, list_y, list_w, list_h = [], [], [], []
        list_blur, list_expr, list_illum, list_occ, list_pose, list_inval = [], [], [], [], [], []
        anno_id = 0

        list_id = []
        list_filename = []
        file_id = 0

        num_annos_total = 0

        with open(anno_path) as afile:
            line = "begin"
            while line != "":
                line = afile.readline()

                if line.rstrip().endswith('jpg'):   # it is a file
                    file_name = line.strip()
                    list_id.append(file_id)
                    list_filename.append(file_name)

                    num_annos = int(afile.readline().strip())

                    for i in range(num_annos):
                        px, py, pw, ph, blur, expr, illum, inval, occ, pose = afile.readline().strip().split(' ')
                        px, py, pw, ph = int(px), int(py), int(pw), int(ph)

                        if pw == 0 or ph == 0:      # ignore invalid faces (0 width or height)
                            continue

                        if pw < 0:
                            px = px+pw
                            pw = abs(pw)
                        if ph < 0:
                            py = py+ph
                            ph = abs(ph)

                        list_file_id.append(file_id)
                        list_anno_id.append(anno_id)
                        list_x.append(px)
                        list_y.append(py)
                        list_w.append(pw)
                        list_h.append(ph)
                        list_blur.append(int(blur))
                        list_expr.append(int(expr))
                        list_illum.append(int(illum))
                        list_occ.append(int(occ))
                        list_pose.append(int(pose))
                        list_inval.append(int(inval))
                        anno_id = anno_id + 1

                    file_id = file_id + 1
                    num_annos_total += num_annos

        files = {'id': np.array(list_id), 'filename': list_filename }
        annos = {'file_id': np.array(list_file_id), 'anno_id': np.array(list_anno_id), \
                'x': np.array(list_x), 'y': np.array(list_y), \
                'w': np.array(list_w), 'h': np.array(list_h), \
                'blur': np.array(list_blur), 'expression': np.array(list_expr), \
                'illumination': np.array(list_illum), 'occlusion': np.array(list_occ), \
                'pose': np.array(list_pose), 'invalid': np.array(list_inval) }

        assert (len(list_id) == len(list_filename)), \
                "file_id and filename lists should have the same length"

        self._num_annos = num_annos_total
        self._num_images = file_id

        return files, annos


def draw_annos(image, annos):
    """ Test code to validate loaded annoataion values
    """
    for a in annos:
        blur = a['blur']
        expr = a['expression']
        occ = a['occlusion']
        pose = a['pose']
        illum = a['illumination']
        inval = a['invalid']

        x, y, w, h = a['x'], a['y'], a['w'], a['h']
        strs = []
        odds = 0
        color = (255, 255, 255)

        if expr == 1:
            color = (50, 50, 255)
            odds = odds + 1
            strs.append("expr")
        
        if occ > 0:
            color = (255, 50, 50)
            odds = odds + 1
            strs.append("occ (%d)"%occ)

        if pose == 1:
            color = (50, 255, 50)
            odds = odds + 1
            strs.append("pose")

        if blur > 0:
            color = (180, 180, 180)
            odds = odds + 1
            strs.append("blur (%d)"%blur)

        if illum == 1:
            color = (0, 255, 255)
            odds = odds + 1
            strs.append("illum")

        if odds > 1:
            color = (255, 0, 255)

        if inval == 1:
            color = (0, 0, 0)
            strs.append("invalid")

        cv2.rectangle(image, (x, y), (x+w+1, y+h+1), (0, 0, 0), 2)
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)

        for i, s in enumerate(strs):
            cv2.putText(image, s, (x+1, y+i*20+1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            cv2.putText(image, s, (x, y+i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

if __name__ == "__main__":
    # DB_ROOT = '/Users/gglee/Data/WiderFace/WIDER_val/images'
    # GT_PATH = '/Users/gglee/Data/WiderFace/wider_face_split/wider_face_val_bbx_gt.txt'
    DB_ROOT = '/Users/gglee/Data/tmp'
    GT_PATH = '/Users/gglee/Data/tmp/gt.txt'

    SAVE_PATH = '/Users/gglee/Data/WiderFace'
    
    wdb = wider_face_db(DB_ROOT, GT_PATH)
    total_images = wdb.get_image_count()
    total_annotations = wdb.get_annotation_count()

    print("DB load: %d images, %d annotations"%(total_images, total_annotations))

    for i in range(100):     # draw 10 images randomly
        idx = randint(0, total_images)
        data = wdb.get_annos_by_image_index(idx)

        print("%dth image: %s"%(idx, data['image_path']))
        image = cv2.imread(data['image_path'])

        draw_annos(image, data['annos'])
        for anno in data['annos']:
            print(anno)

        cv2.imshow("image", image)
        cv2.imwrite(os.path.join(SAVE_PATH, "val_%03d.jpg"%i), image)
        cv2.waitKey(01)
