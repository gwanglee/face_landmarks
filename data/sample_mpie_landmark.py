''' Pick random sample of Multi-PIE dataset to use as landmark training data.
Multi-PIE dataset contains very large amount of images. So use random sample to balance the data with others.
While sampling and copying image and data (.pts), valid data (contains 68 points) are only considered.
'''

import os
from shutil import copyfile
from random import random

RANDOM_SAMPLE = 30

mpie_dir = '/home/gglee/Data/Multi-Pie/data'
save_dir = '/home/gglee/Data/Multi-Pie/landmark'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

num_copied = 0
for root, dirs, files in os.walk(mpie_dir):
    for f in files:
        if random() > 1.0/RANDOM_SAMPLE:
            continue

        if f.endswith('.pts'):
            basename = f.split('.')[0]
            if os.path.exists(os.path.join(root, basename+'.png')):
                imgfile = basename + '.png'
            elif os.path.exists(os.path.join(root, basename+'.jpg')):
                imgfile = basename + '.jpg'
            else:
                continue

            with open(os.path.join(root, f), 'r') as rf:
                num_pts = int(rf.readline().strip())
                if num_pts != 68:
                    continue
                pts = []
                for i in range(68):
                    x, y = rf.readline().split()
                    x, y = int(x.strip()), int(y.strip())
                    pts.append([x, y])

            with open(os.path.join(save_dir, f), 'w') as wf:
                wf.write('version: m\n')
                wf.write('n_points: %d\n' % num_pts)
                wf.write('{\n')
                for p in pts:
                    wf.write('%.3f %.3f\n' %(p[0], p[1]))
                wf.write('}\n')

            copyfile(os.path.join(root, imgfile), os.path.join(save_dir, imgfile))
            print('[%d] copying: %s > %s' % (num_copied, imgfile, save_dir))
            num_copied += 1