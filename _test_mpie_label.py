import os
from scipy.io import loadmat
from shutil import copyfile

mpie_dir = '/home/gglee/Data/Multi-Pie/data'
save_dir = '/home/gglee/Data/Multi-Pie/landmark'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

num_copied = 0
for root, dirs, files in os.walk(mpie_dir):
    for f in files:
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


# labels_dir = '/home/gglee/Data/Multi-Pie/Labels/labels'
# images_dir = '/home/gglee/Data/Multi-Pie/data'
# # label_data = []
# mat_files = []
#
# def save_pts(dir, filename, m):
#     if not os.path.exists(dir) and not os.path.isdir(dir):
#         raise ValueError('no such directory: %s ' % (dir))
#
#     with open(os.path.join(dir, os.path.splitext(filename)[0] + '.pts'), 'w') as wf:
#         wf.write('version: %s\n' % m['__version__'])
#         wf.write('n_points: %d\n' % len(m['pts']))
#         wf.write('{\n')
#         for p in m['pts']:
#             wf.write('%.3f %.3f\n' % (p[0], p[1]))
#         wf.write('}\n')
#
#
# save_dir = '/home/gglee/Data/Multi-Pie/landmark'
# if not os.path.exists(save_dir):
#     os.mkdir(save_dir)
#
# for root, dirs, files in os.walk(labels_dir):
#     for f in files:
#         if f.endswith('.mat'):
#             matpath = os.path.join(root, f)
#             m = loadmat(matpath)
#             if len(m['pts']) == 68:
#                 #label_data.append({'mat': matpath})
#                 mat_files.append(matpath)
#                 save_pts(save_dir, f, m)
#
# num_copied = 0
# # valid_files = [x['mat'] for x in label_data]
# for root, dirs, files in os.walk(images_dir):
#     for name in files:
#         print(name)
#         if name.split('.')[0]+'_lm.pts' in mat_files:#valid_files:
#             print(name.split('.')[0]+'_lm.pts')
#             img_path = os.path.join(root, name)
#             copy_path = os.path.join(save_dir, name.split('.')[0] + '_lm.png')
#             copyfile(img_path, copy_path)
#             print('copying: %s > %s' % (img_path, copy_path))
#             num_copied += 1
#
# print('Done! %d image/annos. check %s' % (num_copied, save_dir))