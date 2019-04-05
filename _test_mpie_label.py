import os
from scipy.io import loadmat

labels_dir = '/home/gglee/Data/Multi-Pie/Labels/labels'
images_dir = '/home/gglee/Data/Multi-Pie/data'
label_data = []

for d in os.listdir(labels_dir):
    curd = os.path.join(labels_dir, d)
    if os.path.isfile(curd):
        continue

    for f in os.listdir(curd):
        curf = os.path.join(curd, f)
        if os.path.isfile(curf):# and not curf.startswith('_') and curf.endswith('mat'):
            m = loadmat(curf)
            num_pts = len(m['pts'])
            label_data.append({'filename': f.split('.')[0], 'num_pts': num_pts})

valids = filter(lambda x: x['num_pts']==68, label_data)
# print(valids)
print('68 points labels = %d' % len(filter(lambda x: x['num_pts'] == 68, label_data)))

valid_files = [x['filename'] for x in valids]
# print(valid_files)

data = []
for root, dirs, files in os.walk(images_dir):
    # print('root: %s, dirs: %s, files: %s\n' % (root, dirs, files))
    for name in files:
        if name.split('.')[0]+'_lm' in valid_files:
            data.append({'dir': root, 'png': name})

# print(data)
print(len(data))