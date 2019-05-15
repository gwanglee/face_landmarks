1. Prepare training data

face_landmarks gglee$ python refine_widerface_2.py --image_dir=/Users/gglee/Data/WiderFace/WIDER_train/images/ --gt_path=/Users/gglee/Data/WiderFace/wider_face_split/wider_face_train_bbx_gt.txt --output_image_dir=/Users/gglee/Data/WiderRefine/train/ --output_gt_path=/Users/gglee/Data/WiderRefine/train_gt.txt


face_landmarks gglee$ python refine_widerface_2.py --image_dir=/Users/gglee/Data/face_ours/ --gt_path=/Users/gglee/Data/face_ours/face_ours.txt --output_image_dir=/Users/gglee/Data/WiderRefine/face_ours --output_gt_path=/Users/gglee/Data/WiderRefine/face_ours_train_gt.txt


python make_widerface_tfrecord.py --image_dir=/Users/gglee/Data/face_train/ --gt_path=/Users/gglee/Data/face_train/train_gt.txt --output_path=/Users/gglee/Data/face_train/face_train_0515.tfrecord --negative_path=/Users/gglee/Data/face_negative/


2. Run training