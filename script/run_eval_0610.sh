#!/usr/bin/env bash
#NAME='ssd_face_128_v21'
#python util/detect_face_landmark.py --face_checkpoint_dir=/Users/gglee/Data/TFModels/0610/$NAME/freeze \
#                                --images_dir=/Users/gglee/Data/face_eval/data \
#                                --write_txt_dir=/Users/gglee/Data/face_eval/det/$NAME \
#                                --disp=True
#
#NAME='ssd_face_128_v22'
#python util/detect_face_landmark.py --face_checkpoint_dir=/Users/gglee/Data/TFModels/0610/$NAME/freeze \
#                                --images_dir=/Users/gglee/Data/face_eval/data \
#                                --write_txt_dir=/Users/gglee/Data/face_eval/det/$NAME \
#                                --disp=True
#
#NAME='ssd_face_128_v24'
#python util/detect_face_landmark.py --face_checkpoint_dir=/Users/gglee/Data/TFModels/0610/$NAME/freeze \
#                                --images_dir=/Users/gglee/Data/face_eval/data \
#                                --write_txt_dir=/Users/gglee/Data/face_eval/det/$NAME \
#                                --disp=True
#
#NAME='ssd_face_128_v25'
#python util/detect_face_landmark.py --face_checkpoint_dir=/Users/gglee/Data/TFModels/0610/$NAME/freeze \
#                                --images_dir=/Users/gglee/Data/face_eval/data \
#                                --write_txt_dir=/Users/gglee/Data/face_eval/det/$NAME \
#                                --disp=True
#
#NAME='ssd_face_160_v23'
#python util/detect_face_landmark.py --face_checkpoint_dir=/Users/gglee/Data/TFModels/0610/$NAME/freeze \
#                                --images_dir=/Users/gglee/Data/face_eval/data \
#                                --write_txt_dir=/Users/gglee/Data/face_eval/det/$NAME \
#                                --disp=True
#
#NAME='ssd_face_160_v26'
#python util/detect_face_landmark.py --face_checkpoint_dir=/Users/gglee/Data/TFModels/0610/$NAME/freeze \
#                                --images_dir=/Users/gglee/Data/face_eval/data \
#                                --write_txt_dir=/Users/gglee/Data/face_eval/det/$NAME \
#                                --disp=True
#
#NAME='ssd_face_160_v29'
#python util/detect_face_landmark.py --face_checkpoint_dir=/Users/gglee/Data/TFModels/0610/$NAME/freeze \
#                                --images_dir=/Users/gglee/Data/face_eval/data \
#                                --write_txt_dir=/Users/gglee/Data/face_eval/det/$NAME \
#                                --disp=True

#NAME='ssd_face_128_v28'
#python util/detect_face_landmark.py --face_checkpoint_dir=/Users/gglee/Data/TFModels/0610/$NAME/freeze \
#                                --images_dir=/Users/gglee/Data/face_eval/data \
#                                --write_txt_dir=/Users/gglee/Data/face_eval/det/$NAME \
#                                --disp=True
#
#NAME='ssd_face_128_v30'
#python util/detect_face_landmark.py --face_checkpoint_dir=/Users/gglee/Data/TFModels/0610/$NAME/freeze \
#                                --images_dir=/Users/gglee/Data/face_eval/data \
#                                --write_txt_dir=/Users/gglee/Data/face_eval/det/$NAME \
#                                --disp=True
#
#NAME='ssd_face_144_v27'
#python util/detect_face_landmark.py --face_checkpoint_dir=/Users/gglee/Data/TFModels/0610/$NAME/freeze \
#                                --images_dir=/Users/gglee/Data/face_eval/data \
#                                --write_txt_dir=/Users/gglee/Data/face_eval/det/$NAME \
#                                --disp=True

NAME='ssd_face_128_v0'
python util/detect_face_landmark.py --face_checkpoint_dir=/Users/gglee/Data/TFModels/0610/$NAME/freeze \
                                --images_dir=/Users/gglee/Data/face_eval/data \
                                --write_txt_dir=/Users/gglee/Data/face_eval/det/$NAME \
                                --disp=True

NAME='ssd_face_300_v0'
python util/detect_face_landmark.py --face_checkpoint_dir=/Users/gglee/Data/TFModels/0610/$NAME/freeze \
                                --images_dir=/Users/gglee/Data/face_eval/data \
                                --write_txt_dir=/Users/gglee/Data/face_eval/det/$NAME \
                                --disp=True

python util/eval_wider_dets.py --det_root=/Users/gglee/Data/face_eval/det --image_dir=/Users/gglee/Data/face_eval/data \
                                --gt_path=/Users/gglee/Data/face_eval/data/gt.txt