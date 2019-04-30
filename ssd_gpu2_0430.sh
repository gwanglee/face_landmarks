# gpu=0
# export CUDA_VISIBLE_DEVICES=$gpu
CONFIG=ssd_face_144_v1.config
TRAIN=ssd_face_144_v1
export PYTHONPATH=/youjin/face_detection/Program/models_latest/research:/youjin/face_detection/Program/models_latest/research/slim

python ../../face_detection/Program/models_latest/research/object_detection/model_main.py --logtostderr \
--pipeline_config_path=./$CONFIG \
--model_dir=../train/$TRAIN