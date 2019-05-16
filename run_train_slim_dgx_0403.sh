#!/usr/bin/env bash
TFR_PATH="/youjin/face_landmark/data"
TRAIN_PATH="/youjin/face_landmark/train"
TRAIN_NAME="0403"

TRAIN="160v5.0402.train.tfrecord"
VAL="160v5.0402.val.tfrecord"

TRAIN_E="160v5.0402.ext.train.tfrecord"
VAL_E="160v5.0402.ext.val.tfrecord"

TRAIN_C="160v5.0402.cen.train.tfrecord"
VAL_C="160v5.0402.cen.val.tfrecord"

TRAIN_CE="160v5.0402.cen.ext.train.tfrecord"
VAL_CE="160v5.0402.cen.ext.val.tfrecord"

TRAIN=$TRAIN_E
VAL=$VAL_E

MAX_STEP=520000
LR_DECAY_STEP_0=120000
QUANT_STEP=-1

BATCH_SIZE=32
DEPTH_MULTIPLIER=2

# 001, 002, 003: lr scheduling [0.1/120k, 0.3/120k, 0.6/120k]
python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/x001_l1_sgd_0.003_lrd_0.1_120k_bn_l2_0.005 \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=sgd --loss=l1 --learning_rate_decay_type=exponential --learning_rate=0.003 \
			--learning_rate_decay_factor=0.1 --learning_rate_decay_step=$LR_DECAY_STEP_0 \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--use_batch_norm=True --regularizer=l2 --regularizer_lambda=0.005 \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE

python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/x002_l1_sgd_0.003_lrd_0.3_120k_bn_l2_0.005 \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=sgd --loss=l1 --learning_rate_decay_type=exponential --learning_rate=0.003 \
			--learning_rate_decay_factor=0.3 --learning_rate_decay_step=$LR_DECAY_STEP_0 \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--use_batch_norm=True --regularizer=l2 --regularizer_lambda=0.005 \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE

python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/x003_l1_sgd_0.003_lrd_0.5_120k_bn_l2_0.005 \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=sgd --loss=l1 --learning_rate_decay_type=exponential --learning_rate=0.003 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=$LR_DECAY_STEP_0 \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--use_batch_norm=True --regularizer=l2 --regularizer_lambda=0.005 \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE

# 004, 005, 006: wing param: w= [0.1, 1.0, 5.0], eps=[0.1, 1.0, 5.0]
python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/x004_wing_sgd_0.003_lrd_0.5_120k_bn_l2_0.005 \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=sgd --loss=wing --wing_w=0.1 --wing_eps=1.0 \
			--learning_rate_decay_type=exponential --learning_rate=0.003 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=$LR_DECAY_STEP_0 \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--use_batch_norm=True --regularizer=l2 --regularizer_lambda=0.005 \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE

python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/x005_wing_sgd_0.003_lrd_0.5_120k_bn_l2_0.005 \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=sgd --loss=wing --wing_w=1.0 --wing_eps=1.0 \
			--learning_rate_decay_type=exponential --learning_rate=0.003 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=$LR_DECAY_STEP_0 \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--use_batch_norm=True --regularizer=l2 --regularizer_lambda=0.005 \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE

python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/x006_wing_sgd_0.003_lrd_0.5_120k_bn_l2_0.005 \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=sgd --loss=wing --wing_w=5.0 --wing_eps=1.0 \
			--learning_rate_decay_type=exponential --learning_rate=0.003 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=$LR_DECAY_STEP_0 \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--use_batch_norm=True --regularizer=l2 --regularizer_lambda=0.005 \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE

python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/x007_wing_sgd_0.003_lrd_0.5_120k_bn_l2_0.005 \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=sgd --loss=wing --wing_w=1.0 --wing_eps=0.1 \
			--learning_rate_decay_type=exponential --learning_rate=0.003 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=$LR_DECAY_STEP_0 \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--use_batch_norm=True --regularizer=l2 --regularizer_lambda=0.005 \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE

python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/x008_wing_sgd_0.003_lrd_0.5_120k_bn_l2_0.005 \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=sgd --loss=wing --wing_w=1.0 --wing_eps=5.0 \
			--learning_rate_decay_type=exponential --learning_rate=0.003 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=$LR_DECAY_STEP_0 \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--use_batch_norm=True --regularizer=l2 --regularizer_lambda=0.005 \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE

python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/x009_wing_sgd_0.003_lrd_0.5_120k_bn_l2_0.005 \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=sgd --loss=wing --wing_w=0.1 --wing_eps=5.0 \
			--learning_rate_decay_type=exponential --learning_rate=0.003 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=$LR_DECAY_STEP_0 \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--use_batch_norm=True --regularizer=l2 --regularizer_lambda=0.005 \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE

python evaluate.py --tfrecord=$TFR_PATH/$VAL --models_dir=$TRAIN_PATH/$TRAIN_NAME
