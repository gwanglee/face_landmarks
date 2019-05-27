#!/usr/bin/env bash
TFR_PATH="/youjin/face_landmark/data"
TRAIN_PATH="/youjin/face_landmark/train"
TRAIN_NAME="0417_gpu2"

TRAIN="0407.ext.train.tfrecord"
VAL="0407.ext.val.tfrecord"

MAX_STEP=1990000
LR_DECAY_STEP=360000

QUANT_STEP=-1

BATCH_SIZE=64
DEPTH_MULTIPLIER=1

LOSS="l1"
OPTIMIZER="sgd"
REGULARIZER="l2"
REG_LAMBDA=0.005

EXP_NAME="x101"
LR=0.01
LR_DECAY=0.5
python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/$EXP_NAME-$LOSS.$OPTIMIZER.$LR.$LR_DECAY.$LR_DECAY_STEP-$REGULARIZER.$REG_LAMBDA \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=$OPTIMIZER --loss=$LOSS --learning_rate_decay_type=exponential --learning_rate=$LR \
			--learning_rate_decay_factor=$LR_DECAY --learning_rate_decay_step=$LR_DECAY_STEP \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--use_batch_norm=False --regularizer=$REGULARIZER --regularizer_lambda=$REG_LAMBDA \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE


EXP_NAME="x102"
LR=0.02
LR_DECAY=0.5
python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/$EXP_NAME-$LOSS.$OPTIMIZER.$LR.$LR_DECAY.$LR_DECAY_STEP-$REGULARIZER.$REG_LAMBDA \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=$OPTIMIZER --loss=$LOSS --learning_rate_decay_type=exponential --learning_rate=$LR \
			--learning_rate_decay_factor=$LR_DECAY --learning_rate_decay_step=$LR_DECAY_STEP \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--use_batch_norm=False --regularizer=$REGULARIZER --regularizer_lambda=$REG_LAMBDA \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE


EXP_NAME="x103"
LR=0.02
LR_DECAY=0.25
python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/$EXP_NAME-$LOSS.$OPTIMIZER.$LR.$LR_DECAY.$LR_DECAY_STEP-$REGULARIZER.$REG_LAMBDA \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=$OPTIMIZER --loss=$LOSS --learning_rate_decay_type=exponential --learning_rate=$LR \
			--learning_rate_decay_factor=$LR_DECAY --learning_rate_decay_step=$LR_DECAY_STEP \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--use_batch_norm=False --regularizer=$REGULARIZER --regularizer_lambda=$REG_LAMBDA \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE

