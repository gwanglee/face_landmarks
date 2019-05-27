#!/usr/bin/env bash
TFR_PATH="/home/gglee/Data/Landmark"
TRAIN_PATH="/home/gglee/Data/Landmark/train"
TRAIN_NAME="0425"

TRAIN_56="0424.56.train.tfrecord"
VAL_56="0424.56.val.tfrecord"

TRAIN_48="0424.48.train.tfrecord"
VAL_48="0424.48.val.tfrecord"

TRAIN_56_GRAY="0424.56.gray.train.tfrecord"
VAL_56_GRAY="0424.56.gray.val.tfrecord"

MAX_STEP=1100000

LR=0.01
LR_DECAY=0.5
LR_DECAY_STEP=300000

QUANT_STEP=-1

BATCH_SIZE=32
DEPTH_MULTIPLIER=1
DEPTH_GAMMA=1

LOSS="l1"
OPTIMIZER="adam"
REGULARIZER="l2"
REG_LAMBDA=0.0005

DEPTH_MULTIPLIER=5.0
DEPTH_GAMMA=0.75

EXP_NAME="x101"
TRAIN=$TRAIN_56
VAL=$VAL_56

#python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/$EXP_NAME-$LOSS.$OPTIMIZER.$LR.$LR_DECAY.$LR_DECAY_STEP-$REGULARIZER.$REG_LAMBDA \
#			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
#			--optimizer=$OPTIMIZER --loss=$LOSS --learning_rate_decay_type=exponential --learning_rate=$LR \
#			--learning_rate_decay_factor=$LR_DECAY --learning_rate_decay_step=$LR_DECAY_STEP \
#			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER --depth_gamma=$DEPTH_GAMMA \
#			--use_batch_norm=True --regularizer=$REGULARIZER --regularizer_lambda=$REG_LAMBDA \
#			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE


EXP_NAME="x102"
TRAIN=$TRAIN_48
VAL=$VAL_48

python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/$EXP_NAME-$LOSS.$OPTIMIZER.$LR.$LR_DECAY.$LR_DECAY_STEP-$REGULARIZER.$REG_LAMBDA \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=$OPTIMIZER --loss=$LOSS --learning_rate_decay_type=exponential --learning_rate=$LR \
			--learning_rate_decay_factor=$LR_DECAY --learning_rate_decay_step=$LR_DECAY_STEP \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER --depth_gamma=$DEPTH_GAMMA \
			--use_batch_norm=True --regularizer=$REGULARIZER --regularizer_lambda=$REG_LAMBDA \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE --input_size=48