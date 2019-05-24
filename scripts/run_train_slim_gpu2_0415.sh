#!/usr/bin/env bash
TFR_PATH="/youjin/face_landmark/data"
TRAIN_PATH="/youjin/face_landmark/train"
TRAIN_NAME="0415_gpu2"

TRAIN="0407.ext.train.tfrecord"
VAL="0407.ext.val.tfrecord"

MAX_STEP=840000
LR_DECAY_STEP=240000
BATCH_SIZE=32
QUANT_STEP=-1

LOSS="l1"
OPTIMIZER="sgd"
LR=0.01
LR_DECAY=0.25

REGULARIZER="l2"
REG_LAMBDA=0.05

EXP_NAME="x101"
DEPTH_MULTIPLIER=1.0
DEPTH_GAMMA=1.0

python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/$EXP_NAME-$LOSS.$OPTIMIZER.$LR.$LR_DECAY.$LR_DECAY_STEP-$REGULARIZER.$REG_LAMBDA \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=$OPTIMIZER --loss=$LOSS --learning_rate_decay_type=exponential --learning_rate=$LR \
			--learning_rate_decay_factor=$LR_DECAY --learning_rate_decay_step=$LR_DECAY_STEP \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER --depth_gamma=$DEPTH_GAMMA \
			--use_batch_norm=True --regularizer=$REGULARIZER --regularizer_lambda=$REG_LAMBDA \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE

EXP_NAME="x102"
DEPTH_MULTIPLIER=1.6
DEPTH_GAMMA=0.91

python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/$EXP_NAME-$LOSS.$OPTIMIZER.$LR.$LR_DECAY.$LR_DECAY_STEP-$REGULARIZER.$REG_LAMBDA \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=$OPTIMIZER --loss=$LOSS --learning_rate_decay_type=exponential --learning_rate=$LR \
			--learning_rate_decay_factor=$LR_DECAY --learning_rate_decay_step=$LR_DECAY_STEP \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER --depth_gamma=$DEPTH_GAMMA \
			--use_batch_norm=True --regularizer=$REGULARIZER --regularizer_lambda=$REG_LAMBDA \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE


EXP_NAME="x103"
DEPTH_MULTIPLIER=5.0
DEPTH_GAMMA=0.75

python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/$EXP_NAME-$LOSS.$OPTIMIZER.$LR.$LR_DECAY.$LR_DECAY_STEP-$REGULARIZER.$REG_LAMBDA \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=$OPTIMIZER --loss=$LOSS --learning_rate_decay_type=exponential --learning_rate=$LR \
			--learning_rate_decay_factor=$LR_DECAY --learning_rate_decay_step=$LR_DECAY_STEP \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER --depth_gamma=$DEPTH_GAMMA \
			--use_batch_norm=True --regularizer=$REGULARIZER --regularizer_lambda=$REG_LAMBDA \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE


EXP_NAME="x104"
DEPTH_MULTIPLIER=0.5
DEPTH_GAMMA=1.16

python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/$EXP_NAME-$LOSS.$OPTIMIZER.$LR.$LR_DECAY.$LR_DECAY_STEP-$REGULARIZER.$REG_LAMBDA \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=$OPTIMIZER --loss=$LOSS --learning_rate_decay_type=exponential --learning_rate=$LR \
			--learning_rate_decay_factor=$LR_DECAY --learning_rate_decay_step=$LR_DECAY_STEP \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER --depth_gamma=$DEPTH_GAMMA \
			--use_batch_norm=True --regularizer=$REGULARIZER --regularizer_lambda=$REG_LAMBDA \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE


EXP_NAME="x105"
DEPTH_MULTIPLIER=10.0
DEPTH_GAMMA=0.675

python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/$EXP_NAME-$LOSS.$OPTIMIZER.$LR.$LR_DECAY.$LR_DECAY_STEP-$REGULARIZER.$REG_LAMBDA \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=$OPTIMIZER --loss=$LOSS --learning_rate_decay_type=exponential --learning_rate=$LR \
			--learning_rate_decay_factor=$LR_DECAY --learning_rate_decay_step=$LR_DECAY_STEP \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER --depth_gamma=$DEPTH_GAMMA \
			--use_batch_norm=True --regularizer=$REGULARIZER --regularizer_lambda=$REG_LAMBDA \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE


EXP_NAME="x106"
DEPTH_MULTIPLIER=2.0
DEPTH_GAMMA=1.0

python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/$EXP_NAME-$LOSS.$OPTIMIZER.$LR.$LR_DECAY.$LR_DECAY_STEP-$REGULARIZER.$REG_LAMBDA \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=$OPTIMIZER --loss=$LOSS --learning_rate_decay_type=exponential --learning_rate=$LR \
			--learning_rate_decay_factor=$LR_DECAY --learning_rate_decay_step=$LR_DECAY_STEP \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER --depth_gamma=$DEPTH_GAMMA \
			--use_batch_norm=True --regularizer=$REGULARIZER --regularizer_lambda=$REG_LAMBDA \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE


EXP_NAME="x107"
DEPTH_MULTIPLIER=0.7
DEPTH_GAMMA=1.225

python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/$EXP_NAME-$LOSS.$OPTIMIZER.$LR.$LR_DECAY.$LR_DECAY_STEP-$REGULARIZER.$REG_LAMBDA \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=$OPTIMIZER --loss=$LOSS --learning_rate_decay_type=exponential --learning_rate=$LR \
			--learning_rate_decay_factor=$LR_DECAY --learning_rate_decay_step=$LR_DECAY_STEP \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER --depth_gamma=$DEPTH_GAMMA \
			--use_batch_norm=True --regularizer=$REGULARIZER --regularizer_lambda=$REG_LAMBDA \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE


EXP_NAME="x108"
DEPTH_MULTIPLIER=10.0
DEPTH_GAMMA=0.775

python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/$EXP_NAME-$LOSS.$OPTIMIZER.$LR.$LR_DECAY.$LR_DECAY_STEP-$REGULARIZER.$REG_LAMBDA \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=$OPTIMIZER --loss=$LOSS --learning_rate_decay_type=exponential --learning_rate=$LR \
			--learning_rate_decay_factor=$LR_DECAY --learning_rate_decay_step=$LR_DECAY_STEP \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER --depth_gamma=$DEPTH_GAMMA \
			--use_batch_norm=True --regularizer=$REGULARIZER --regularizer_lambda=$REG_LAMBDA \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE


EXP_NAME="x109"
DEPTH_MULTIPLIER=40.0
DEPTH_GAMMA=0.645

python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/$EXP_NAME-$LOSS.$OPTIMIZER.$LR.$LR_DECAY.$LR_DECAY_STEP-$REGULARIZER.$REG_LAMBDA \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=$OPTIMIZER --loss=$LOSS --learning_rate_decay_type=exponential --learning_rate=$LR \
			--learning_rate_decay_factor=$LR_DECAY --learning_rate_decay_step=$LR_DECAY_STEP \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER --depth_gamma=$DEPTH_GAMMA \
			--use_batch_norm=True --regularizer=$REGULARIZER --regularizer_lambda=$REG_LAMBDA \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE


EXP_NAME="x110"
DEPTH_MULTIPLIER=440.0
DEPTH_GAMMA=0.5

python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/$EXP_NAME-$LOSS.$OPTIMIZER.$LR.$LR_DECAY.$LR_DECAY_STEP-$REGULARIZER.$REG_LAMBDA \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=$OPTIMIZER --loss=$LOSS --learning_rate_decay_type=exponential --learning_rate=$LR \
			--learning_rate_decay_factor=$LR_DECAY --learning_rate_decay_step=$LR_DECAY_STEP \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER --depth_gamma=$DEPTH_GAMMA \
			--use_batch_norm=True --regularizer=$REGULARIZER --regularizer_lambda=$REG_LAMBDA \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE