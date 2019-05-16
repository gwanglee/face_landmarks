#!/usr/bin/env bash
TFR_PATH="/home/gglee/Data/Landmark"
TRAIN_PATH="/home/gglee/Data/Landmark/train"
TRAIN_NAME="0408"

TRAIN="0407.ext.train.tfrecord"
VAL="0407.ext.val.tfrecord"

MAX_STEP=1300000

LR_1=0.1
LR_2=0.005
LR_3=0.001

LR_DECAY_STEP_1=100000
LR_DECAY_STEP_2=200000
LR_DECAY_STEP_3=300000

LR_DECAY_FACTOR_1=0.1
LR_DECAY_FACTOR_2=0.3
LR_DECAY_FACTOR_3=0.5

QUANT_STEP=1000000

BATCH_SIZE=32
DEPTH_MULTIPLIER=2

REG_L1=l1
REG_L2=l2
REG_LAMBDA_1=0.001
REG_LAMBDA_2=0.005
REG_LAMBDA_3=0.01

# common: loss=l1, optimizer=sgd, max_iter=1.1M, depth_multiplier=2

EX_NAME="x1"
LOSS="chain"
LR=$LR_1
LRD_FACTOR=$LR_DECAY_FACTOR_1
LRD_STEP=$LR_DECAY_STEP_3
REG=$REG_L2
REG_LAMBDA=$REG_LAMBDA_2
BATCH_NORM=True
python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/$EX_NAME \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--loss=$LOSS \
			--optimizer=sgd --learning_rate_decay_type=exponential --learning_rate=$LR \
			--learning_rate_decay_factor=$LRD_FACTOR --learning_rate_decay_step=$LRD_STEP \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--use_batch_norm=$BATCH_NORM --regularizer=$REG --regularizer_lambda=$REG_LAMBDA \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE

EX_NAME="x2"
LOSS="chain"
LR=$LR_1
LRD_FACTOR=$LR_DECAY_FACTOR_2
LRD_STEP=$LR_DECAY_STEP_3
REG=$REG_L2
REG_LAMBDA=$REG_LAMBDA_2
BATCH_NORM=True
python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/$EX_NAME \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--loss=$LOSS \
			--optimizer=sgd --learning_rate_decay_type=exponential --learning_rate=$LR \
			--learning_rate_decay_factor=$LRD_FACTOR --learning_rate_decay_step=$LRD_STEP \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--use_batch_norm=$BATCH_NORM --regularizer=$REG --regularizer_lambda=$REG_LAMBDA \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE
EX_NAME="x3"
LOSS="chain"
LR=$LR_1
LRD_FACTOR=$LR_DECAY_FACTOR_3
LRD_STEP=$LR_DECAY_STEP_3
REG=$REG_L2
REG_LAMBDA=$REG_LAMBDA_2
BATCH_NORM=True
python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/$EX_NAME \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--loss=$LOSS \
			--optimizer=sgd --learning_rate_decay_type=exponential --learning_rate=$LR \
			--learning_rate_decay_factor=$LRD_FACTOR --learning_rate_decay_step=$LRD_STEP \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--use_batch_norm=$BATCH_NORM --regularizer=$REG --regularizer_lambda=$REG_LAMBDA \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE

python evaluate.py --tfrecord=$TFR_PATH/$VAL --models_dir=$TFR_PATH/$TRAIN_NAME
