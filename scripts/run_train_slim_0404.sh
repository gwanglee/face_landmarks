#!/usr/bin/env bash
TFR_PATH="/home/gglee/Data/Landmark"
TRAIN_PATH="/home/gglee/Data/Landmark/train"
TRAIN_NAME="0404"

TRAIN="160v5.0402.train.tfrecord"
VAL="160v5.0402.val.tfrecord"

TRAIN_G="160v5.0402.ext.gray.train.tfrecord"
VAL_G="160v5.0402.ext.gray.val.tfrecord"

TRAIN_E="160v5.0402.ext.train.tfrecord"
VAL_E="160v5.0402.ext.val.tfrecord"

TRAIN_C="160v5.0402.cen.train.tfrecord"
VAL_C="160v5.0402.cen.val.tfrecord"

TRAIN_CE="160v5.0402.cen.ext.train.tfrecord"
VAL_CE="160v5.0402.cen.ext.val.tfrecord"

TRAIN=$TRAIN_E
VAL=$VAL_E

MAX_STEP=1100000

LR_1=0.005
LR_2=0.001
LR_3=0.1

LR_DECAY_STEP_1=100000
LR_DECAY_STEP_2=200000
LR_DECAY_STEP_3=300000

LR_DECAY_FACTOR_1=0.5
LR_DECAY_FACTOR_2=0.2

QUANT_STEP=1000000

BATCH_SIZE=32
DEPTH_MULTIPLIER=2

REG_L1=l2
REG_L2=l1
REG_LAMBDA_1=0.001
REG_LAMBDA_2=0.005
REG_LAMBDA_3=0.01

# common: loss=l1, optimizer=sgd, max_iter=1.1M, depth_multiplier=4

EX_NAME="x1"
VAL_LR=$LR_1				# 0.005
VAL_LRD_STEP=$LR_DECAY_STEP_2		# 200K
VAL_LRD_FACTOR=$LR_DECAY_FACTOR_1	# 0.5
VAL_REG=$REG_L2				# l2
VAL_REG_LAMBDA=$REG_LAMBDA_2		# 0.005
VAL_BATCH_NORM=True
python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/$EX_NAME \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=sgd --loss=l1 --learning_rate_decay_type=exponential --learning_rate=$VAL_LR \
			--learning_rate_decay_factor=$VAL_LRD_FACTOR --learning_rate_decay_step=$VAL_LRD_STEP \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--use_batch_norm=$VAL_BATCH_NORM --regularizer=$VAL_REG --regularizer_lambda=$VAL_REG_LAMBDA \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE


EX_NAME=x2
VAL_LR=$LR_1				# 0.005
VAL_LRD_STEP=$LR_DECAY_STEP_3		# 300K
VAL_LRD_FACTOR=$LR_DECAY_FACTOR_2	# 0.2
VAL_REG=$REG_L2				# l2
VAL_REG_LAMBDA=$REG_LAMBDA_2		# 0.005
VAL_BATCH_NORM=True
python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/$EX_NAME \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=sgd --loss=l1 --learning_rate_decay_type=exponential --learning_rate=$VAL_LR \
			--learning_rate_decay_factor=$VAL_LRD_FACTOR --learning_rate_decay_step=$VAL_LRD_STEP \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--use_batch_norm=$VAL_BATCH_NORM --regularizer=$VAL_REG --regularizer_lambda=$VAL_REG_LAMBDA \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE

EX_NAME=x3
VAL_LR=$LR_3				# 0.01
VAL_LRD_STEP=$LR_DECAY_STEP_2		# 200K
VAL_LRD_FACTOR=$LR_DECAY_FACTOR_2	# 0.2
VAL_REG=$REG_L2				# l2
VAL_REG_LAMBDA=$REG_LAMBDA_2		# 0.005
VAL_BATCH_NORM=True
python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/$EX_NAME \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=sgd --loss=l1 --learning_rate_decay_type=exponential --learning_rate=$VAL_LR \
			--learning_rate_decay_factor=$VAL_LRD_FACTOR --learning_rate_decay_step=$VAL_LRD_STEP \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--use_batch_norm=$VAL_BATCH_NORM --regularizer=$VAL_REG --regularizer_lambda=$VAL_REG_LAMBDA \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE

EX_NAME=x4
VAL_LR=$LR_1				# 0.005
VAL_LRD_STEP=$LR_DECAY_STEP_2		# 200K
VAL_LRD_FACTOR=$LR_DECAY_FACTOR_1	# 0.5
VAL_REG=$REG_L2				# l2
VAL_REG_LAMBDA=$REG_LAMBDA_2		# 0.005
VAL_BATCH_NORM=False
python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/$EX_NAME \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=sgd --loss=l1 --learning_rate_decay_type=exponential --learning_rate=$VAL_LR \
			--learning_rate_decay_factor=$VAL_LRD_FACTOR --learning_rate_decay_step=$VAL_LRD_STEP \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--use_batch_norm=$VAL_BATCH_NORM --regularizer=$VAL_REG --regularizer_lambda=$VAL_REG_LAMBDA \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE

EX_NAME=x5
VAL_LR=$LR_1				# 0.005
VAL_LRD_STEP=$LR_DECAY_STEP_2		# 200K
VAL_LRD_FACTOR=$LR_DECAY_FACTOR_1	# 0.5
VAL_REG=$REG_L1				# l2
VAL_REG_LAMBDA=$REG_LAMBDA_2		# 0.005
VAL_BATCH_NORM=True
python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/$EX_NAME \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=sgd --loss=l1 --learning_rate_decay_type=exponential --learning_rate=$VAL_LR \
			--learning_rate_decay_factor=$VAL_LRD_FACTOR --learning_rate_decay_step=$VAL_LRD_STEP \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--use_batch_norm=$VAL_BATCH_NORM --regularizer=$VAL_REG --regularizer_lambda=$VAL_REG_LAMBDA \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE

EX_NAME=x6
VAL_LR=$LR_1				# 0.005
VAL_LRD_STEP=$LR_DECAY_STEP_2		# 200K
VAL_LRD_FACTOR=$LR_DECAY_FACTOR_1	# 0.5
VAL_REG=None				# l2
VAL_REG_LAMBDA=$REG_LAMBDA_2		# 0.005
VAL_BATCH_NORM=True
python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/$EX_NAME \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=sgd --loss=l1 --learning_rate_decay_type=exponential --learning_rate=$VAL_LR \
			--learning_rate_decay_factor=$VAL_LRD_FACTOR --learning_rate_decay_step=$VAL_LRD_STEP \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--use_batch_norm=$VAL_BATCH_NORM --regularizer=$VAL_REG --regularizer_lambda=$VAL_REG_LAMBDA \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE

python evaluate.py --tfrecord=$TFR_PATH/$VAL --models_dir=$TFR_PATH/$TRAIN_NAME
