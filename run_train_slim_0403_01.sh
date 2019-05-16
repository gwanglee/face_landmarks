#!/usr/bin/env bash
TFR_PATH="/home/gglee/Data/Landmark"
TRAIN_PATH="/home/gglee/Data/Landmark/train"
TRAIN_NAME="0403_01"

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

MAX_STEP=460000
MAX_STEP_2=900000

LR_DECAY_STEP_1=100000
LR_DECAY_STEP_2=200000

QUANT_STEP=-1

BATCH_SIZE=32
DEPTH_MULTIPLIER=2

# loss: l1, pl2, ewing
xpython train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/x2_ewing_sgd_0.004_lrd_0.5_100k_bn_l2_0.002 \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=sgd --loss=euc_wing --learning_rate_decay_type=exponential --learning_rate=0.004 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=100000 \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--use_batch_norm=True --regularizer=l2 --regularizer_lambda=0.002 \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE

xpython train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/x3_l1_sgd_0.004_lrd_0.1_100k_bn_l2_0.002 \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=sgd --loss=l1 --learning_rate_decay_type=exponential --learning_rate=0.004 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=100000 \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--use_batch_norm=True --regularizer=l2 --regularizer_lambda=0.002 \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE

xpython train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/x1_pl2_sgd_0.004_lrd_0.5_100k_bn_l2_0.002 \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=sgd --loss=pointwise_l2 --learning_rate_decay_type=exponential --learning_rate=0.004 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=100000 \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--use_batch_norm=True --regularizer=l2 --regularizer_lambda=0.002 \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE

# gray input
xpython train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/x4_l1_sgd_0.004_lrd_0.1_100k_bn_l2_0.002_gray \
			--train_tfr=$TFR_PATH/$TRAIN_G --val_tfr=$TFR_PATH/$VAL_G \
			--optimizer=sgd --loss=l1 --learning_rate_decay_type=exponential --learning_rate=0.004 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=$LR_DECAY_STEP_1 \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--use_batch_norm=True --regularizer=l2 --regularizer_lambda=0.002 \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE --is_gray=True


# very long
xpython train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/x5_l1_sgd_0.004_lrd_0.1_100k_bn_l2_0.002_gray \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=sgd --loss=l1 --learning_rate_decay_type=exponential --learning_rate=0.004 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=$LR_DECAY_STEP_1 \
			--max_number_of_steps=$MAX_STEP_2 --depth_multiplier=$DEPTH_MULTIPLIER \
			--use_batch_norm=True --regularizer=l2 --regularizer_lambda=0.002 \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE --is_gray=True

xpython train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/x6_l1_sgd_0.004_lrd_0.1_200k_bn_l2_0.002 \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=sgd --loss=l1 --learning_rate_decay_type=exponential --learning_rate=0.004 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=$LR_DECAY_STEP_2 \
			--max_number_of_steps=$MAX_STEP_2 --depth_multiplier=$DEPTH_MULTIPLIER \
			--use_batch_norm=True --regularizer=l2 --regularizer_lambda=0.002 \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE --is_gray=False

python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/x7_pl2_sgd_0.004_lrd_0.5_200k_bn_l2_0.002 \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=sgd --loss=pointwise_l2 --learning_rate_decay_type=exponential --learning_rate=0.004 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=$LR_DECAY_STEP_2 \
			--max_number_of_steps=$MAX_STEP_2 --depth_multiplier=$DEPTH_MULTIPLIER \
			--use_batch_norm=True --regularizer=l2 --regularizer_lambda=0.002 \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE
