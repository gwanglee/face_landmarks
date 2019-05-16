TFR_PATH="/youjin/face_landmark/data"
TRAIN_PATH="/youjin/face_landmark/train"
TRAIN_NAME="0402"

TRAIN="160v5.0402.train.tfrecord"
VAL="160v5.0402.val.tfrecord"

TRAIN_E="160v5.0402.ext.train.tfrecord"
VAL_E="160v5.0402.ext.val.tfrecord"

TRAIN_C="160v5.0402.cen.train.tfrecord"
VAL_C="160v5.0402.cen.val.tfrecord"

TRAIN_CE="160v5.0402.cen.ext.train.tfrecord"
VAL_CE="160v5.0402.cen.ext.val.tfrecord"

TRAIN=$TRAIN_CE
VAL=$VAL_CE

MAX_STEP=380000
QUANT_STEP=320000

BATCH_SIZE=32
DEPTH_MULTIPLIER=2

# 001, 002, 003: loss comparison [wing, l1]
python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/x001_wing_sgd_0.003_lrd_0.5_72k_bn_l2_0.005 \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=sgd --loss=wing --learning_rate_decay_type=exponential --learning_rate=0.003 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=72000 \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--use_batch_norm=True --regularizer=l2 --regularizer_lambda=0.005 \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE

python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/x002_l1_sgd_0.003_lrd_0.5_72k_bn_l2_0.005 \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=sgd --loss=l1 --learning_rate_decay_type=exponential --learning_rate=0.003 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=72000 \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--use_batch_norm=True --regularizer=l2 --regularizer_lambda=0.005 \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE

# 001, 003, 004: loss decay : decay_factor=[0.5, 0.1, 0.3], step=[72k, 100k, 80k]
python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/x003_wing_sgd_0.003_lrd_0.1_100k_bn_l2_0.005 \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=sgd --loss=wing --learning_rate_decay_type=exponential --learning_rate=0.003 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=100000 \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--use_batch_norm=True --regularizer=l2 --regularizer_lambda=0.005 \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE

python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/x004_wing_sgd_0.003_lrd_0.3_80k_bn_l2_0.005 \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=sgd --loss=wing --learning_rate_decay_type=exponential --learning_rate=0.003 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=80000 \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--use_batch_norm=True --regularizer=l2 --regularizer_lambda=0.005 \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE

# 001, 005, 006: no regularizer, regularization w/ 0.01
python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/x005_wing_sgd_0.003_lrd_0.5_72k_bn_l2_0.01 \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=sgd --loss=wing --learning_rate_decay_type=exponential --learning_rate=0.003 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=72000 \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--use_batch_norm=True --regularizer=l2 --regularizer_lambda=0.01 \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE

python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/x006_wing_sgd_0.003_lrd_0.5_72k_bn_l2_0.005 \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=sgd --loss=wing --learning_rate_decay_type=exponential --learning_rate=0.003 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=72000 \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--use_batch_norm=True \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE

# 006, 007: no batch_norm
python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/x007_wing_sgd_0.003_lrd_0.5_72k_bn_l2_0.005 \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=sgd --loss=wing --learning_rate_decay_type=exponential --learning_rate=0.003 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=72000 \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE


# 001, 008, 009: optimizer [sgd, adam, momentum]
python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/x009_wing_adam_0.003_lrd_0.5_72k_bn_l2_0.005 \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=adam --loss=wing --learning_rate_decay_type=exponential --learning_rate=0.003 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=72000 \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--use_batch_norm=True --regularizer=l2 --regularizer_lambda=0.005 \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE

python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/x010_wing_momentum_0.003_lrd_0.5_72k_bn_l2_0.005 \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=momentum --loss=wing --learning_rate_decay_type=exponential --learning_rate=0.003 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=72000 \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--use_batch_norm=True --regularizer=l2 --regularizer_lambda=0.005 \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE

