TFR_PATH="/home/gglee/Data/Landmark"
TRAIN_PATH="/home/gglee/Data/Landmark/train"
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

MAX_STEP=480000
QUANT_STEP=-1

BATCH_SIZE=32
DEPTH_MULTIPLIER=2

# 001, 002, 003, 008, 009: loss comparison [l1, wing, l2, pointwise_l2, euc_wing]

python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/x009_ewing_sgd_0.004_lrd_0.5_80k_bn_l2_0.001 \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=sgd --loss=euc_wing --learning_rate_decay_type=exponential --learning_rate=0.004 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=80000 \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--use_batch_norm=True --regularizer=l2 --regularizer_lambda=0.001 \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE

python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/x000_pl2_sgd_0.003_lrd_0.5_64k_bn_l2_0.005 \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=sgd --loss=pointwise_l2 --learning_rate_decay_type=exponential --learning_rate=0.003 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=64000 \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--regularizer=l2 --regularizer_lambda=0.005 \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE

python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/x002_wing_sgd_0.003_lrd_0.5_64k_bn_l2_0.005 \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=sgd --loss=wing --learning_rate_decay_type=exponential --learning_rate=0.003 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=64000 \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--regularizer=l2 --regularizer_lambda=0.005 \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE

python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/x001_l1_sgd_0.003_lrd_0.5_64k_bn_l2_0.005 \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=sgd --loss=l1 --learning_rate_decay_type=exponential --learning_rate=0.003 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=64000 \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--regularizer=l2 --regularizer_lambda=0.005 \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE

python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/x003_l2_sgd_0.003_lrd_0.5_64k_bn_l2_0.005 \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=sgd --loss=l2 --learning_rate_decay_type=exponential --learning_rate=0.003 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=64000 \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--regularizer=l2 --regularizer_lambda=0.005 \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE

# 001, 004: loss decay : decay_factor=0.5->0.1, step=64k -> 96k
python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/x004_wing_sgd_0.003_lrd_0.1_96k_bn_l2_0.005 \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=sgd --loss=wing --learning_rate_decay_type=exponential --learning_rate=0.003 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=99000 \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--regularizer=l2 --regularizer_lambda=0.005 \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE

# 001, 005: tfrecord [centered+extended, centered, ]
python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/x005_l1_sgd_0.003_lrd_0.5_64k_bn_l2_0.005 \
			--train_tfr=$TFR_PATH/$TRAIN_C --val_tfr=$TFR_PATH/$VAL_C \
			--optimizer=sgd --loss=l1 --learning_rate_decay_type=exponential --learning_rate=0.003 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=64000 \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--regularizer=l2 --regularizer_lambda=0.005 \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE

# 002, 009, 010, 011, 012, 013: w in wing_loss and euc_wing_loss

python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/x010_wing_sgd_0.003_lrd_0.5_64k_bn_l2_0.01 \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=sgd --loss=wing --learning_rate_decay_type=exponential --learning_rate=0.003 \
			--wing_w=5.0 -wing_eps=2.0 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=64000 \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--regularizer=l2 --regularizer_lambda=0.005 \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE

python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/x011_ewing_sgd_0.003_lrd_0.5_64k_bn_l2_0.01 \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=sgd --loss=euc_wing --learning_rate_decay_type=exponential --learning_rate=0.003 \
			--wing_w=5.0 -wing_eps=2.0 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=64000 \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--regularizer=l2 --regularizer_lambda=0.005 \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE

python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/x012_wing_sgd_0.003_lrd_0.5_64k_bn_l2_0.01 \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=sgd --loss=wing --learning_rate_decay_type=exponential --learning_rate=0.003 \
			--wing_w=0.1 -wing_eps=2.0 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=64000 \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--regularizer=l2 --regularizer_lambda=0.005 \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE

python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/x013_ewing_sgd_0.003_lrd_0.5_64k_bn_l2_0.01 \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=sgd --loss=euc_wing --learning_rate_decay_type=exponential --learning_rate=0.003 \
			--wing_w=0.1 -wing_eps=2.0 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=64000 \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--regularizer=l2 --regularizer_lambda=0.005 \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE
