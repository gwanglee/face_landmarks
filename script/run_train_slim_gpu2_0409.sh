TFR_PATH="/youjin/face_landmark/data"
TRAIN_PATH="/youjin/face_landmark/train"
TRAIN_NAME="0409_gpu2"

TRAIN="0407.ext.train.tfrecord"
VAL="0407.ext.val.tfrecord"

MAX_STEP=700000
LR_DECAY_STEP_1=200000
LR_DECAY_STEP_2=100000

QUANT_STEP=660000

BATCH_SIZE=32
DEPTH_MULTIPLIER=2

LR_1=0.01
LR_2=0.02
LR_3=0.005

LR_DECAY_1=0.1
LR_DECAY_2=0.25
LR_DECAY_3=0.5

LOSS_L1="l1"
LOSS_L2="pointwise_l2"
LOSS_WING="wing"

OPTIMIZER="sgd"

REGULARIZER_L1="l1"
REGULARIZER_L2="l2"

REG_LAMBDA_1=0.001
REG_LAMBDA_2=0.005
REG_LAMBDA_3=0.01


EXP_NAME="x106"
LOSS=$LOSS_L1
LR=$LR_2
LR_DECAY=$LR_DECAY_3
LR_DECAY_STEP=$LR_DECAY_STEP_1
REGULARIZER=$REGULARIZER_L2
REG_LAMBDA=$REG_LAMBDA_2

python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/$EXP_NAME-$LOSS.$OPTIMIZER.$LR.$LR_DECAY.$LR_DECAY_STEP-$REGULARIZER.$REG_LAMBDA \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=$OPTIMIZER --loss=$LOSS --learning_rate_decay_type=exponential --learning_rate=$LR \
			--learning_rate_decay_factor=$LR_DECAY --learning_rate_decay_step=$LR_DECAY_STEP \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--use_batch_norm=True --regularizer=$REGULARIZER --regularizer_lambda=$REG_LAMBDA \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE

EXP_NAME="x107"
LOSS=$LOSS_L1
LR=$LR_3
LR_DECAY=$LR_DECAY_1
LR_DECAY_STEP=$LR_DECAY_STEP_1
REGULARIZER=$REGULARIZER_L2
REG_LAMBDA=$REG_LAMBDA_2

python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/$EXP_NAME-$LOSS.$OPTIMIZER.$LR.$LR_DECAY.$LR_DECAY_STEP-$REGULARIZER.$REG_LAMBDA \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=$OPTIMIZER --loss=$LOSS --learning_rate_decay_type=exponential --learning_rate=$LR \
			--learning_rate_decay_factor=$LR_DECAY --learning_rate_decay_step=$LR_DECAY_STEP \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--use_batch_norm=True --regularizer=$REGULARIZER --regularizer_lambda=$REG_LAMBDA \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE

EXP_NAME="x108"
LOSS=$LOSS_L1
LR=$LR_3
LR_DECAY=$LR_DECAY_2
LR_DECAY_STEP=$LR_DECAY_STEP_1
REGULARIZER=$REGULARIZER_L2
REG_LAMBDA=$REG_LAMBDA_2

python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/$EXP_NAME-$LOSS.$OPTIMIZER.$LR.$LR_DECAY.$LR_DECAY_STEP-$REGULARIZER.$REG_LAMBDA \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=$OPTIMIZER --loss=$LOSS --learning_rate_decay_type=exponential --learning_rate=$LR \
			--learning_rate_decay_factor=$LR_DECAY --learning_rate_decay_step=$LR_DECAY_STEP \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--use_batch_norm=True --regularizer=$REGULARIZER --regularizer_lambda=$REG_LAMBDA \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE

EXP_NAME="x109"
LOSS=$LOSS_L1
LR=$LR_3
LR_DECAY=$LR_DECAY_3
LR_DECAY_STEP=$LR_DECAY_STEP_1
REGULARIZER=$REGULARIZER_L2
REG_LAMBDA=$REG_LAMBDA_2

python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/$EXP_NAME-$LOSS.$OPTIMIZER.$LR.$LR_DECAY.$LR_DECAY_STEP-$REGULARIZER.$REG_LAMBDA \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=$OPTIMIZER --loss=$LOSS --learning_rate_decay_type=exponential --learning_rate=$LR \
			--learning_rate_decay_factor=$LR_DECAY --learning_rate_decay_step=$LR_DECAY_STEP \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--use_batch_norm=True --regularizer=$REGULARIZER --regularizer_lambda=$REG_LAMBDA \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE

EXP_NAME="x110"
LOSS=$LOSS_L1
LR=$LR_3
LR_DECAY=0.3
LR_DECAY_STEP=$LR_DECAY_STEP_2
REGULARIZER=$REGULARIZER_L2
REG_LAMBDA=$REG_LAMBDA_2

python train_slim.py --train_dir=$TRAIN_PATH/$TRAIN_NAME/$EXP_NAME-$LOSS.$OPTIMIZER.$LR.$LR_DECAY.$LR_DECAY_STEP-$REGULARIZER.$REG_LAMBDA \
			--train_tfr=$TFR_PATH/$TRAIN --val_tfr=$TFR_PATH/$VAL \
			--optimizer=$OPTIMIZER --loss=$LOSS --learning_rate_decay_type=exponential --learning_rate=$LR \
			--learning_rate_decay_factor=$LR_DECAY --learning_rate_decay_step=$LR_DECAY_STEP \
			--max_number_of_steps=$MAX_STEP --depth_multiplier=$DEPTH_MULTIPLIER \
			--use_batch_norm=True --regularizer=$REGULARIZER --regularizer_lambda=$REG_LAMBDA \
			--quantize_delay=$QUANT_STEP --batch_size=$BATCH_SIZE

python evaluate.py --tfrecord=$TFR_PATH/$VAL --models_dir=$TRAIN_PATH/$TRAIN_NAME
