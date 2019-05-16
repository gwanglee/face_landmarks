python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0401/x001_l1_sgd_0.003_lrd_0.5_72k_bn_l2_0.01 \
			--optimizer=sgd --loss=l1 --learning_rate_decay_type=exponential --learning_rate=0.003 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=72000 \
			--max_number_of_steps=360000 --depth_multiplier=2 \
			--use_batch_norm=True --regularizer=l2 --regularizer_lambda=0.005 \
			--quantize_delay=320000 --batch_size=32

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0401/x002_wing_sgd_0.003_lrd_0.5_72k_bn_l2_0.01 \
			--optimizer=sgd --loss=wing --learning_rate_decay_type=exponential --learning_rate=0.003 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=72000 \
			--max_number_of_steps=360000 --depth_multiplier=2 \
			--use_batch_norm=True --regularizer=l2 --regularizer_lambda=0.005 \
			--quantize_delay=320000 --batch_size=32

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0401/x003_l2_sgd_0.003_lrd_0.5_72k_bn_l2_0.01 \
			--optimizer=sgd --loss=l2 --learning_rate_decay_type=exponential --learning_rate=0.003 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=72000 \
			--max_number_of_steps=360000 --depth_multiplier=2 \
			--use_batch_norm=True --regularizer=l2 --regularizer_lambda=0.005 \
			--quantize_delay=320000 --batch_size=32

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0401/x004_wing_sgd_0.003_lrd_0.8_20k_bn_l2_0.01 \
			--optimizer=sgd --loss=wing --learning_rate_decay_type=exponential --learning_rate=0.003 \
			--learning_rate_decay_factor=0.8 --learning_rate_decay_step=20000 \
			--max_number_of_steps=360000 --depth_multiplier=2 \
			--use_batch_norm=True --regularizer=l2 --regularizer_lambda=0.005 \
			--quantize_delay=320000 --batch_size=32

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0401/x005_momentum_0.003_0.95_wing_l2_bn \
			--optimizer=momentum --loss=wing \
			--learning_rate_decay_type=exponential --learning_rate=0.003 --momentum=0.95 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=48000 \
			--max_number_of_steps=360000 --depth_multiplier=2 \
			--regularizer=l2 --use_batch_norm=True \
			--quantize_delay=320000 --batch_size=32

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0401/x006_sgd_0.004_lrd_0.1_100k_bn_l2 \
			--optimizer=sgd --loss=wing \
			--learning_rate_decay_type=exponential --learning_rate=0.004 \
			--learning_rate_decay_factor=0.1 --learning_rate_decay_step=100000 \
			--max_number_of_steps=360000 --depth_multiplier=2 \
			--regularizer=l2  --regularizer_lambda=0.005 --use_batch_norm=True \
			--quantize_delay=320000 --batch_size=32

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0401/x007_sgd_0.01_lrd_0.6_45k_bn_l2 \
			--optimizer=sgd --loss=wing \
			--learning_rate_decay_type=exponential --learning_rate=0.01 \
			--learning_rate_decay_factor=0.1 --learning_rate_decay_step=45000 \
			--max_number_of_steps=360000 --depth_multiplier=2 \
			--regularizer=l2 --regularizer_lambda=0.005 --use_batch_norm=True \
			--quantize_delay=320000 --batch_size=32

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0401/x008_momentum_0.003_l2 \
			--optimizer=momentum \
			--learning_rate_decay_type=exponential --learning_rate=0.003 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=55000 \
			--max_number_of_steps=480000 --depth_multiplier=2 \
			--regularizer=l2 \
			--quantize_delay=400000 --batch_size=32

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0401/x009_momentum_0.003_lrd_0.5_68k_l2 \
			--optimizer=momentum --loss=wing \
			--learning_rate_decay_type=exponential --learning_rate=0.003 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=68000 \
			--max_number_of_steps=360000 --depth_multiplier=2 \
			--regularizer=l2 --regularizer_lambda=0.01 \
			--quantize_delay=300000 --batch_size=32

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0401/x010_momentum_0.003_0.99_l2 \
			--optimizer=momentum --loss=wing \
			--learning_rate_decay_type=exponential --learning_rate=0.003 --momentum=0.99 \
			--learning_rate_decay_factor=0.1 --learning_rate_decay_step=100000 \
			--max_number_of_steps=360000 --depth_multiplier=2 \
			--regularizer=l2 \
			--quantize_delay=300000 --batch_size=32

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0401/x011_momentum_0.003_0.999_l2_bn \
			--optimizer=momentum --loss=wing \
			--learning_rate_decay_type=exponential --learning_rate=0.003 --momentum=0.999 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=48000 \
			--max_number_of_steps=360000 --depth_multiplier=2 \
			--regularizer=l2 --use_batch_norm=True \
			--quantize_delay=300000 --batch_size=32

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0401/x14_adam_0.004_0.9_0.999 \
			--optimizer=adam --loss=wing \
			--learning_rate_decay_type=fixed --learning_rate=0.004 \
			--max_number_of_steps=360000 --depth_multiplier=2 \
			--adam_beta1=0.9 --adam_beta2=0.999 \
			--regularizer=l2 --regularizer_lambda=0.005 \
			--quantize_delay=300000 --batch_size=32

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0401/x15_adam_0.001_lrd_0.1_54k \
			--optimizer=adam --loss=wing \
			--learning_rate_decay_type=exponential --learning_rate=0.004 \
			--learning_rate_decay_factor=0.1 --learning_rate_decay_step=100000 \
			--max_number_of_steps=360000 --depth_multiplier=2 \
			--regularizer=l2 --regularizer_lambda=0.005 \
			--quantize_delay=300000 --batch_size=32
