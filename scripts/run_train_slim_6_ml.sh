python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0328/x0_sgd_0.003_lrd_0.5_64k_bn_l2 \
			--optimizer=sgd \
			--learning_rate_decay_type=exponential --learning_rate=0.003 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=64000 \
			--max_number_of_steps=240000 --depth_multiplier=2 \
			--use_batch_norm=True --regularizer=l2 \
			--quantize_delay=160000

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0328/x001_sgd_0.003_lrd_0.5_64k_bn_l2_0.001 \
			--optimizer=sgd \
			--learning_rate_decay_type=exponential --learning_rate=0.003 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=64000 \
			--max_number_of_steps=240000 --depth_multiplier=2 \
			--use_batch_norm=True --regularizer=l2 --regularizer_lambda=0.001 \
			--quantize_delay=160000

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0328/x002_sgd_0.003_lrd_0.5_64k_bn_l2_0.01 \
			--optimizer=sgd \
			--learning_rate_decay_type=exponential --learning_rate=0.003 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=64000 \
			--max_number_of_steps=240000 --depth_multiplier=2 \
			--use_batch_norm=True --regularizer=l2 --regularizer_lambda=0.01 \
			--quantize_delay=160000

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0328/x101_sgd_0.001_lrd_0.5_49k_l2_0.01 \
			--optimizer=sgd \
			--learning_rate_decay_type=exponential --learning_rate=0.001 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=49000 \
			--max_number_of_steps=240000 --depth_multiplier=2 \
			--regularizer=l2 --regularizer_lambda=0.01 \
			--quantize_delay=160000

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0328/x1_sgd_0.001_lrd_0.5_49k_l2 \
			--optimizer=sgd \
			--learning_rate_decay_type=exponential --learning_rate=0.001 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=49000 \
			--max_number_of_steps=240000 --depth_multiplier=2 \
			--regularizer=l2 \
			--quantize_delay=160000

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0328/x2_sgd_0.003_lrd_0.1_66k_l2 \
			--optimizer=sgd \
			--learning_rate_decay_type=exponential --learning_rate=0.003 \
			--learning_rate_decay_factor=0.1 --learning_rate_decay_step=66000 \
			--max_number_of_steps=240000 --depth_multiplier=2 \
			--regularizer=l2 \
			--quantize_delay=160000

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0328/x3_sgd_0.01_lrd_0.1_66k_bn_l2 \
			--optimizer=sgd \
			--learning_rate_decay_type=exponential --learning_rate=0.01 \
			--learning_rate_decay_factor=0.1 --learning_rate_decay_step=66000 \
			--max_number_of_steps=240000 --depth_multiplier=2 \
			--regularizer=l2 --use_batch_norm=True \
			--quantize_delay=160000

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0328/x4_sgd_0.01_lrd_0.1_48k \
			--optimizer=sgd \
			--learning_rate_decay_type=exponential --learning_rate=0.01 \
			--learning_rate_decay_factor=0.1 --learning_rate_decay_step=54000 \
			--max_number_of_steps=240000 --depth_multiplier=2 \
			--quantize_delay=160000

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0328/x5_sgd_0.005_lrd_0.2_48k \
			--optimizer=sgd \
			--learning_rate_decay_type=exponential --learning_rate=0.01 \
			--learning_rate_decay_factor=0.1 --learning_rate_decay_step=54000 \
			--max_number_of_steps=240000 --depth_multiplier=2 \
			--quantize_delay=160000

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0328/x6_sgd_0.0005_lrd_0.2_48k \
			--optimizer=sgd \
			--learning_rate_decay_type=exponential --learning_rate=0.01 \
			--learning_rate_decay_factor=0.1 --learning_rate_decay_step=54000 \
			--max_number_of_steps=240000 --depth_multiplier=2 \
			--quantize_delay=160000

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0328/x7_sgd_0.002_lrd_0.1_68k_l2 \
			--optimizer=sgd \
			--learning_rate_decay_type=exponential --learning_rate=0.002 \
			--learning_rate_decay_factor=0.1 --learning_rate_decay_step=68000 \
			--max_number_of_steps=240000 --depth_multiplier=2 \
			--regularizer=l2 \
			--quantize_delay=160000

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0328/x8_sgd_0.003_lrd_0.5_30k_bn_l2_mvg_0.999 \
			--optimizer=sgd \
			--learning_rate_decay_type=exponential --learning_rate=0.003 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=32000 \
			--max_number_of_steps=240000 --depth_multiplier=2 \
			--use_batch_norm=True --regularizer=l2 \
			--moving_average_decay=0.999 \
			--quantize_delay=160000

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0328/x9_sgd_0.002_lrd_0.1_68k_l2_mvg_0.999 \
			--optimizer=sgd \
			--learning_rate_decay_type=exponential --learning_rate=0.002 \
			--learning_rate_decay_factor=0.1 --learning_rate_decay_step=68000 \
			--max_number_of_steps=240000 --depth_multiplier=2 \
			--regularizer=l2 \
			--moving_average_decay=0.999 \
			--quantize_delay=160000

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0328/x10_sgd_0.01_lrd_0.1_48k_bn_l2_mvg_0.999 \
			--optimizer=sgd \
			--learning_rate_decay_type=exponential --learning_rate=0.01 \
			--learning_rate_decay_factor=0.1 --learning_rate_decay_step=54000 \
			--max_number_of_steps=240000 --depth_multiplier=2 \
			--regularizer=l2 --use_batch_norm=True \
			--moving_average_decay=0.999 \
			--quantize_delay=160000

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0328/x11_adam_0.003_lrd_0.5_32k_bn_l2 \
			--optimizer=adam \
			--learning_rate_decay_type=fixed --learning_rate=0.003 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=32000 \
			--max_number_of_steps=240000 --depth_multiplier=2 \
			--use_batch_norm=True --regularizer=l2 \
			--quantize_delay=160000

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0328/x12_adam_0.003_bn_l2 \
			--optimizer=adam \
			--learning_rate_decay_type=fixed --learning_rate=0.003 \
			--max_number_of_steps=240000 --depth_multiplier=2 \
			--use_batch_norm=True --regularizer=l2 \
			--quantize_delay=160000

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0328/x13_adam_0.003_0.99_0.999_l2 \
			--optimizer=adam \
			--learning_rate_decay_type=fixed --learning_rate=0.003 \
			--max_number_of_steps=240000 --depth_multiplier=2 \
			--regularizer=l2 \
			--adam_beta1=0.99 --adam_beta2=0.999 \
			--quantize_delay=160000

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0328/x14_adam_0.001_0.99_0.999 \
			--optimizer=adam \
			--learning_rate_decay_type=fixed --learning_rate=0.001 \
			--max_number_of_steps=240000 --depth_multiplier=2 \
			--adam_beta1=0.99 --adam_beta2=0.999 \
			--quantize_delay=160000

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0328/x15_adam_0.001_lrd_0.1_54k \
			--optimizer=adam \
			--learning_rate_decay_type=exponential --learning_rate=0.001 \
			--learning_rate_decay_factor=0.1 --learning_rate_decay_step=54000 \
			--max_number_of_steps=240000 --depth_multiplier=2 \
			--quantize_delay=160000

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0328/x16_adam_0.003_bn_l2 \
			--optimizer=adam \
			--learning_rate_decay_type=exponential --learning_rate=0.003 \
			--learning_rate_decay_factor=0.1 --learning_rate_decay_step=54000 \
			--max_number_of_steps=240000 --depth_multiplier=2 \
			--use_batch_norm=True --regularizer=l2 \
			--quantize_delay=160000

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0328/x17_adam_0.003_0.99_0.999_l2 \
			--optimizer=adam \
			--learning_rate_decay_type=exponential --learning_rate=0.003 \
			--learning_rate_decay_factor=0.1 --learning_rate_decay_step=54000 \
			--max_number_of_steps=240000 --depth_multiplier=2 \
			--regularizer=l2 \
			--adam_beta1=0.99 --adam_beta2=0.999 \
			--quantize_delay=160000

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0328/x18_momentum_0.003_l2 \
			--optimizer=momentum \
			--learning_rate_decay_type=fixed --learning_rate=0.003 \
			--max_number_of_steps=240000 --depth_multiplier=2 \
			--regularizer=l2 \
			--quantize_delay=160000

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0328/x19_momentum_0.003_0.999_l2 \
			--optimizer=momentum \
			--learning_rate_decay_type=fixed --learning_rate=0.003 --momentum=0.999 \
			--max_number_of_steps=240000 --depth_multiplier=2 \
			--regularizer=l2 \
			--quantize_delay=160000

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0328/x20_momentum_0.003_l2_mvg_0.999 \
			--optimizer=momentum \
			--learning_rate_decay_type=fixed --learning_rate=0.003 \
			--max_number_of_steps=240000 --depth_multiplier=2 \
			--regularizer=l2 --moving_average_decay=0.999 \
			--quantize_delay=160000

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0328/x21_momentum_0.005_mvg_0.999 \
			--optimizer=momentum \
			--learning_rate_decay_type=fixed --learning_rate=0.005 \
			--max_number_of_steps=240000 --depth_multiplier=2 \
			--moving_average_decay=0.999 \
			--quantize_delay=160000

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0328/x22_momentum_0.005_0.95 \
			--optimizer=momentum --momentum=0.95 \
			--learning_rate_decay_type=fixed --learning_rate=0.005 \
			--max_number_of_steps=240000 --depth_multiplier=2 \
			--quantize_delay=160000

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0328/x23_momentum_0.005_0.95_l2 \
			--optimizer=momentum --momentum=0.95 \
			--learning_rate_decay_type=fixed --learning_rate=0.005 \
			--max_number_of_steps=240000 --depth_multiplier=2 \
			--regularizer=l2 
			--quantize_delay=160000

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0328/x2401_momentum_0.003_l2 \
			--optimizer=momentum \
			--learning_rate_decay_type=exponential --learning_rate=0.003 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=55000 \
			--max_number_of_steps=240000 --depth_multiplier=2 \
			--regularizer=l2 \
			--quantize_delay=160000

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0328/x2501_momentum_0.003_0.999_l2 \
			--optimizer=momentum \
			--learning_rate_decay_type=exponential --learning_rate=0.003 --momentum=0.999 \
			--learning_rate_decay_factor=0.5 --learning_rate_decay_step=55000 \
			--max_number_of_steps=240000 --depth_multiplier=2 \
			--regularizer=l2 \
			--quantize_delay=160000

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0328/x26_momentum_0.005_l2_mvg_0.995 \
			--optimizer=momentum \
			--learning_rate_decay_type=fixed --learning_rate=0.005 \
			--max_number_of_steps=240000 --depth_multiplier=2 \
			--regularizer=l2 --moving_average_decay=0.995 \
			--quantize_delay=160000

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0328/x27_momentum_0.01_0.97 \
			--optimizer=momentum --momentum=0.97 \
			--learning_rate_decay_type=fixed --learning_rate=0.01 \
			--max_number_of_steps=240000 --depth_multiplier=2 \
			--quantize_delay=160000

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0328/x28_rms_0.005_0.95_0.95 \
			--optimizer=rmsprop --rmsprop_decay=0.95 --rmsprop_momentum=0.95\
			--learning_rate_decay_type=fixed --learning_rate=0.005 \
			--max_number_of_steps=240000 --depth_multiplier=2 \
			--quantize_delay=160000

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0328/x29_rms_0.005_0.95_0.95_bn_l2 \
			--optimizer=rmsprop --rmsprop_decay=0.95 --rmsprop_momentum=0.95\
			--learning_rate_decay_type=fixed --learning_rate=0.005 \
			--max_number_of_steps=240000 --depth_multiplier=2 \
			--regularizer=l2 --use_batch_norm=True \
			--quantize_delay=160000

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0328/x30_rms_0.001_0.95_0.95_bn_l2 \
			--optimizer=rmsprop --rmsprop_decay=0.95 --rmsprop_momentum=0.95\
			--learning_rate_decay_type=fixed --learning_rate=0.001 \
			--max_number_of_steps=240000 --depth_multiplier=2 \
			--regularizer=l2 --use_batch_norm=True \
			--quantize_delay=160000

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0328/x31_rms_0.001_0.9_0.99 \
			--optimizer=rmsprop --rmsprop_decay=0.9 --rmsprop_momentum=0.99\
			--learning_rate_decay_type=fixed --learning_rate=0.001 \
			--max_number_of_steps=240000 --depth_multiplier=2 \
			--quantize_delay=160000

