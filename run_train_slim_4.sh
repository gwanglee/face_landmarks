xpython train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0327/x7_adam_0.001_bn_l2 \
			--optimizer=adam \
			--learning_rate_decay_type=fixed --learning_rate=0.001 \
			--max_number_of_steps=180000 --depth_multiplier=2 \
			--use_batch_norm=True --regularizer=l2

xpython train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0327/x8_adam_0.001_0.99_0.999_bn_l2 \
			--optimizer=adam --adam_beta1=0.99 --adam_beta2=0.999 \
			--learning_rate_decay_type=fixed --learning_rate=0.001 \
			--max_number_of_steps=180000 --depth_multiplier=2 \
			--use_batch_norm=True --regularizer=l2

pxython train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0327/x9_adam_0.001 \
			--optimizer=adam \
			--learning_rate_decay_type=fixed --learning_rate=0.001 \
			--max_number_of_steps=180000 --depth_multiplier=2

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0327/x10_adam_0.005_l2_mvg_0.998 \
			--optimizer=adam \
			--learning_rate_decay_type=fixed --learning_rate=0.005 \
			--max_number_of_steps=180000 --depth_multiplier=2 \
			--use_batch_norm=False --regularizer=l2 --moving_average_decay=0.998

xpython train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0327/x11_adam_0.005_l2_bn \
			--optimizer=adam \
			--learning_rate_decay_type=fixed --learning_rate=0.005 \
			--max_number_of_steps=180000 --depth_multiplier=2 \
			--use_batch_norm=True --regularizer=l2

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0327/x12_adam_0.005_l2_mvg_0.99 \
			--optimizer=adam \
			--learning_rate_decay_type=fixed --learning_rate=0.005 \
			--max_number_of_steps=180000 --depth_multiplier=2 \
			--use_batch_norm=False --regularizer=l2 --moving_average_decay=0.99

xpython train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0327/x13_momentum_0.005 \
			--optimizer=momentum \
			--learning_rate_decay_type=fixed --learning_rate=0.005 \
			--max_number_of_steps=180000 --depth_multiplier=2

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0327/x14_momentum_0.005_bn_l2_mvg_0.995 \
			--optimizer=momentum \
			--learning_rate_decay_type=fixed --learning_rate=0.005 \
			--max_number_of_steps=160000 --depth_multiplier=2 \
			--use_batch_norm=True --regularizer=l2 --moving_average_decay=0.995

xpython train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0327/x15_momentum_0.001_bn_l2 \
			--optimizer=momentum \
			--learning_rate_decay_type=fixed --learning_rate=0.001 \
			--max_number_of_steps=160000 --depth_multiplier=2 \
			--use_batch_norm=True --regularizer=l2

xpython train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0327/x16_momentum_0.005 \
			--optimizer=momentum \
			--learning_rate_decay_type=fixed --learning_rate=0.005 \
			--max_number_of_steps=180000 --depth_multiplier=2

xpython train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0327/x17_momentum_0.001_bn_l1 \
			--optimizer=momentum \
			--learning_rate_decay_type=fixed --learning_rate=0.001 \
			--max_number_of_steps=180000 --depth_multiplier=2 \
			--use_batch_norm=True --regularizer=l1

xpython train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0327/x18_momentum_0.005_lrd_0.95_30k_bn_l1 \
			--optimizer=momentum \
			--learning_rate_decay_type=exponential --learning_rate=0.005 \
			--learning_rate_decay_type=exponential --learning_rate=0.01 \
			--learning_rate_decay_factor=0.95 --learning_rate_decay_step=30000 --learning_rate_decay_staircase=True \
			--max_number_of_steps=180000 --depth_multiplier=2 \
			--use_batch_norm=True --regularizer=l1

xpython train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0327/x0_sgd_0.01_lrd_0.95_30k \
			--optimizer=sgd \
			--learning_rate_decay_type=exponential --learning_rate=0.01 \
			--learning_rate_decay_factor=0.95 --learning_rate_decay_step=30000 --learning_rate_decay_staircase=True \
			--max_number_of_steps=180000 --depth_multiplier=2 --use_batch_norm=False

xpython train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0327/x1_sgd_0.001_lrd_0.95_30k \
			--optimizer=sgd \
			--learning_rate_decay_type=exponential --learning_rate=0.001 \
			--learning_rate_decay_factor=0.95 --learning_rate_decay_step=30000 --learning_rate_decay_staircase=True \
			--max_number_of_steps=180000 --depth_multiplier=2 --use_batch_norm=False

xpython train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0327/x2_sgd_0.005_lrd_0.95_15k \
			--optimizer=sgd \
			--learning_rate_decay_type=exponential --learning_rate=0.005 \
			--learning_rate_decay_factor=0.95 --learning_rate_decay_step=15000 --learning_rate_decay_staircase=True \
			--max_number_of_steps=180000 --depth_multiplier=2 --use_batch_norm=False

xpython train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0327/x3_sgd_0.005_lrd_0.95_15k_ns \
			--optimizer=sgd \
			--learning_rate_decay_type=exponential --learning_rate=0.005 \
			--learning_rate_decay_factor=0.95 --learning_rate_decay_step=15000 --learning_rate_decay_staircase=False \
			--max_number_of_steps=180000 --depth_multiplier=2 --use_batch_norm=False

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0327/x6_sgd_0.001_lrd_0.95_30k_mvg_0.998_l2 \
			--optimizer=sgd \
			--learning_rate_decay_type=exponential --learning_rate=0.001 \
			--learning_rate_decay_factor=0.95 --learning_rate_decay_step=30000 --learning_rate_decay_staircase=True \
			--max_number_of_steps=180000 --depth_multiplier=2 --use_batch_norm=True \
			--regularizer=l2 --moving_average_decay=0.998

xpython train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0327/x19_rms_0.01_0.9_0.9_bn \
			--optimizer=rmsprop \
			--learning_rate_decay_type=fixed --learning_rate=0.01 --rmsprop_decay=0.9 --rmsprop_momentum=0.9 \
			--learning_rate_decay_factor=0.95 --learning_rate_decay_step=30000 --learning_rate_decay_staircase=True \
			--max_number_of_steps=180000 --depth_multiplier=2 --use_batch_norm=True

xpython train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0327/x20_rms_0.01_0.99_0.99 \
			--optimizer=rmsprop \
			--learning_rate_decay_type=fixed --learning_rate=0.01 --rmsprop_decay=0.99 --rmsprop_momentum=0.99 \
			--learning_rate_decay_factor=0.95 --learning_rate_decay_step=30000 --learning_rate_decay_staircase=True \
			--max_number_of_steps=180000 --depth_multiplier=2

xpython train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0327/x21_rms_0.01_lrd_0.95_30k_bn \
			--optimizer=rmsprop \
			--learning_rate_decay_type=exponential --learning_rate=0.01 \
			--learning_rate_decay_factor=0.95 --learning_rate_decay_step=30000 --learning_rate_decay_staircase=True \
			--max_number_of_steps=180000 --depth_multiplier=2 --use_batch_norm=True

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0327/x4_sgd_0.001_lrd_0.95_30k_bn \
			--optimizer=sgd \
			--learning_rate_decay_type=exponential --learning_rate=0.001 \
			--learning_rate_decay_factor=0.95 --learning_rate_decay_step=30000 --learning_rate_decay_staircase=True \
			--max_number_of_steps=180000 --depth_multiplier=2 --use_batch_norm=True

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train_0327/x5_sgd_0.001_lrd_0.95_30k_bn_l2 \
			--optimizer=sgd \
			--learning_rate_decay_type=exponential --learning_rate=0.001 \
			--learning_rate_decay_factor=0.95 --learning_rate_decay_step=30000 --learning_rate_decay_staircase=True \
			--max_number_of_steps=180000 --depth_multiplier=2 --use_batch_norm=True \
			--regularizer=l2
