python train_slim.py --train_dir=/home/gglee/Data/Landmark/train/x8 \
			--optimizer=adam \
			--learning_rate_decay_type=exponential --learning_rate=0.001 \
			--learning_rate_decay_factor=0.995 --num_epochs_per_decay=5.0 \
			--max_number_of_steps=135000

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train/x9 \
			--optimizer=momentum \
			--learning_rate_decay_type=exponential --learning_rate=0.01 \
			--learning_rate_decay_factor=0.995 --num_epochs_per_decay=5.0 \
			--max_number_of_steps=135000 --depth_multiplier=1

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train/x10 \
			--optimizer=momentum \
			--learning_rate_decay_type=exponential --learning_rate=0.01 \
			--learning_rate_decay_factor=0.995 --num_epochs_per_decay=5.0 \
			--max_number_of_steps=135000 --depth_multiplier=2

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train/x11 \
			--optimizer=momentum \
			--learning_rate_decay_type=exponential --learning_rate=0.01 \
			--learning_rate_decay_factor=0.995 --num_epochs_per_decay=5.0 \
			--max_number_of_steps=135000 --depth_multiplier=2 --use_batch_norm=True

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train/x12 \
			--optimizer=momentum \
			--learning_rate_decay_type=exponential --learning_rate=0.01 \
			--learning_rate_decay_factor=0.995 --num_epochs_per_decay=5.0 \
			--max_number_of_steps=135000 --depth_multiplier=2 --moving_average_decay=0.999

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train/x13 \
			--optimizer=sgd \
			--learning_rate_decay_type=exponential --learning_rate=0.01 \
			--learning_rate_decay_factor=0.995 --num_epochs_per_decay=5.0 \
			--max_number_of_steps=135000 --depth_multiplier=2 --moving_average_decay=0.999

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train/x14 \
			--optimizer=momentum \
			--learning_rate_decay_type=exponential --learning_rate=0.01 \
			--learning_rate_decay_factor=0.995 --num_epochs_per_decay=5.0 \
			--max_number_of_steps=135000 --depth_multiplier=2 \
			--regularizer=l1

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train/x15 \
			--optimizer=momentum \
			--learning_rate_decay_type=exponential --learning_rate=0.01 \
			--learning_rate_decay_factor=0.995 --num_epochs_per_decay=5.0 \
			--max_number_of_steps=135000 --depth_multiplier=2 \
			--regularizer=l2

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train/x17 \
			--optimizer=momentum \
			--learning_rate_decay_type=exponential --learning_rate=0.01 \
			--learning_rate_decay_factor=0.995 --num_epochs_per_decay=1.0 \
			--max_number_of_steps=135000 --depth_multiplier=2

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train/x18 \
			--optimizer=momentum \
			--learning_rate_decay_type=exponential --learning_rate=0.01 \
			--learning_rate_decay_factor=0.995 --num_epochs_per_decay=10.0 \
			--max_number_of_steps=135000 --depth_multiplier=2


