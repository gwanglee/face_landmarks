python train_slim.py --train_dir=/home/gglee/Data/Landmark/train/x11 \
			--optimizer=momentum \
			--learning_rate_decay_type=exponential --learning_rate=0.01 \
			--learning_rate_decay_factor=0.995 --num_epochs_per_decay=5.0 \
			--max_number_of_steps=135000 --depth_multiplier=2 --use_batch_norm=True

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train/x16 \
			--optimizer=adam \
			--learning_rate_decay_type=exponential --learning_rate=0.001 \
			--learning_rate_decay_factor=0.995 --num_epochs_per_decay=10.0 \
			--max_number_of_steps=135000 --depth_multiplier=2 \
			--regularizer=l1

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train/x19 \
			--optimizer=adam \
			--learning_rate_decay_type=fixed --learning_rate=0.001 \
			--max_number_of_steps=135000 --depth_multiplier=2 \
			--use_batch_norm=True --moving_average_decay=0.999

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train/x20 \
			--optimizer=adam --learning_rate_decay_type=polynomial \
			--learning_rate=0.001 --num_epochs_per_decay=5.0 \
			--max_number_of_steps=135000 --depth_multiplier=2 \
			--use_batch_norm=True --moving_average_decay=0.999 --regularizer=l2

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train/x21 \
			--optimizer=adam --learning_rate_decay_type=exponential \
			--learning_rate=0.001 --num_epochs_per_decay=5.0 \
			--max_number_of_steps=135000 --depth_multiplier=2 \
			--use_batch_norm=True --moving_average_decay=0.9999 --regularizer=l2

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train/x22 \
			--optimizer=momentum \
			--learning_rate_decay_type=exponential --learning_rate=0.001 \
			--learning_rate_decay_factor=0.995 --num_epochs_per_decay=10.0 \
			--max_number_of_steps=135000 --depth_multiplier=2 \
			--regularizer=l1

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train/x23 \
			--optimizer=momentum \
			--learning_rate_decay_type=exponential --learning_rate=0.001 \
			--learning_rate_decay_factor=0.999 --num_epochs_per_decay=5.0 \
			--max_number_of_steps=135000 --depth_multiplier=2 \
			--regularizer=l1

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train/x24 \
			--optimizer=momentum \
			--learning_rate_decay_type=exponential --learning_rate=0.001 \
			--learning_rate_decay_factor=0.995 --num_epochs_per_decay=5.0 \
			--max_number_of_steps=135000 --depth_multiplier=2 \
			--use_batch_norm=True

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train/x25 \
			--optimizer=momentum \
			--learning_rate_decay_type=fixed --learning_rate=0.001 \
			--max_number_of_steps=135000 --depth_multiplier=2 \
			--use_batch_norm=True --moving_average_decay=0.999

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train/x26 \
			--optimizer=momentum --learning_rate_decay_type=polynomial \
			--learning_rate=0.001 --num_epochs_per_decay=5.0 \
			--max_number_of_steps=135000 --depth_multiplier=2 \
			--use_batch_norm=True --moving_average_decay=0.999 --regularizer=l2

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train/x27 \
			--optimizer=momentum --learning_rate_decay_type=exponential \
			--learning_rate=0.001 --num_epochs_per_decay=5.0 \
			--max_number_of_steps=135000 --depth_multiplier=2 \
			--use_batch_norm=True --moving_average_decay=0.9999 --regularizer=l2

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train/x28 \
			--optimizer=sgd \
			--learning_rate_decay_type=exponential --learning_rate=0.005 \
			--learning_rate_decay_factor=0.995 --num_epochs_per_decay=10.0 \
			--max_number_of_steps=135000 --depth_multiplier=2 \
			--regularizer=l2 --use_batch_norm=True

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train/x29 \
			--optimizer=sgd \
			--learning_rate_decay_type=exponential --learning_rate=0.001 \
			--learning_rate_decay_factor=0.999 --num_epochs_per_decay=10.0 \
			--max_number_of_steps=135000 --depth_multiplier=2 \
			--regularizer=l1 --use_batch_norm=True

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train/x30 \
			--optimizer=sgd \
			--learning_rate_decay_type=exponential --learning_rate=0.001 \
			--learning_rate_decay_factor=0.995 --num_epochs_per_decay=5.0 \
			--max_number_of_steps=135000 --depth_multiplier=2 \
			--use_batch_norm=True --moving_average_decay=0.99 --regularizer=l2


python train_slim.py --train_dir=/home/gglee/Data/Landmark/train/x31 \
			--optimizer=adam \
			--learning_rate_decay_type=exponential --learning_rate=0.001 \
			--learning_rate_decay_factor=0.999 --num_epochs_per_decay=5.0 \
			--max_number_of_steps=135000 --depth_multiplier=2 \
			--regularizer=l1

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train/x32 \
			--optimizer=adam \
			--learning_rate_decay_type=exponential --learning_rate=0.001 \
			--learning_rate_decay_factor=0.995 --num_epochs_per_decay=5.0 \
			--max_number_of_steps=135000 --depth_multiplier=2 \
			--use_batch_norm=True

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
