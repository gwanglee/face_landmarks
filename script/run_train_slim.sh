python train_slim.py --train_dir=/home/gglee/Data/Landmark/train/x0 \
			--optimizer=sgd \
			--learning_rate_decay_type=exponential --learning_rate=0.01 \
			--learning_rate_decay_factor=0.995 --num_epochs_per_decay=5.0 \
			--max_number_of_steps=135000

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train/x1 \
			--optimizer=sgd \
			--learning_rate_decay_type=exponential --learning_rate=0.01 \
			--learning_rate_decay_factor=0.995 --num_epochs_per_decay=5.0 \
			--moving_average_decay=0.995 \
			--max_number_of_steps=135000

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train/x2 \
			--optimizer=sgd \
			--learning_rate_decay_type=exponential --learning_rate=0.001 \
			--learning_rate_decay_factor=0.995 --num_epochs_per_decay=5.0 \
			--max_number_of_steps=135000

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train/x3 \
			--optimizer=sgd \
			--learning_rate_decay_type=exponential --learning_rate=0.001 \
			--learning_rate_decay_factor=0.999 --num_epochs_per_decay=5.0 \
			--max_number_of_steps=135000

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train/x4 \
			--optimizer=adam \
			--learning_rate_decay_type=exponential --learning_rate=0.01 \
			--learning_rate_decay_factor=0.995 --num_epochs_per_decay=5.0 \
			--max_number_of_steps=135000

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train/x5 \
			--optimizer=adagrad \
			--learning_rate_decay_type=exponential --learning_rate=0.01 \
			--learning_rate_decay_factor=0.995 --num_epochs_per_decay=5.0 \
			--max_number_of_steps=135000

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train/x6 \
			--optimizer=momentum \
			--learning_rate_decay_type=exponential --learning_rate=0.01 \
			--learning_rate_decay_factor=0.995 --num_epochs_per_decay=5.0 \
			--max_number_of_steps=135000

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train/x7 \
			--optimizer=ftrl \
			--learning_rate_decay_type=exponential --learning_rate=0.01 \
			--learning_rate_decay_factor=0.995 --num_epochs_per_decay=5.0 \
			--max_number_of_steps=135000

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train/x8 \
			--optimizer=adam \
			--learning_rate_decay_type=exponential --learning_rate=0.001 \
			--learning_rate_decay_factor=0.995 --num_epochs_per_decay=5.0 \
			--max_number_of_steps=135000
