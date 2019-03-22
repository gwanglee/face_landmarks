python train_slim.py --train_dir=/home/gglee/Data/Landmark/train/x0 \
			--optimizer=sgd \
			--learning_rate_decay_type=exponential --learning_rate=0.01 \
			--learning_rate_decay_factor=0.995 --num_epochs_per_decay=5.0 \
			--max_number_of_steps=234567

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train/x1 \
			--optimizer=adam \
			--learning_rate_decay_type=exponential --learning_rate=0.01 \
			--learning_rate_decay_factor=0.995 --num_epochs_per_decay=5.0 \
			--max_number_of_steps=234567

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train/x2 \
			--optimizer=adagrad \
			--learning_rate_decay_type=exponential --learning_rate=0.01 \
			--learning_rate_decay_factor=0.995 --num_epochs_per_decay=5.0 \
			--max_number_of_steps=234567

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train/x3 \
			--optimizer=momentum \
			--learning_rate_decay_type=exponential --learning_rate=0.01 \
			--learning_rate_decay_factor=0.995 --num_epochs_per_decay=5.0 \
			--max_number_of_steps=234567

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train/x4 \
			--optimizer=ftrl \
			--learning_rate_decay_type=exponential --learning_rate=0.01 \
			--learning_rate_decay_factor=0.995 --num_epochs_per_decay=5.0 \
			--max_number_of_steps=234567

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train/x6 \
			--optimizer=sgd \
			--learning_rate_decay_type=exponential --learning_rate=0.01 \
			--learning_rate_decay_factor=0.995 --num_epochs_per_decay=5.0 \
			--moving_averae_decay=0.999 \
			--max_number_of_steps=234567

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train/x7 \
			--optimizer=adam \
			--learning_rate_decay_type=exponential --learning_rate=0.01 \
			--learning_rate_decay_factor=0.995 --num_epochs_per_decay=5.0 \
			--moving_averae_decay=0.999 \
			--max_number_of_steps=234567

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train/x8 \
			--optimizer=sgd \
			--learning_rate_decay_type=exponential --learning_rate=0.01 \
			--learning_rate_decay_factor=0.999 --num_epochs_per_decay=5.0 \
			--max_number_of_steps=234567

python train_slim.py --train_dir=/home/gglee/Data/Landmark/train/x9 \
			--optimizer=adam \
			--learning_rate_decay_type=exponential --learning_rate=0.01 \
			--learning_rate_decay_factor=0.999 --num_epochs_per_decay=5.0 \
			--max_number_of_steps=234567
