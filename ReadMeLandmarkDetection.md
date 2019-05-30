# Facial Landmark Detection

## 1. Training data 준비
- 학습에 사용된 dataset: 300W, menpo, Multi-PIE
    - 300W: https://ibug.doc.ic.ac.uk/resources/300-W/
    - menpo: https://ibug.doc.ic.ac.uk/resources/2nd-facial-landmark-tracking-competition-menpo-ben/
    - CMU Multi-PIE: http://www.cs.cmu.edu/afs/cs/project/PIE/MultiPie/Multi-Pie/Home.html (허준희M 보유)
- 학습 데이터는 (1) *extract_landmark_training_data.py* 와 (2) *make_landmark_tfrecord.py* 를 이용해 생성
    - Landmark 검출기는 N x N 크기 (N=56, currently) 의 입력 영상에 대해서 68 points (x, y)의 좌표를 predict 하며, 학습 data로 얼굴 영역의 patch 와 68 point 좌표가 필요하다.
    - *extract_landmark_training_data.py*: 원본 dataset 에서 얼굴 검출 후 검출된 영역의 영상 저장 + 검출된 영역으로 landmark 좌표 normalize 하여 저장
        - Argument 입력은 구현되어 있지 않고, 필요한 부분이 script 앞부분에 아래와 같이 정의되어 있음. 필요한 설정 변경 후 *extract_landmark_data.py* 실행하여 데이터 변환 진행.
            ~~~
            MIN_CONF = 0.3      # 얼굴 검출의 confidence threshold
            MIN_OVERLAP = 0.3   # IOU threshold: face의 ground truth가 없기 때문에, landmark points의 bounding box와 비교
            EXTEND = True       # 검출된 얼굴 영역 영상 저장 시 영역 확장 여부 
            EXTEND_RATIO = 0.1  # 검출된 얼굴 영역의 확장 시 확장 크기를 결정 (0.1 이면 10% 확장하여 저장)
            
            ROTATE = True       # rotation augmentation 을 사용할 것인지
            MAX_ROTATE = 45     # rotation augmentation 의 범위
            DETECTOR_INPUT_SIZE = 128   # 검출기의 입력 크기
            
            # todo: add jittering?
            
            WRITE_PATH = '/Users/gglee/Data/Landmark/export/0424'   # Crop된 patch와 landmark annotation을 저장할 폴더
            # RAND_FACTOR = 16.0        # Multi-PIE의 경우 양이 많기 때문에 RAND_FACTOR의 비율로 sampling 하여 이용했음
            
            IMAGES_DIR_PATHS = [    # 학습 data를 추출하기 위한 원본 학습 data가 저장되어 있는 위치
                    '/Users/gglee/Data/Landmark/300W/01_Indoor',
                    '/Users/gglee/Data/Landmark/300W/02_Outdoor',
                    '/Users/gglee/Data/Landmark/menpo_train_release',
                    '/Users/gglee/Data/Landmark/mpie'
                ]
            
            FACE_DETECTOR_PATH = '/Users/gglee/Data/TFModels/ssd_mobilenet_v2_quantized_128_v1/freeze/frozen_inference_graph.pb'    # 사용할 얼굴 검출기
            ~~~
        - Script 실행 후 target directory 에 얼굴 영역의 영상은 .png 파일로 저장된다. Landmark annoation 은 영상과 동일한 이름의 .npts, .cpts 확장자로 저장된다.
            - .npts: landmark point 좌표가 0 에서 1 사이로 정규화 된다.
            - .cpts: landmark point 좌표가 -1 에서 1 사이로 정규화 된다.
    - *make_landmark_tfrecord.py*: 검출된 얼굴 영역 patch와 landmark annotation 으로부터 .tfrecord 를 생성한다.
        - 데이터 경로 등 필요한 설정은 main 함수에 다음과 깉이 정의되어 있다. 필요한 부분 수정 후 *make_landmark_tfrecord.py* 를 실행하여 tfrecord 파일를 생성한다.
            ~~~
            DATA_PATH = '/Users/gglee/Data/Landmark/export/0424'    # 얼굴 영상과 point annotation이 저장되어 있는 폴더 경로
            TRAIN_RATIO = 0.9                                       # Training data와 validation data 분할을 위한 비율
            USE_GRAY=True                                           # deprecated: gray color 변환은 경우 학습 코드에서 tfrecord parsing time에 실행하는 것으로 변경
                                                                    # why need gray? 참고 논문이 gray 입력으로 되어있어, 성능에 차이 보이는지 확인 위해 gray도 시도 -> rgb가 더 좋은 성능 보임
            SIZE = 56                                               # landmark 검출 network의 입력 size
        
            TRAIN_TFR_PATH = '/Users/gglee/Data/Landmark/export/0424.%d.gray.train.tfrecord' % SIZE     # train tfrecord 저장 경로
            VAL_TFR_PATH = '/Users/gglee/Data/Landmark/export/0424.%d.gray.val.tfrecord' % SIZE         # validation tfrecord 저장 경로
            ~~~

## 2. Training
- *train_slim.py* 를 이용하여 학습한다 (*script/run_train_slim_MMDD.sh* 참고)
- *train_slim.py* 에 사용할 수 있는 인자는 다음과 같다.
    ~~~
    tf.app.flags.DEFINE_string('train_dir', '/home/gglee/Data/Landmark/train', '학습된 모델을 저장할 경로')
    tf.app.flags.DEFINE_string('train_tfr', '/home/gglee/Data/160v5.0322.train.tfrecord', '.tfrecord for training')
    tf.app.flags.DEFINE_string('val_tfr', '/home/gglee/Data/160v5.0322.val.tfrecord', '.tfrecord for validation')
    tf.app.flags.DEFINE_boolean('is_color', True, 'True 이면 RGB, False 이면 gray 입력 사용')
    tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size to use')
    tf.app.flags.DEFINE_integer('input_size', 56, 'N x N for the network input')
    
    tf.app.flags.DEFINE_string('loss', 'l1', 'Loss func: [l1, l2, wing, euc_wing, pointwise_l2, chain, sqrt]')
    tf.app.flags.DEFINE_string('optimizer', 'sgd', 'Optimizer to use: [adadelt, adagrad, adam, ftrl, momentum, sgd or rmsprop]')
    tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate')
    tf.app.flags.DEFINE_float('wing_w', 0.5, 'w for wing_loss')
    tf.app.flags.DEFINE_float('wing_eps', 2, 'eps for wing_loss')
    
    tf.app.flags.DEFINE_float('momentum', 0.99, 'Momentum for momentum optimizer')
    tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')
    tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')
    tf.app.flags.DEFINE_float('adam_beta1', 0.9, 'The exponential decay rate for the 1st moment estimates.')
    tf.app.flags.DEFINE_float('adam_beta2', 0.999, 'The exponential decay rate for the 2nd moment estimates.')
    
    tf.app.flags.DEFINE_string('learning_rate_decay_type', 'exponential', 'Which learning rate decay to use: [fixed, exponential, or polynomial]')
    tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.94, 'Learning rate decay factor')
    tf.app.flags.DEFINE_integer('learning_rate_decay_step', 50000, 'Decay lr at every N steps')
    tf.app.flags.DEFINE_boolean('learning_rate_decay_staircase', True, 'Staircase decay or not')
    tf.app.flags.DEFINE_float('end_learning_rate', 0.00005, 'The minimal lr used by a polynomial lr decay')
    
    tf.app.flags.DEFINE_float('moving_average_decay', None, 'The decay to use for moving average decay. If left as None, no moving average decay')
    tf.app.flags.DEFINE_integer('max_number_of_steps', 300000, 'The maximum number of training steps')
    tf.app.flags.DEFINE_boolean('use_batch_norm', False, 'To use or not BatchNorm on conv layers')
    
    tf.app.flags.DEFINE_string('regularizer', None, 'l1, l2 or l1_12')
    tf.app.flags.DEFINE_float('regularizer_lambda', 0.004, 'Lambda for the regularization')
    tf.app.flags.DEFINE_float('regularizer_lambda_2', 0.004, 'Lambda for the regularization (for l1_l2)')
    
    tf.app.flags.DEFINE_integer('quantize_delay', -1, 'Number of steps to start quantized training. -1 to disable quantization')
    tf.app.flags.DEFINE_float('depth_multiplier', 2.0, 'Network의 channel depth를 결정')
    tf.app.flags.DEFINE_float('depth_gamma', 1.0, 'Layer 별로 channel depth를 다르게 설정하기 위한 parameter')
    ~~~
- 학습 시 *train_dir* 내에 *train_settin.txt* 가 생성되며, 학습에 사용된 입력 설정을 저장하고 있다.

## 3. PC에서 검증
- 학습된 model을 실행시켜 보기 위해서는 *detect_face_landmark.py* 를  이용한다.
    - *face_checkpoint_dir*: face detector 의 폴더 경로 (fronzen_inference_graph.pb가 해당 폴더 내에 있다고 가정하고 있음)
    - *landmark_checkpoint_path*: 학습된 모델 (ckpt)의 경로
        ~~~
        python detect_face_landmark.py \
                --face_checkpoint_dir=/Users/gglee/Data/TFModels/0515/ssd_face_128_v13/freeze/ \
                --landmark_checkpoint_path=/Users/gglee/Data/Landmark/train/0403_gpu1/x103_l1_sgd_0.003_lrd_0.6_200k_bn_l2_0.005/model.ckpt-900000
        ~~~
- 학습된 model의 성능 평가를 위해서는 *evaluate_landmark_model.py* 를 이용한다.
    - Argument로는 *tfrecord* 와 *models_dir* 을 전달한다.
    - *models_dir*: *models_dir* 의 sub-folders 에 학습된 landmark 모델이 저장되어 있다 (여러 설정으로 학습을 진행하는 경우가 많아 한번에 평가하기 위해 root dir 을 지정) 
        ~~~
        tf.app.flags.DEFINE_string('tfrecord', '/home/gglee/Data/160v5.0322.val.tfrecord', '.tfrecord for evaluation')
        tf.app.flags.DEFINE_string('models_dir', '/home/gglee/Data/Landmark/train/0408', 'where trained models are stored')
        ~~~
    - 평가 완료 시 root dir 에 eval.txt 파일이 생성된다 (Column은 각각 directory 명, ckpt 명, 평균 error)
        ~~~
        x150-wing.sgd.0.01.0.25.240000-l2.0.01	model.ckpt-840000	0.021192
        x151-wing.sgd.0.01.0.25.240000-l2.0.01	model.ckpt-840000	0.019291
        x152-wing.sgd.0.01.0.25.240000-l2.0.01	model.ckpt-840000	0.018687
        x153-wing.sgd.0.01.0.25.240000-l2.0.01	model.ckpt-840000	0.032973
        x154-wing.sgd.0.01.0.25.240000-l2.0.01	model.ckpt-840000	0.026730
        x155-wing.sgd.0.01.0.25.240000-l2.0.01	model.ckpt-840000	0.018690
        x156-wing.sgd.0.01.0.25.240000-l2.0.01	model.ckpt-840000	0.030149
        x157-wing.sgd.0.01.0.25.240000-l2.0.01	model.ckpt-840000	0.019453
        ~~~
    - 평가 완료 시 각 model directory 에 evaluation set 에서 검출한 landmark 결과, best 100 images, worst 100 images 가 mosaic 형태로 저장된다.
## 4. .tflite 변환
- Use *save_with_bnorm_off.py*: 지정된 위치에 있는 모델을 tflite로 저장한다.
    - models_dir: 변환할 모델들이 모여있는 root dir
    - 변환된 모델은 원본 모델과 동일한 폴더 내에 landmark.float.tflite 라는 파일 명으로 생성된다.
    ~~~
    tf.app.flags.DEFINE_string('models_dir', '/Users/gglee/Data/Landmark/train/0508-2', 'where trained models are stored')
    ~~~
    > note: convert_landmark_checkpoint.sh 와 convert_landmark_tflite.st 파일은 사용하지 않음. 처음에 landmark 모델을 tflite 변환하기 위해 사용한 파일이지만, 해당 파일로 변환한 모델은 Android app 에서 동작하지 않는다. BatchNorm이 training mode로 동작되는 것으로 추정.