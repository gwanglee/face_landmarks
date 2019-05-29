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
            MIN_CONF = 0.3      # 얼굴 검출 시 Confidence
            MIN_OVERLAP = 0.3   # 얼굴 검출 시 IOU threshold: face의 ground truth가 없기 때문에, landmark points의 bounding box와 비교
            EXTEND = True       # 검출된 얼굴 영역 영상 저장 시 영역을 확장할 것인지 결정 
            EXTEND_RATIO = 0.1  # 검출된 얼굴 영역의 확장 시 확장 크기를 결정 (0.1 이면 10% 확장하여 저장)
            
            ROTATE = True       # rotation augmentation 을 사용할 것인지
            MAX_ROTATE = 45     # rotation augmentation 의 범위
            DETECTOR_INPUT_SIZE = 128   # 검출기의 입력 크기
            
            # todo: add jittering?
            
            WRITE_PATH = '/Users/gglee/Data/Landmark/export/0424'   # Crop된 patch와 landmark annotation을 저장할 폴더
            # RAND_FACTOR = 16.0        # Multi-PIE의 경우 양이 많기 때문에 RAND_FACTOR의 비율로 sampling 하여 이용
            
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
    - *make_landmark_tfrecord.py*: 검출된 얼굴 영역 patch와 landmark annotation (.txt)로부터 .tfrecord 를 생성한다.
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

## 2. Train
- script 내의 *run_train_slim_MMDD.sh* 와 *ssd_face_XXX_vYY.config* 참고
    
## 3. .tflite 변환
- Use *save_with_bnorm_off.py*
    > note: convert_landmark_checkpoint.sh 와 convert_landmark_tflite.st 파일은 사용하지 않음. 처음에 landmark 모델을 tflite 변환하기 위해 사용한 파일이지만, 해당 파일로 변환한 모델은 Android app 에서 동작하지 않는다. BatchNorm이 training mode로 동작되는 것으로 추정.