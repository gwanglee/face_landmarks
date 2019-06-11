# Making Face Detector for Landmark Detection
##1. 학습 데이터 생성
- WiderFace DB와 자체 DB를 학습에 이용하며, spec에 맞는 크기의 얼굴만 DB에 포함되도록 DB를 refine 하는 과정을 거친다.

    - 가장 작은 크기 얼굴이 프레임 너비의 10%가 되도록 image를 crop 하고 ground truth 를 변경한다 (use *refine_widerface_2.py*).
    
        ~~~
        python data/refine_widerface_2.py --image_dir=/Users/gglee/Data/WiderFace/WIDER_train/images/ --gt_path=/Users/gglee/Data/WiderFace/wider_face_split/wider_face_train_bbx_gt.txt --output_image_dir=/Users/gglee/Data/WiderRefine/wider_train_0522_10% --output_gt_path=/Users/gglee/Data/WiderRefine/wider_train_0522_10%.txt --min_rel_size=0.1
        python data/refine_widerface_2.py --image_dir=/Users/gglee/Data/face_ours/ --gt_path=/Users/gglee/Data/face_ours/face_ours.txt --output_image_dir=/Users/gglee/Data/face_train --output_gt_path=/Users/gglee/Data/face_train/gt.txt --min_rel_size=0.1
        ~~~
        - image_dir 과 gt_path 는 원 영상과 원 ground truth file (WiderFace format)이고, 변경된 영상과 ground truth 는 output_image_dir 과 output_gt_path 에 생성된다.
        - min_rel_size는 수정된 DB에서 얼굴의 최소 크기를 설정한다. 0.1인 경우 학습 DB 내 얼굴이 영상의 너비 대비 10% 이상 되도록 조절한다.
        - 위 command 는 WiderFaceDB와 자체 DB를 각각 refine 하는 예제이다.
        > Note: refine_widerface_2.py 의 logic 에 관한 설명은 [여기](https://tde.sktelecom.com/wiki/download/attachments/230971196/refine_wider_2.pptx?version=1&modificationDate=1558663095778&api=v2) 참고한다.
- 변경한 image / ground truth 로부터 .tfrecord 를 생성한다.
    - WiderFace DB와 자체 DB 를 manually 합친다 (영상을 동일 폴더에 복사. gt 파일을 edit하여 하나로 합친다).
    - make_widerface_tfrecord.py 를 이용해서 tfrecord 파일을 만든다.
    - negative 영상이 있을 경우 negative_path option을 통해 설정한다. Negative data가 positive에 비해 너무 많은 경우 negative_sample을 이용해 random sample 한다.
    
        ~~~
        python data/make_widerface_tfrecord.py --image_dir=/Users/gglee/Data/face_train/ --gt_path=/Users/gglee/Data/face_train/gt.txt --output_path=/Users/gglee/Data/face_train/face_train_0522_10%.tfrecord --negative_path=/Users/gglee/Data/face_negative/ --negative_sample=30
        ~~~

##2. Run training
- Tensorflow Object Detection API (TF_OD_API)를 이용하여 얼굴검출기를 학습한다 (ssd_gpuX_MMDD.sh과 ssd_face_YYY_vZZ.config 형식의 파일을 참고)
    - ssd_gpuX_MMDD.sh: 학습을 위한 script (dgx(242) 기준으로 폴더 설정이 되어있음)
    - ssd_face_YYY_vZZ.config: 모델 및 학습 configuration
> - Note: 검출 성능 향상을 위해 TF_OD_API 일부 코드를 변경한 부분이 있으며, 상세 내용은 [여기](https://tde.sktelecom.com/wiki/download/attachments/229489277/Anchors%20in%20TF_OD_API.pptx?version=2&modificationDate=1556844425000&api=v2)를 참고한다.
> - 코드 변경은 *models/research/object_detection/anchor_generators/multiple_grid_anchor_generator.py* 파일의 line 322를 다음과 같이 고친다.
~~~
    if layer == 0 and reduce_boxes_in_lowest_layer:
    # layer_box_specs = [(0.1, 1.0), (scale, 2.0), (scale, 0.5)]      # 기존: TF original
      layer_box_specs = [(0.1, 1.0), (0.173, 1.0), (0.246, 1.0)]      # handcraft
~~~ 

- Code 위치@DGX: 172.27.107.242:/home/app_su/youjin/face_landmark/
    - code: face_landmarks
    - models_latest: Tensorflow (기존 코드를 위 내용대로 변경해서 사용해도 무방)
    - data: tfrecords
    - train: trained models


##3. ckpt를 tflite 로 변환
- *./util/conver_face_checkpoint.sh* 내 파일 경로를 수정한다
    ~~~
    echo "Converting .ckpt to fronzen_graph"
    CKPT_PATH="/Users/gglee/Data/TFModels/0522/ssd_face_128_v18"
    CKPT_NAME="model.ckpt-382254"
    PIPELINE_CONFIG_NAME="ssd_face_128_v14.config"
    ~~~
    - CKPT_PATH: 학습된 모델 (.ckpt)가 들어있는 폴더 경로
    - CKPT_NAME: 변경할 모델의 파일 명 (mode.ckpt-XXXXXXX)
    - PIPELINE_CONFIG_NAME: 학습에 사용된 config file
    
- *./util/convert_face_checkpoint.sh* 를 실행시켜 .ckpt를 .tflite로 변환한다.
    ~~~
    ./util/convert_face_checkpoint.sh
    ~~~
- box_prior.txt 파일을 생성한다.
    - tflite의 ssd model에는 box_prior.txt 파일이 함께 배포되는데, 해당 파일 생성을 위한 documentation이나 별도 tool이 공개되어 있지 않다. TF_OD_API 내의 anchor generator 코드를 수정하여 box_prior.txt 를 생성한다. 
    
        >  #### 참고: box_prior.txt 생성을 위한 TF_OD_API 변경
        >- *models/research/object_detection/core/anchor_generator/multiple_grid_anchor_generator_test.py* 를 수정하여 이용한다
        >- 아래와 같이 *multiple_grid_anchor_generator_test.py* 파일을 TF_OD_API 학습 시 .config 파일와 일치하도록 변경한다.
        ~~~
          def test_create_ssd_anchors_returns_correct_shape(self):
    
            def graph_fn1():
              anchor_generator = ag.create_ssd_anchors(
                  num_layers=6,
                  min_scale=0.2,        
                  max_scale=0.8,
                  # max_scale=0.9,      # max_scale을 0.9 -> 0.8로 낮춤
                  aspect_ratios=[1.0],
                  #aspect_ratios=(1.0, 2.0, 3.0, 1.0 / 2, 1.0 / 3),  # 한개의 aspect_ratio만 이용하도록 변경
                  reduce_boxes_in_lowest_layer=True)
        
              # feature_map_shape_list = [(38, 38), (19, 19), (10, 10), # feature_max_shape 변경
              #                           (5, 5), (3, 3), (1, 1)]
        
              # feature_map_shape_list = [(19, 19), (10, 10), (5, 5),
              #                           (3, 3), (2, 2), (1, 1)]     # 300x300 => makes 1917 anchors
        
              # feature_map_shape_list = [(12, 12), (6, 6), (3, 3),
              #                           (2, 2), (1, 1), (1, 1)]     # for 192x192
        
              # feature_map_shape_list = [(10, 10), (5, 5), (3, 3),
              #                           (2, 2), (1, 1), (1, 1)]  # for 160x160
        
              feature_map_shape_list = [(8, 8), (4, 4), (2, 2),
                                        (1, 1), (1, 1), (1, 1)]  # for 128x128
        ~~~
        >- 생성된 anchor를 text file로 저장할 수 있도록 코드를 추가한다.
        ~~~
        anchor_corners_out = np.concatenate(self.execute(graph_fn1, []), axis=0)
        aco_trans = anchor_corners_out.transpose()
        np.savetxt('/Users/gglee/Data/anchors_128x128_handcraft.txt', aco_trans, fmt='%.8f')
        print("box_prior:", aco_trans.shape)
        ~~~
        >- TF_OD_API에서 생성한 anchor 좌표와 box_prior.txt에서 사용하는 anchor 좌표는 형식에 차이([*x, y ,w, h*] vs [*l, t, r, b*])가 있기 때문에 *grid_anchor_generator.py*를 변경하여 출력 형태를 변경한다.
        ~~~
        def _center_size_bbox_to_corners_bbox(centers, sizes):
          """Converts bbox center-size representation to corners representation.
        
          Args:
            centers: a tensor with shape [N, 2] representing bounding box centers
            sizes: a tensor with shape [N, 2] representing bounding boxes
        
          Returns:
            corners: tensor with shape [N, 4] representing bounding boxes in corners
              representation
          """
          return tf.concat([centers, sizes], 1)     # TF Lite: box_prior format
          # return tf.concat([centers - .5 * sizes, centers + .5 * sizes], 1)    # TF: original
        ~~~ 
        > - 주의사항: 위 _center_size_bbox_to_corners_bbox 함수 내용이 변경된 채로 TF_OD_API를 실행시키면 (학습, model freezing, inference 등) 변경된 anchor 형식으로 인해 잘못된 결과가 나오게 된다. box_prior.txt 생성시에만 변경된 코드가 사용될 수 있도록 주의해야 한다. 
        

- 생성된 tflite 및 box_prior.txt 파일을 안드로이드 소스코드의 assets 폴더에 복사하여 이용한다.
    - 여러 모델이 assets 폴더에 있을 경우 .tflite 파일은 java 코드 내에서 파일 명을 변경해서 이용 가능하나, box_prior.txt는 파일명을 설정하는 부분이 없음. 사용할 파일명을 box_prior.txt로 변경해서 이용해야 함