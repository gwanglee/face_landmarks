#!/bin/bash

echo "Check if PYTHONPATH contains models/research:models/research/slim"
echo "PYTHONPATH="$PYTHONPATH

echo "Converting .ckpt to fronzen_graph"
CKPT_PATH="/Users/gglee/Data/TFModels/0515/ssd_face_128_v14"
CKPT_NAME="model.ckpt-560000"
PIPELINE_CONFIG_NAME="ssd_face_128_v14.config"

cd /Users/gglee/Develop/models/research

python object_detection/export_inference_graph.py --input_type image_tensor \
        --pipeline_config_path=$CKPT_PATH/$PIPELINE_CONFIG_NAME \
        --trained_checkpoint_prefix=$CKPT_PATH/$CKPT_NAME \
        --output_directory=$CKPT_PATH/freeze

echo "Converting .ckpt to "

python object_detection/export_tflite_ssd_graph.py \
        --pipeline_config_path=$CKPT_PATH/$PIPELINE_CONFIG_NAME \
        --trained_checkpoint_prefix=$CKPT_PATH/$CKPT_NAME \
        --output_directory=$CKPT_PATH/tflite \
       --add_postprocessing_op=true

toco --graph_def_file=$CKPT_PATH/tflite/tflite_graph.pb \
        --output_file=$CKPT_PATH/tflite/model.tflite \
        --input_shapes=1,128,128,3 \
        --input_arrays=normalized_input_image_tensor \
        --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' --inference_type=QUANTIZED_UINT8 \
        --mean_values=128 --std_dev_values=128 --changed_concat_input_ranges=false --allow_custom_ops