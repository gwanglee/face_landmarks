#!/bin/bash

echo "Check if PYTHONPATH contains models/research:models/research/slim"
echo "PYTHONPATH="$PYTHONPATH

echo "Converting .ckpt to fronzen_graph"
TRAINED_DIR="/Users/gglee/Data/TFModels/ssd_mobilenet_v2_quantized_160_v5"
PBTXT_NAME="graph.pbtxt"
CKPT_NAME="model.ckpt-299331"

cd /Users/gglee/Develop/models/research

freeze_graph --input_graph=$TRAINED_DIR/$PBTXT_NAME --input_binary=false \
             --input_checkpoint=$TRAINED_DIR/$CKPT_NAME \
             --output_graph=$TRAINED_DIR/frozen.pb --output_node_names=model/lannet/fc7f/BiasAdd

tflite_convert --output_file=$TRAINED_DIR/tflite/landmark_float.tflite \
               --graph_def_file=$TRAINED_DIR/frozen.pb \
               --input_arrays=model/input \
               --output_arrays=lannet/fc7/BiasAdd \
               -—inference_type=FLOAT

tflite_convert --output_file=$TRAINED_DIR/tflite/landmark_qint8.tflite \
               --graph_def_file=$TRAINED_DIR/frozen.pb \
               --input_arrays=model/input \
               --output_arrays=lannet/fc7/BiasAdd \
               -—inference_type=QUANTIZED_INT8 \
               --mean_values=127.5 \
               --std_dev_values=127.5