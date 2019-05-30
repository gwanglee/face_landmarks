#!/bin/bash

echo "Check if PYTHONPATH contains models/research:models/research/slim"
echo "PYTHONPATH="$PYTHONPATH

echo "Converting .ckpt to fronzen_graph"
TRAINED_DIR=/Users/gglee/Data/Landmark/train/0502/x009-l1.sgd.0.01.0.5.240000-l2.0.0005
PBTXT_NAME="graph.pbtxt"
CKPT_NAME=model.ckpt-831795

freeze_graph --input_graph=$TRAINED_DIR/$PBTXT_NAME --input_binary=false \
             --input_checkpoint=$TRAINED_DIR/$CKPT_NAME \
             --output_graph=$TRAINED_DIR/frozen.pb --output_node_names=model/lannet/fc7/BiasAdd

tflite_convert --output_file=$TRAINED_DIR/landmark_float.tflite \
               --graph_def_file=$TRAINED_DIR/frozen.pb \
               --input_arrays=model/input \
               --output_arrays=model/lannet/fc7/BiasAdd \
               -—inference_type=FLOAT

tflite_convert --output_file=$TRAINED_DIR/landmark_qint8.tflite \
               --graph_def_file=$TRAINED_DIR/frozen.pb \
               --input_arrays=model/input \
               --output_arrays=model/lannet/fc7/BiasAdd \
               -—inference_type=QUANTIZED_INT8 \
               --mean_values=127.5 \
               --std_dev_values=127.5