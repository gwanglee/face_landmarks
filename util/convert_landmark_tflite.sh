#!/usr/bin/env bash

export PYTHONPATH=/Users/gglee/Develop/models/research:/Users/gglee/Develop/models/research/slim
echo $PYTHONPATH

MODEL_DIR=/Users/gglee/Data/Landmark/train/0508-2/x001-l1.sgd.0.01.0.25.240000-l2.0.005
CKPT=model.ckpt-840000

freeze_graph --input_graph=$MODEL_DIR/graph.pbtxt \
            --input_checkpoint=$MODEL_DIR/$CKPT \
            --output_graph=$MODEL_DIR/frozen.pb \
            --input_binary=False \
            --output_node_names=model/lannet/fc7/Relu

#python /Users/gglee/Develop/tensorflow/tensorflow/python/tools/optimize_for_inference.py --input=$MODEL_DIR/frozen.pb \
#            --output=$MODEL_DIR/frozen_optimized.pb --frozen_graph=True \
#            --input_names=model/input --output_names=model/lannet/fc7/Relu

tflite_convert --graph_def_file=$MODEL_DIR/frozen.pb \
            --output_file=$MODEL_DIR/landmark.tflite \
            --input_arrays=model/input --output_arrays=model/lannet/fc7/Relu --inference_type=FLOAT