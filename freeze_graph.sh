export PYTYOHPATH=/Users/gglee/Develop/models/research:/Users/gglee/Develop/model/research/slim

MODEL_DIR=/Users/gglee/Data/TFModels/ssd_face_128_v3
CONFIG=ssd_mobilenet_v2_quantized_128_v3.config
CKPT=model.ckpt-97234

python /Users/gglee/Develop/models/research/object_detection/export_inference_graph.py --input_type=image_tensor \
        --pipeline_config_path=$MODEL_DIR/$CONFIG --trained_checkpoint_prefix=$MODEL_DIR/$CKPT --output_directory=$MODEL_DIR/freeze