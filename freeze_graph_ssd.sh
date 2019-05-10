export PYTHONPATH=/Users/gglee/Develop/models/research:/Users/gglee/Develop/models/research/slim
echo $PYTHONPATH

MODEL_DIR=/Users/gglee/Data/TFModels/ssd_revisit/ssd_face_128_v6
CONFIG=ssd_face_128_v6.config
CKPT=model.ckpt-560000

python /Users/gglee/Develop/models/research/object_detection/export_inference_graph.py --input_type=image_tensor \
        --pipeline_config_path=$MODEL_DIR/$CONFIG --trained_checkpoint_prefix=$MODEL_DIR/$CKPT --output_directory=$MODEL_DIR/freeze