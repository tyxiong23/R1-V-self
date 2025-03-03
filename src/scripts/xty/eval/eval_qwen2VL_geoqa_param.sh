MODEL_PATH=$1
OUTPUT_DIR=$2

cd /home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/src/eval

python test_qwen2vl_geoqa.py --model_path $MODEL_PATH --output_dir $OUTPUT_DIR