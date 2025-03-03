MODEL_PATH=$1

cd /home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/src/eval

python test_qwen2vl_geoqa.py --model_path Qwen/Qwen2-VL-2B-Instruct --output_dir /home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/eval/Qwen2-VL-2B-Instruct/eval_geoqa

python test_qwen2vl_geoqa.py --model_path Qwen/Qwen2-VL-7B-Instruct --output_dir /home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/eval/Qwen2-VL-7B-Instruct/eval_geoqa