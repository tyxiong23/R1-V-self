cd /home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/src/eval

MODEL_BASE=$1
checkpoints=$(ls $MODEL_BASE)
OUTPUT_BASE=${MODEL_BASE}/eval/eval_geoqa

for ckpt in $checkpoints; do

    if [[ ! $ckpt == checkpoint-* ]]; then
        continue
    fi

    OUTPUT_PATH=${OUTPUT_BASE}/${ckpt}
    MODEL_PATH=${MODEL_BASE}/${ckpt}

    echo ckpt $ckpt

    # echo MODEL_PATH $MODEL_PATH

    bash /home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/src/scripts/xty/eval/eval_qwen2VL_geoqa_param.sh ${MODEL_PATH} ${OUTPUT_PATH}

done
# python test_qwen2vl_geoqa.py --model_path Qwen/Qwen2-VL-2B-Instruct --output_dir /home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/eval/Qwen2-VL-2B-Instruct/eval_geoqa