cd /home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/src/eval

MODEL_BASE=$1
checkpoints=$(ls $MODEL_BASE)
echo $checkpoints
OUTPUT_BASE=${MODEL_BASE}/eval/eval_pixmo-count-test

for ckpt in $checkpoints; do

    if [[ ! $ckpt == global_step_* ]]; then
        continue
    fi

    OUTPUT_PATH=${OUTPUT_BASE}/${ckpt}
    MODEL_PATH=${MODEL_BASE}/${ckpt}/actor/huggingface

    echo ckpt $ckpt

    # echo MODEL_PATH $MODEL_PATH

    bash /home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/src/scripts/xty/infer_pixmo-counting/infer_pixmo-counting_think_param.sh ${MODEL_PATH} ${OUTPUT_PATH}

    python /home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/src/eval/calc_result.py $OUTPUT_PATH

done