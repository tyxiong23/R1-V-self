cd /home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/src/eval

MODEL_BASE=$1
checkpoints=$(ls $MODEL_BASE)
OUTPUT_BASE=${MODEL_BASE}/eval/eval_superclevr

for ckpt in $checkpoints; do
    if [[ ! $ckpt == checkpoint-* ]]; then
        continue
    fi
    OUTPUT_PATH=${OUTPUT_BASE}/${ckpt}
    MODEL_PATH=${MODEL_BASE}/${ckpt}

    echo MODEL_PATH $MODEL_PATH

    bash /home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/src/scripts/xty/eval/eval_qwen2VL_superclevr_param.sh ${MODEL_PATH} ${OUTPUT_PATH}

done
