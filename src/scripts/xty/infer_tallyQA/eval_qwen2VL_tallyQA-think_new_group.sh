cd /home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/src/eval

MODEL_BASE=$1
checkpoints=$(ls $MODEL_BASE)
OUTPUT_BASE=${MODEL_BASE}/eval/eval-new_tallyQA-complex-test-subset1.5k

for ckpt in $checkpoints; do

    if [[ ! $ckpt == checkpoint-* ]]; then
        continue
    fi

    OUTPUT_PATH=${OUTPUT_BASE}/${ckpt}
    MODEL_PATH=${MODEL_BASE}/${ckpt}

    echo ckpt $ckpt

    # echo MODEL_PATH $MODEL_PATH

    bash /home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/src/scripts/xty/infer_tallyQA/infer_qwen2VL-2b-instruct_tallyQA_test_complex_subset1.5k_new_think_param.sh ${MODEL_PATH} ${OUTPUT_PATH}

done