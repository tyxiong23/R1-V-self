export WANDB_API_KEY="0bba0b55ec140e44f4519b124800b91b6cbf146f"
export WANDB_ENTITY=xty620682
export WANDB_MODE=online
export WANDB_PROJECT=R1-V

PORTOFOLIOS="nvr"
export HF_HOME="/lustre/fsw/portfolios/${PORTOFOLIOS}/users/tixiong/cache/huggingface"

WORKDIR="/lustre/fsw/portfolios/${PORTOFOLIOS}/users/tixiong/xty-workspace/multimodal-reasoning/R1-V"
cd ${WORKDIR}
# cd ${WORKDIR}/src/r1-v

IMAGE_FOLDER="/lustre/fsw/portfolios/${PORTOFOLIOS}/users/tixiong/datasets/data/coco"

# MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
# MODEL_SHORT=$(basename "$MODEL_PATH")
MODEL_PATH="/home/tixiong/storage/checkpoints/Qwen2.5-VL-3B-Instruct-SpecialToken-v2-NUM-32x32"
MODEL_SHORT='Qwen2.5-VL-3B-Instruct'
EPOCH=1
lr=$1

DATA_NAME="refcocog-select40k-specialToken-v2"
DATA_PATH="/lustre/fsw/portfolios/${PORTOFOLIOS}/users/tixiong/datasets/data/grounding/Refcoco_train_grpo/specialToken/specialToken-v2-NUM32_refcocog_train_processed_40k_select-specialToken.jsonl"
RUN_NAME="warmup-v3-specialToken-v2_${MODEL_SHORT}_${DATA_NAME}_epoch${EPOCH}_lr${lr}"
OUTPUT_DIR=${WORKDIR}/outputs/sft-qwen2p5vl/${DATA_NAME}/${RUN_NAME}


torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/r1-v/src/open_r1/sft_new_v3_specialToken-v2_softReward_qwen2p5vl.py \
    --deepspeed src/r1-v/local_scripts/zero3.json \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path $MODEL_PATH \
    --image_folder ${IMAGE_FOLDER} \
    --dataset_name $DATA_PATH \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --learning_rate ${lr} \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --num_train_epochs $EPOCH \
    --run_name $RUN_NAME \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 250 \
    --save_only_model true \
    --logging_steps 5 \
    --logging_strategy steps 

checkpoints=$(ls $OUTPUT_DIR)

for ckpt in $checkpoints; do
    if [[ ! $ckpt == checkpoint-* ]]; then
        continue
    fi
    OUTPUT_CKPT=${OUTPUT_DIR}/${ckpt}

    cp ${OUTPUT_DIR}/preprocessor_config.json $OUTPUT_CKPT/.
    cp ${OUTPUT_DIR}/chat_template.json $OUTPUT_CKPT/.
done

# bash /home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/src/scripts/xty/eval/eval_qwen2VL_superclevr_group.sh $OUTPUT_DIR

# bash /home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/src/scripts/xty/infer_tallyQA/eval_qwen2VL_tallyQA-think_group.sh $OUTPUT_DIR

# bash /home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/src/scripts/xty/infer_pixmo-points-counting/eval_qwen2VL_pixmo-point-counting_group.sh $OUTPUT_DIR