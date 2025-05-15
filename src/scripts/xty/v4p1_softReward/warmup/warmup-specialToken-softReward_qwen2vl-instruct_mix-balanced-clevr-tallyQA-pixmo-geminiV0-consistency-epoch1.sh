export WANDB_API_KEY="0bba0b55ec140e44f4519b124800b91b6cbf146f"
export WANDB_ENTITY=xty620682
export WANDB_MODE=online
export WANDB_PROJECT=R1-V

PORTOFOLIOS="nvr"
export HF_HOME="/lustre/fsw/portfolios/${PORTOFOLIOS}/users/tixiong/cache/huggingface"

WORKDIR="/lustre/fsw/portfolios/${PORTOFOLIOS}/users/tixiong/xty-workspace/multimodal-reasoning/R1-V"
cd ${WORKDIR}
# cd ${WORKDIR}/src/r1-v

IMAGE_FOLDER="/home/tixiong/storage/xty-workspace/multimodal-reasoning/datasets"

MODEL_PATH="/home/tixiong/storage/checkpoints/Qwen2-VL-2B-Instruct-Count-SpecialToken-NUM30"
MODEL_SHORT=$(basename "$MODEL_PATH")
EPOCH=1
lr=$1

DATA_NAME="warmup_gemini-v0-mix-balanced-consistent-COUNT30-13k"
DATA_PATH="/home/tixiong/storage/xty-workspace/multimodal-reasoning/datasets/pixmo-points/warmup-mix-balanced-Count30-special-Token_clevr-complex+tallyQA+pixmo.jsonl"
REWARD_FUNC_VERSION='v1'
COUNT_TOKEN_LAMBDA=1.0
RUN_NAME="warmup-specialToken-softReward-${REWARD_FUNC_VERSION}-lambda${COUNT_TOKEN_LAMBDA}_${MODEL_SHORT}_${DATA_NAME}_epoch${EPOCH}_lr${lr}"
OUTPUT_DIR=${WORKDIR}/outputs/v4-grpo-softReward/warmup_specialToken_softReward/${DATA_NAME}/${RUN_NAME}


# ACCELERATE_LOG_LEVEL=info \
# accelerate launch --config_file src/r1-v/configs/zero3.yaml \
torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/r1-v/src/open_r1/sft_new_v2_specialToken_softReward.py \
    --output_dir ${OUTPUT_DIR} \
    --deepspeed src/r1-v/local_scripts/zero3.json \
    --model_name_or_path $MODEL_PATH \
    --image_folder ${IMAGE_FOLDER} \
    --dataset_name $DATA_PATH \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --learning_rate ${lr} \
    --warmup_ratio 0.05 \
    --reward_func $REWARD_FUNC_VERSION \
    --count_token_ce_lambda $COUNT_TOKEN_LAMBDA \
    --lr_scheduler_type cosine \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --num_train_epochs $EPOCH \
    --run_name $RUN_NAME \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
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

bash /home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/src/scripts/xty/eval/eval_qwen2VL_superclevr_group.sh $OUTPUT_DIR

bash /home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/src/scripts/xty/infer_tallyQA/eval_qwen2VL_tallyQA-think_group.sh $OUTPUT_DIR

bash /home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/src/scripts/xty/infer_pixmo-points-counting/eval_qwen2VL_pixmo-point-counting_group.sh $OUTPUT_DIR

bash /home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/src/scripts/xty/eval_model-dir_new.sh $OUTPUT_DIR