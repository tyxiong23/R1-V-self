export WANDB_API_KEY="0bba0b55ec140e44f4519b124800b91b6cbf146f"
export WANDB_ENTITY=xty620682
export WANDB_MODE=online
export WANDB_PROJECT=R1-V

PORTOFOLIOS="nvr"
export HF_HOME="/lustre/fsw/portfolios/${PORTOFOLIOS}/users/tixiong/cache/huggingface"


WORKDIR="/lustre/fsw/portfolios/${PORTOFOLIOS}/users/tixiong/xty-workspace/multimodal-reasoning/R1-V"
cd ${WORKDIR}/src/r1-v

DATA_NAME="tallyQA_filterV1_geminiV0-holdout-all"
# DATA_PATH="/home/tixiong/storage/xty-workspace/multimodal-reasoning/datasets/tallyQA/splits/250310-gemini-v0/tallyQA-train-filterV1_no-gemini-distilled_total.jsonl"
DATA_PATH="/home/tixiong/storage/xty-workspace/multimodal-reasoning/datasets/tallyQA/splits/250310-gemini-v0/parquet/tallyQA-train-filterV1_no-gemini-distilled_total"
# IMAGE_FOLDER="/home/tixiong/storage/xty-workspace/multimodal-reasoning/datasets/tallyQA/images"

MODEL_PATH="/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/warmup/warmup_gemini-v0-mix-balanced-consistent-13k-Count30/warmup-new_v0_Qwen2-VL-7B-Instruct_warmup_gemini-v0-mix-balanced-consistent-13k-Count30_epoch1_lr1e-6/checkpoint-842"
MODEL_SHORT="Qwen2-VL-7B-Instruct"
EPOCH=1
MAX_COMPLETION_LEN=512
NUM_GEN=4
LR=$1
FORMAT_REWARD_ALPHA=${FORMAT_REWARD_ALPHA:-1.0}
RUN_NAME="grpo_${MODEL_SHORT}_${DATA_NAME}_maxlen${MAX_COMPLETION_LEN}_numgen${NUM_GEN}_epoch${EPOCH}_lr${LR}_formatAlpha${FORMAT_REWARD_ALPHA}_warmup-gemini-v0-mix-balanced-epoch1-count30"
OUTPUT_DIR=${WORKDIR}/outputs/v3p8_warmup-mix-balanced/warmup+grpo/${DATA_NAME}/${RUN_NAME}

export LOG_PATH=$(dirname "$OUTPUT_DIR")"/${RUN_NAME}.log"
export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL

echo LR $LR
echo OUTPUT_DIR $OUTPUT_DIR

# MASTER_ADDR=$(echo "$SLURM_NODELIST" | sed -E 's/(.+)\[(.+),.*\]/\1\2/')
MASTER_ADDR=$(echo "$SLURM_NODELIST" | \
  sed -E 's/^([^[]+)\[([0-9]+)([-0-9,]*)\]$/\1\2/' | \
  sed 's/,.*//')
# MASTER_ADDR="127.0.0.1"

# MASTER_ADDR=$SLURM_SRUN_COMM_HOST

echo SLURM_NODELIST $SLURM_NODELIST
echo NODEID $SLURM_NODEID
echo MASTER_ADDR $MASTER_ADDR

torchrun --nproc_per_node="8" \
    --nnodes="$SLURM_NNODES" \
    --node_rank="$SLURM_NODEID" \
    --master_addr="$MASTER_ADDR" \
    --master_port="29505" \
    src/open_r1/grpo.py \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path $MODEL_PATH \
    --dataset_name $DATA_PATH \
    --learning_rate $LR \
    --deepspeed local_scripts/zero3.json \
    --max_completion_length $MAX_COMPLETION_LEN \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs $EPOCH \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true \
    --format_reward_alpha $FORMAT_REWARD_ALPHA \
    --num_generations $NUM_GEN   # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance  

#     --max_prompt_length 512 \