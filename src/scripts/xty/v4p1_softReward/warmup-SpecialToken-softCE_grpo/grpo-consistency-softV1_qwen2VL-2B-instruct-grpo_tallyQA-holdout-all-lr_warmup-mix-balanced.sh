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
DATA_PATH="/lustre/fsw/portfolios/${PORTOFOLIOS}/users/tixiong/xty-workspace/multimodal-reasoning/datasets/tallyQA/splits/250310-gemini-v0/parquet/tallyQA-train-filterV1_no-gemini-distilled_total"
# IMAGE_FOLDER="/home/tixiong/storage/xty-workspace/multimodal-reasoning/datasets/tallyQA/images"

WARMUP_LR=1e-5
MODEL_PATH="/lustre/fsw/portfolios/${PORTOFOLIOS}/users/tixiong/xty-workspace/multimodal-reasoning/R1-V/outputs/v4-grpo-softReward/warmup_specialToken_softReward/warmup_gemini-v0-mix-balanced-consistent-COUNT30-13k/warmup-specialToken-softReward-v1-lambda1.0_Qwen2-VL-2B-Instruct-Count-SpecialToken-NUM30_warmup_gemini-v0-mix-balanced-consistent-COUNT30-13k_epoch1_lr${WARMUP_LR}/checkpoint-842"
MODEL_SHORT="Qwen2-VL-2B-Instruct-CountToken-NUM30"
EPOCH=1
MAX_COMPLETION_LEN=512
NUM_GEN=8
LR=$1
FORMAT_REWARD_ALPHA=${FORMAT_REWARD_ALPHA:-1.0}
RUN_NAME="grpo-consistency-softV1_${MODEL_SHORT}_${DATA_NAME}_maxlen${MAX_COMPLETION_LEN}_numgen${NUM_GEN}_epoch${EPOCH}_lr${LR}_formatAlpha${FORMAT_REWARD_ALPHA}_consistency_warmup-gemini-v0-mix-balanced-epoch1-lr${WARMUP_LR}-SpecialToken-SoftCE"
OUTPUT_DIR=${WORKDIR}/outputs/v4-grpo-softReward/warmup-specialToken-SoftCE-mix-balanced-lr${WARMUP_LR}+grpo-consistency-softV1/${DATA_NAME}/${RUN_NAME}

export LOG_PATH=$(dirname "$OUTPUT_DIR")"/${RUN_NAME}.log"
export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL

echo LR $LR
echo OUTPUT_DIR $OUTPUT_DIR

# MASTER_ADDR=$(echo "$SLURM_NODELIST" | sed -E 's/(.+)\[(.+),.*\]/\1\2/')
# MASTER_ADDR=$SLURM_SRUN_COMM_HOST
MASTER_ADDR=$(echo "$SLURM_NODELIST" | \
  sed -E 's/^([^[]+)\[([0-9]+)([-0-9,]*)\]$/\1\2/' | \
  sed 's/,.*//')

echo SLURM_NODELIST $SLURM_NODELIST
echo NODEID $SLURM_NODEID
echo MASTER_ADDR $MASTER_ADDR

torchrun --nproc_per_node="8" \
    --nnodes="$SLURM_NNODES" \
    --node_rank="$SLURM_NODEID" \
    --master_addr="$MASTER_ADDR" \
    --master_port="29505" \
    src/open_r1/grpo_consistency_soft.py \
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


bash /lustre/fsw/portfolios/${PORTOFOLIOS}/users/tixiong/xty-workspace/multimodal-reasoning/R1-V/src/scripts/xty/eval_model-dir.sh $OUTPUT_DIR

# srun --container-image=/home/tixiong/storage/docker/r1-V.sqsh \
#      --container-mounts=/lustre \
#      bash /lustre/fsw/portfolios/nvr/users/tixiong/xty-workspace/multimodal-reasoning/R1-V/src/scripts/xty/v4p1_softReward/warmup-SpecialToken-softCE_grpo/grpo-consistency-softV1_qwen2VL-2B-instruct-grpo_tallyQA-holdout-all-lr_warmup-mix-balanced.sh 1e-6