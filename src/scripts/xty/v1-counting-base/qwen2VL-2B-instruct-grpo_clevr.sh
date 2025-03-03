export WANDB_API_KEY="0bba0b55ec140e44f4519b124800b91b6cbf146f"
export WANDB_ENTITY=xty620682
export WANDB_MODE=online
export WANDB_PROJECT=R1-V

PORTOFOLIOS="nvr"
export HF_HOME="/lustre/fsw/portfolios/${PORTOFOLIOS}/users/tixiong/cache/huggingface"


WORKDIR="/lustre/fsw/portfolios/${PORTOFOLIOS}/users/tixiong/xty-workspace/multimodal-reasoning/R1-V"
cd ${WORKDIR}/src/r1-v

DATA_NAME="clevr_cogen_a_train"
DATA_PATH="/lustre/fsw/portfolios/${PORTOFOLIOS}/users/tixiong/datasets/multimodal-r1/${DATA_NAME}"
MODEL_PATH="/lustre/fsw/portfolios/${PORTOFOLIOS}/users/tixiong/checkpoints/Qwen2-VL-2B"
MODEL_SHORT=$(basename "$MODEL_PATH")
EPOCH=2
MAX_COMPLETION_LEN=512
NUM_GEN=8
RUN_NAME="${MODEL_SHORT}_${DATA_NAME}_maxlen${MAX_COMPLETION_LEN}_numgen${NUM_GEN}_epoch${EPOCH}"
OUTPUT_DIR=${WORKDIR}/outputs/training_v1_base/${DATA_NAME}/${RUN_NAME}

export LOG_PATH=$(dirname "$OUTPUT_DIR")"/${RUN_NAME}.log"
export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL

echo OUTPUT_DIR $OUTPUT_DIR

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo.py \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path $MODEL_PATH \
    --dataset_name $DATA_PATH \
    --deepspeed local_scripts/zero3.json \
    --max_completion_length $MAX_COMPLETION_LEN \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs $EPOCH \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true \
    --num_generations $NUM_GEN   # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance  

#     --max_prompt_length 512 \