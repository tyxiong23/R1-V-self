export WANDB_API_KEY="0bba0b55ec140e44f4519b124800b91b6cbf146f"
export WANDB_ENTITY=xty620682
export WANDB_MODE=online
export WANDB_PROJECT=R1-V

PORTOFOLIOS="nvr"
export HF_HOME="/lustre/fsw/portfolios/${PORTOFOLIOS}/users/tixiong/cache/huggingface"

WORKDIR="/lustre/fsw/portfolios/${PORTOFOLIOS}/users/tixiong/xty-workspace/multimodal-reasoning/R1-V"
cd ${WORKDIR}
# cd ${WORKDIR}/src/r1-v

IMAGE_FOLDER="/home/tixiong/storage/xty-workspace/multimodal-reasoning/datasets/tallyQA/images"

MODEL_PATH="Qwen/Qwen2-VL-2B-Instruct"
MODEL_SHORT=$(basename "$MODEL_PATH")
EPOCH=1
lr=1e-6

DATA_NAME="warmup_gemini_250310-v0_tallyQA"
DATA_PATH="/home/tixiong/storage/xty-workspace/multimodal-reasoning/datasets/tallyQA/splits/250310-gemini-v0/250310_v0_gemini-2.0-flash-thinking-exp_consistent.jsonl"
RUN_NAME="warmup_v0_${MODEL_SHORT}_${DATA_NAME}_epoch${EPOCH}_lr${lr}"
OUTPUT_DIR=${WORKDIR}/outputs/warmup/${DATA_NAME}/${RUN_NAME}


ACCELERATE_LOG_LEVEL=info \
accelerate launch --config_file src/r1-v/configs/zero3.yaml \
    src/r1-v/src/open_r1/sft_new.py \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path $MODEL_PATH \
    --image_folder ${IMAGE_FOLDER} \
    --dataset_name $DATA_PATH \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
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
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_only_model true \
    --logging_steps 5 \
    --logging_strategy steps 

bash /home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/src/scripts/xty/eval/eval_qwen2VL_superclevr_group.sh $OUTPUT_DIR

bash /home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/src/scripts/xty/infer_tallyQA/eval_qwen2VL_tallyQA-think_group.sh $OUTPUT_DIR

bash /home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/src/scripts/xty/infer_pixmo-points-counting/eval_qwen2VL_pixmo-point-counting_group.sh $OUTPUT_DIR