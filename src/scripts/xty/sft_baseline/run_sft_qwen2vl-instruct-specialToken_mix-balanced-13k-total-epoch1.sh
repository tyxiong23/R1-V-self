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

DATA_NAME="mixed-balanced-COUNT30-13k"
DATA_PATH="/home/tixiong/storage/xty-workspace/multimodal-reasoning/datasets/pixmo-points/warmup-mix-balanced-Count30-special-Token_clevr-complex+tallyQA+pixmo_SFT.jsonl"
RUN_NAME="sft-baseline-v2_${MODEL_SHORT}_${DATA_NAME}_epoch${EPOCH}_lr${lr}"
OUTPUT_DIR=${WORKDIR}/outputs/sft_baseline/${DATA_NAME}/${RUN_NAME}


ACCELERATE_LOG_LEVEL=info \
accelerate launch --config_file src/r1-v/configs/zero3.yaml \
    src/r1-v/src/open_r1/sft_baseline_v2.py \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path $MODEL_PATH \
    --image_folder ${IMAGE_FOLDER} \
    --dataset_name $DATA_PATH \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --answer_key "answer_specialToken" \
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
    --save_steps 100 \
    --save_only_model true \
    --logging_steps 5 \
    --logging_strategy steps 

    # --save_strategy "epoch" \
    # --save_total_limit 4 \

checkpoints=$(ls $OUTPUT_DIR)

for ckpt in $checkpoints; do
    if [[ ! $ckpt == checkpoint-* ]]; then
        continue
    fi
    OUTPUT_CKPT=${OUTPUT_DIR}/${ckpt}

    cp ${OUTPUT_DIR}/preprocessor_config.json $OUTPUT_CKPT/.
    cp ${OUTPUT_DIR}/chat_template.json $OUTPUT_CKPT/.
done

bash /home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/src/scripts/xty/eval_model-dir_direct.sh $OUTPUT_DIR
