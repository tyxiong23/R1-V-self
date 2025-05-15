RP=${RP:-0}

LOG_ROOT="/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/logs"

# MODEL_DIRS=(
#     '/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/training_v2/pixmo-point-counting_filterV1_train/checkpoint-300_pixmo-point-counting_filterV1_train_maxlen512_numgen8_epoch2_formatAlpha1.0_lr1e-6_pretrain-CLEVR-300iter' 
#     '/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/training_v2/pixmo-point-counting_filterV1_train/checkpoint-300_pixmo-point-counting_filterV1_train_maxlen1024_numgen4_epoch2_formatAlpha1.0_lr1e-6_pretrain-CLEVR-complex-300iter' 
#     '/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/training_v2/tallyQA_filterV1_len7_amt_vg_vqa_qwenVL2-2B/checkpoint-300_tallyQA_filterV1_len7_amt_vg_vqa_qwenVL2-2B_maxlen512_numgen8_epoch2_formatAlpha1.0_pretrain-clevr-iter300' 
#     '/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/training_v2_base/tallyQA_filterV1_len7_amt_vg_vqa_qwenVL2-2B/Qwen2-VL-2B_tallyQA_filterV1_len7_amt_vg_vqa_qwenVL2-2B_maxlen512_numgen8_epoch2_formatAlpha1.0' 
#     '/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/training_v2_base/pixmo-point-counting_filterV1_train/Qwen2-VL-2B_pixmo-point-counting_filterV1_train_maxlen512_numgen8_epoch2_formatAlpha1.0'
# )
# MODEL_DIRS=(
#     "/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/training_v2_consistency/tallyQA_filterV1_len7_amt_vg_vqa_qwenVL2-2B/Qwen2-VL-2B-Instruct_tallyQA_filterV1_len7_amt_vg_vqa_qwenVL2-2B_maxlen512_numgen8_epoch2_formatAlpha1.0_consistency"
#     "/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/training_v1_consistency/Clevr_CoGenT_TrainA_70K_Complex/Qwen2-VL-7B-Instruct_Clevr_CoGenT_TrainA_70K_Complex_maxlen1024_numgen4_epoch2_consistency"
#     "/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/training_v2/tallyQA_filterV1_len7_amt_vg_vqa_qwenVL2-2B/Qwen2-VL-2B-Instruct_tallyQA_filterV1_len7_amt_vg_vqa_qwenVL2-2B_maxlen512_numgen8_epoch2_formatAlpha1.0_new"
#     "/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/training_v1_base/Clevr_CoGenT_TrainA_70K_Complex/Qwen2-VL-2B_Clevr_CoGenT_TrainA_70K_Complex_maxlen1024_numgen8_epoch2_consistency"
# )
# MODEL_DIRS=(
#     "/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/training_v2/tallyQA_filterV1_len7_amt_vg_vqa_qwenVL2-2B/Qwen2-VL-2B-Instruct_tallyQA_filterV1_len7_amt_vg_vqa_qwenVL2-2B_maxlen512_numgen8_epoch2_formatAlpha1.0_new"
#     "/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/training_v1_base/Clevr_CoGenT_TrainA_70K_Complex/Qwen2-VL-2B_Clevr_CoGenT_TrainA_70K_Complex_maxlen1024_numgen8_epoch2_consistency"
# )
# MODEL_DIRS=(
#     "/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/training_v2_consistency/pixmo-point-counting_filterV1_train/checkpoint-300_pixmo-point-counting_filterV1_train_maxlen512_numgen8_epoch2_formatAlpha1.0_consistency"
#     "/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/training_v2_consistency/pixmo-point-counting_filterV1_train/Qwen2-VL-2B-Instruct_pixmo-point-counting_filterV1_train_maxlen512_numgen8_epoch2_formatAlpha1.0_consistency"
#     "/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/training_v2_consistency/tallyQA_filterV1_len7_amt_vg_vqa_qwenVL2-2B/checkpoint-300_tallyQA_filterV1_len7_amt_vg_vqa_qwenVL2-2B_maxlen512_numgen8_epoch2_formatAlpha1.0_consistency__pretrain-clevr-complex-iter300"
# )

# MODEL_DIRS=(
# )

MODEL_DIRS=(
    # "/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/warmup/warmup_gemini_250310-v1_tallyQA/warmup_v1_Qwen2-VL-2B-Instruct_warmup_gemini_250310-v1_tallyQA_epoch3_lr1e-5"
    # "/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/warmup/warmup_gemini_250310-v1_tallyQA/warmup_v1_Qwen2-VL-2B_warmup_gemini_250310-v1_tallyQA_epoch3_lr1e-5"
    # "/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/warmup/warmup_gemini_250310-v1-total_tallyQA/warmup_v1_Qwen2-VL-2B_warmup_gemini_250310-v1-total_tallyQA_epoch3_lr1e-5"
    # "/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/warmup/warmup_gemini_250310-v1-total_tallyQA/warmup_v1_Qwen2-VL-2B-Instruct_warmup_gemini_250310-v1-total_tallyQA_epoch3_lr1e-5"
    # "/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/training_v2p5_consistency-wamrup/pixmo-point-counting_filterV1_train/Qwen2-VL-2B-Instruct_pixmo-point-counting_filterV1_train_maxlen512_numgen8_epoch2_formatAlpha1.0_consistency__warmup-v1-250310-tallyQA-epoch2"
    # "/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/training_v2p5_consistency-wamrup/tallyQA_filterV1_len7_amt_vg_vqa_qwenVL2-2B/Qwen2-VL-2B-Instruct_tallyQA_filterV1_len7_amt_vg_vqa_qwenVL2-2B_maxlen512_numgen8_epoch2_formatAlpha1.0_consistency__warmup-v1-250310-tallyQA-epoch2"
    # '/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/training_v2p5_consistency-wamrup/pixmo-point-counting_filterV1_train/Qwen2-VL-2B_pixmo-point-counting_filterV1_train_maxlen512_numgen8_epoch2_formatAlpha1.0_consistency__warmup-v1-250310-tallyQA-epoch2'
    # '/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/training_v2p5_consistency-wamrup/no_consistency/tallyQA_filterV1_len7_amt_vg_vqa_qwenVL2-2B/Qwen2-VL-2B-Instruct_tallyQA_filterV1_len7_amt_vg_vqa_qwenVL2-2B_maxlen512_numgen8_epoch2__warmup-v1-250310-tallyQA-epoch2'
    # '/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/training_v2p5_consistency-wamrup/no_consistency/tallyQA_filterV1_len7_amt_vg_vqa_qwenVL2-2B/Qwen2-VL-2B_tallyQA_filterV1_len7_amt_vg_vqa_qwenVL2-2B_maxlen512_numgen8_epoch2__warmup-v1-250310-tallyQA-epoch2'
    # '/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/training_v2p5_consistency-wamrup/no_consistency/pixmo-point-counting_filterV1_train/Qwen2-VL-2B_pixmo-point-counting_filterV1_train_maxlen512_numgen4_epoch2__warmup-v1-250310-tallyQA-epoch2'
    # '/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/training_v2p5_consistency-wamrup/no_consistency/pixmo-point-counting_filterV1_train/Qwen2-VL-2B-Instruct_pixmo-point-counting_filterV1_train_maxlen512_numgen4_epoch2__warmup-v1-250310-tallyQA-epoch2'
    /home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/training_v2/tallyQA_filterV1_len7_amt_vg_vqa_qwenVL2-2B/checkpoint-300_tallyQA_filterV1_len7_amt_vg_vqa_qwenVL2-2B_maxlen512_numgen8_epoch2_formatAlpha1.0_pretrain-CLEVR-complex-300iter
)

group_script="/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/src/scripts/xty/infer_pixmo-points-counting/eval_qwen2VL_pixmo-point-counting_group.sh"

for model in ${MODEL_DIRS[@]}; do
    echo model $model
    bash /home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/src/scripts/xty/submit_job_short.sh "bash $group_script $model"
    # bash $group_script $model
done


# submit_job \
#     --gpu 8 \
#     --tasks_per_node 1 \
#     --nodes 1 \
#     -n mm-reasoning \
#     --image=/home/tixiong/storage/docker/r1-V.sqsh \
#     --logroot $LOG_ROOT \
#     --email_mode never \
#     --dependent_clones $RP \
#     --duration 2 \
#     --cpu 128 \
#     --partition batch_short \
#     -c "$SCRIPT"
