RP=${RP:-0}

LOG_ROOT="/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/logs"

MODEL_DIRS=(
    '/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/warmup/warmup_gemini_250310-v1_tallyQA/warmup_v1_Qwen2-VL-2B_warmup_gemini_250310-v1_tallyQA_epoch3_lr1e-5'
    '/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/warmup/warmup_gemini_250310-v1_tallyQA/warmup_v1_Qwen2-VL-2B-Instruct_warmup_gemini_250310-v1_tallyQA_epoch3_lr1e-5'
    '/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/warmup/warmup_gemini_250310-v1-total_tallyQA/warmup_v1_Qwen2-VL-2B_warmup_gemini_250310-v1-total_tallyQA_epoch3_lr1e-5'
    '/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/warmup/warmup_gemini_250310-v1-total_tallyQA/warmup_v1_Qwen2-VL-2B-Instruct_warmup_gemini_250310-v1-total_tallyQA_epoch3_lr1e-5'
    '/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/training_v2p5_consistency-wamrup/no_consistency/pixmo-point-counting_filterV1_train/Qwen2-VL-2B_pixmo-point-counting_filterV1_train_maxlen512_numgen4_epoch2__warmup-v1-250310-tallyQA-epoch2'
    '/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/training_v2p5_consistency-wamrup/no_consistency/pixmo-point-counting_filterV1_train/Qwen2-VL-2B-Instruct_pixmo-point-counting_filterV1_train_maxlen512_numgen4_epoch2__warmup-v1-250310-tallyQA-epoch2'
    '/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/training_v2p5_consistency-wamrup/no_consistency/tallyQA_filterV1_len7_amt_vg_vqa_qwenVL2-2B/Qwen2-VL-2B_tallyQA_filterV1_len7_amt_vg_vqa_qwenVL2-2B_maxlen512_numgen8_epoch2__warmup-v1-250310-tallyQA-epoch2'
    '/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/training_v2p5_consistency-wamrup/no_consistency/tallyQA_filterV1_len7_amt_vg_vqa_qwenVL2-2B/Qwen2-VL-2B-Instruct_tallyQA_filterV1_len7_amt_vg_vqa_qwenVL2-2B_maxlen512_numgen8_epoch2__warmup-v1-250310-tallyQA-epoch2'
    '/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/training_v2p5_consistency-wamrup/pixmo-point-counting_filterV1_train/Qwen2-VL-2B_pixmo-point-counting_filterV1_train_maxlen512_numgen8_epoch2_formatAlpha1.0_consistency__warmup-v1-250310-tallyQA-epoch2'
    '/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/training_v2p5_consistency-wamrup/pixmo-point-counting_filterV1_train/Qwen2-VL-2B-Instruct_pixmo-point-counting_filterV1_train_maxlen512_numgen8_epoch2_formatAlpha1.0_consistency__warmup-v1-250310-tallyQA-epoch2'
    '/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/training_v2p5_consistency-wamrup/tallyQA_filterV1_len7_amt_vg_vqa_qwenVL2-2B/Qwen2-VL-2B_tallyQA_filterV1_len7_amt_vg_vqa_qwenVL2-2B_maxlen512_numgen4_epoch2_formatAlpha1.0_consistency__warmup-v1-250310-tallyQA-epoch2'
    '/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/training_v2p5_consistency-wamrup/tallyQA_filterV1_len7_amt_vg_vqa_qwenVL2-2B/Qwen2-VL-2B-Instruct_tallyQA_filterV1_len7_amt_vg_vqa_qwenVL2-2B_maxlen512_numgen8_epoch2_formatAlpha1.0_consistency__warmup-v1-250310-tallyQA-epoch2'
    '/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/training_v2p6_warmup-total/no_consistency/Clevr_CoGenT_TrainA_70K_Complex/Qwen2-VL-2B-Instruct_Clevr_CoGenT_TrainA_70K_Complex_maxlen512_numgen8_epoch2__warmup-v1-total-250310-tallyQA-epoch1'
    '/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/training_v2p6_warmup-total/no_consistency/pixmo-point-counting_filterV1_train/Qwen2-VL-2B_pixmo-point-counting_filterV1_train_maxlen512_numgen8_epoch2__warmup-v1-total-250310-tallyQA-epoch1'
    '/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/training_v2p6_warmup-total/no_consistency/pixmo-point-counting_filterV1_train/Qwen2-VL-2B-Instruct_pixmo-point-counting_filterV1_train_maxlen512_numgen8_epoch2__warmup-v1-total-250310-tallyQA-epoch1'
    '/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/training_v2p6_warmup-total/no_consistency/tallyQA-filterV0-warmup-holdout/Qwen2-VL-2B_tallyQA-filterV0-warmup-holdout_maxlen512_numgen8_epoch2__warmup-v1-total-250310-tallyQA-epoch1'
    '/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/training_v2p6_warmup-total/no_consistency/tallyQA-filterV0-warmup-holdout/Qwen2-VL-2B-Instruct_tallyQA-filterV0-warmup-holdout_maxlen512_numgen8_epoch2__warmup-v1-total-250310-tallyQA-epoch1'
)

group_script="/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/src/scripts/xty/eval/eval_qwen2VL_superclevr_group.sh"

ls $group_script

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
