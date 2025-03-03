RP=${RP:-0}

LOG_ROOT="/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/logs"

MODEL_DIRS=(
    '/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/training_v2/pixmo-point-counting_filterV1_train/checkpoint-300_pixmo-point-counting_filterV1_train_maxlen512_numgen8_epoch2_formatAlpha1.0_lr1e-6_pretrain-CLEVR-300iter' 
    '/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/training_v2/pixmo-point-counting_filterV1_train/checkpoint-300_pixmo-point-counting_filterV1_train_maxlen1024_numgen4_epoch2_formatAlpha1.0_lr1e-6_pretrain-CLEVR-complex-300iter' 
    '/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/training_v2/tallyQA_filterV1_len7_amt_vg_vqa_qwenVL2-2B/checkpoint-300_tallyQA_filterV1_len7_amt_vg_vqa_qwenVL2-2B_maxlen512_numgen8_epoch2_formatAlpha1.0_pretrain-clevr-iter300' 
    '/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/training_v2_base/tallyQA_filterV1_len7_amt_vg_vqa_qwenVL2-2B/Qwen2-VL-2B_tallyQA_filterV1_len7_amt_vg_vqa_qwenVL2-2B_maxlen512_numgen8_epoch2_formatAlpha1.0' 
    '/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/training_v2_base/pixmo-point-counting_filterV1_train/Qwen2-VL-2B_pixmo-point-counting_filterV1_train_maxlen512_numgen8_epoch2_formatAlpha1.0'
)

group_script="/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/src/scripts/xty/infer_pixmo-points-counting/eval_qwen2VL_pixmo-point-counting_group.sh"

for model in ${MODEL_DIRS[@]}; do
    echo model $model
    bash /home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/src/scripts/xty/submit_job_short.sh "bash $group_script $model"
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
