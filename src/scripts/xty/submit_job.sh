RP=${RP:-0}

LOG_ROOT="/lustre/fsw/portfolios/llmservice/users/tixiong/xty-workspace/multimodal-reasoning/R1-V/outputs/logs"

SCRIPT=$1


submit_job \
    --gpu 8 \
    --tasks_per_node 1 \
    --nodes 1 \
    -n mm-reasoning \
    --image=/home/tixiong/storage/docker/r1-V.sqsh \
    --logroot $LOG_ROOT \
    --email_mode never \
    --dependent_clones $RP\
    --duration 4 \
    --cpu 128 \
    --partition batch \
    -c "bash $SCRIPT"