model_dir=$1

RP=${RP:-1}

echo RP $RP
echo model_dir $model_dir

RP=$RP bash /home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/src/scripts/xty/submit_job.sh "bash /home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/src/scripts/xty/eval_model-dir.sh $model_dir"

RP=$RP bash /home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/src/scripts/xty/submit_job.sh "bash /home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/src/scripts/xty/eval_model-dir_new.sh $model_dir"

# RP=$RP bash /home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/src/scripts/xty/submit_job_short.sh "bash /home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/src/scripts/xty/eval_model-dir.sh $model_dir"

# RP=$RP bash /home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/src/scripts/xty/submit_job_short.sh "bash /home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/src/scripts/xty/eval_model-dir_new.sh $model_dir"