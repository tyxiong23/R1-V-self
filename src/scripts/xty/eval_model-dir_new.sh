model_dir=$1

bash /home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/src/scripts/xty/eval/eval_qwen2VL_superclevr_new_group.sh $model_dir

bash /home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/src/scripts/xty/infer_pixmo-points-counting/eval_qwen2VL_pixmo-point-counting_new_group.sh $model_dir

bash /home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/src/scripts/xty/infer_tallyQA/eval_qwen2VL_tallyQA-think_new_group.sh $model_dir