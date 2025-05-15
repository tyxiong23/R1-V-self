NUM_CHUNKS=8
MAX_TOKENS=2048


cd /home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/src/eval


# JUDGE_MODE="score"
MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_SHORT=$(basename "$MODEL_PATH")
# OUTPUT_BASE="/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/eval/${MODEL_SHORT}/eval-direct_tallyQA_test_complex_subset_1.5k"
# QUESTION_FILE="/home/tixiong/storage/datasets/multimodal-r1/tallyQA/tallyQA_test_complex_subset_1.5k.jsonl"
# IMAGE_FOLDER="/home/tixiong/storage/xty-workspace/multimodal-reasoning/datasets/tallyQA/images"
# IMAGE_KEY='image'
TASK="pixmo-count_test"
OUTPUT_BASE="/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/eval/${MODEL_SHORT}/eval-direct_${TASK}"
QUESTION_FILE="/home/tixiong/storage/xty-workspace/multimodal-reasoning/datasets/pixmo-counts/${TASK}.jsonl"
IMAGE_FOLDER="/home/tixiong/storage/xty-workspace/multimodal-reasoning/datasets/pixmo-counts"
IMAGE_KEY='image_path'

output_file_merge="${OUTPUT_BASE}/all_merge.jsonl"

echo output_file_merge $output_file_merge
echo model_path $MODEL_PATH

BATCH_SIZE=32

for chunk in $(seq 0 $((NUM_CHUNKS-1))); do
    echo $chunk
    CUDA_VISIBLE_DEVICES=${chunk} \
        python3 /home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/src/eval/inference_qwen2vl_counting_direct.py \
        --model-path $MODEL_PATH \
        --image-folder $IMAGE_FOLDER \
        --question-file $QUESTION_FILE \
        --answers-file "${OUTPUT_BASE}/split/${NUM_CHUNKS}_${chunk}.jsonl" \
        --max_new_tokens $MAX_TOKENS \
        --num-chunks $NUM_CHUNKS \
        --chunk-idx $chunk \
        --image_key $IMAGE_KEY \
        --answer_key "count" \
        --temperature 0.1 \
        --batch_size 4 &
done

wait

# Clear out the output file if it exists.
> "$output_file_merge"

# Loop through the indices and concatenate each file.
for chunk in $(seq 0 $((NUM_CHUNKS-1))); do
    cat "${OUTPUT_BASE}/split/${NUM_CHUNKS}_${chunk}.jsonl" >> "$output_file_merge"
done

python /home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/src/eval/calc_result.py $OUTPUT_BASE

# rm -r "${OUTPUT_BASE}/split"