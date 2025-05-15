NUM_CHUNKS=8
MAX_TOKENS=2048

IMAGE_FOLDER="/home/tixiong/storage/xty-workspace/multimodal-reasoning/datasets/tallyQA/images"

cd /home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/src/eval


# JUDGE_MODE="score"
MODEL_PATH="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_PATH="/home/tixiong/storage/checkpoints/Qwen2-VL-2B"
MODEL_SHORT=$(basename "$MODEL_PATH")
OUTPUT_BASE="/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/inference/tallyQA_test_complex_subset1.5k/${MODEL_SHORT}/think_then_answer"
QUESTION_FILE="/home/tixiong/storage/datasets/multimodal-r1/tallyQA/tallyQA_test_complex_subset_1.5k.jsonl"

output_file_merge="${OUTPUT_BASE}/all_merge.jsonl"

echo output_file_merge $output_file_merge
echo model_path $MODEL_PATH

BATCH_SIZE=32

for chunk in $(seq 0 $((NUM_CHUNKS-1))); do
    echo $chunk
    CUDA_VISIBLE_DEVICES=${chunk} \
        python3 inference_qwen2vl_counting_thinking.py \
        --model-path $MODEL_PATH \
        --image-folder $IMAGE_FOLDER \
        --question-file $QUESTION_FILE \
        --answers-file "${OUTPUT_BASE}/split/${NUM_CHUNKS}_${chunk}.jsonl" \
        --max_new_tokens $MAX_TOKENS \
        --num-chunks $NUM_CHUNKS \
        --chunk-idx $chunk \
        --image_key 'image' \
        --temperature 0.2 \
        --top_p 0.4 \
        --batch_size 32 &
done

wait

# Clear out the output file if it exists.
> "$output_file_merge"

# Loop through the indices and concatenate each file.
for chunk in $(seq 0 $((NUM_CHUNKS-1))); do
    cat "${OUTPUT_BASE}/split/${NUM_CHUNKS}_${chunk}.jsonl" >> "$output_file_merge"
done

# rm -r "${OUTPUT_BASE}/split"