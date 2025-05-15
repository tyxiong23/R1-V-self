NUM_CHUNKS=8
MAX_TOKENS=2048

IMAGE_FOLDER="/home/tixiong/storage/xty-workspace/multimodal-reasoning"

cd /home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/src/eval


# JUDGE_MODE="score"
# MODEL_PATH="Qwen/Qwen2-VL-2B-Instruct"
# MODEL_SHORT=$(basename "$MODEL_PATH")
# OUTPUT_BASE="/home/tixiong/storage/xty-workspace/multimodal-reasoning/R1-V/outputs/inference/pixmo-points-counting-filterV1-rand1k/${MODEL_SHORT}/think_then_answer"
MODEL_PATH=$1
OUTPUT_BASE=$2

QUESTION_FILE="/home/tixiong/storage/xty-workspace/multimodal-reasoning/datasets/pixmo-points/pixmo_points_counting_filterV1_rand1k.jsonl"

output_file_merge="${OUTPUT_BASE}/all_merge.jsonl"

echo output_file_merge $output_file_merge
echo model_path $MODEL_PATH

BATCH_SIZE=8

for chunk in $(seq 0 $((NUM_CHUNKS-1))); do
    echo $chunk
    CUDA_VISIBLE_DEVICES=${chunk} \
        python3 inference_qwen2vl_counting_direct.py \
        --model-path $MODEL_PATH \
        --image-folder $IMAGE_FOLDER \
        --question-file $QUESTION_FILE \
        --answers-file "${OUTPUT_BASE}/split/${NUM_CHUNKS}_${chunk}.jsonl" \
        --max_new_tokens $MAX_TOKENS \
        --num-chunks $NUM_CHUNKS \
        --chunk-idx $chunk \
        --image_key "image_path" \
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

# rm -r "${OUTPUT_BASE}/split"