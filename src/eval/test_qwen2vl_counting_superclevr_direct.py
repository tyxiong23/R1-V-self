from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import re
from argparse import ArgumentParser
import os

def extract_number_answer(output_str):
    # Try to find the number within <answer> tags, if can not find, return None
    # answer_pattern = r'<answer>\s*(\d+)\s*</answer>'
    # match = re.search(answer_pattern, output_str)
    
    # if match:
    #     return int(match.group(1))
    # return None

    answer_pattern2 = r'<c(\d+)/>'
    match2 = re.search(answer_pattern2, output_str)
    if match2:
        return int(match2.group(1))
        
    try:
        ans = int(output_str.strip())
        return ans
    except:
        return None


def main(args):

    MODEL_PATH=args.model_path # Qwen2vl-2b-Instruct for original scores
    BSZ=args.batch_size # reduce it if GPU OOM
    # OUTPUT_PATH="./logs/counting_results_superclevr_200_qwen2vl_2b_instruct_grpo_100.json"
    OUTPUT_DIR=args.output_dir

    print("model", MODEL_PATH)
    print('output_dir', OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'result.json')

    PROMPT_PATH="./prompts/superclevr_test200_counting_problems.jsonl"

    #We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    # default processer
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    data = []
    with open(PROMPT_PATH, "r") as f:
        for line in f:
            data.append(json.loads(line))


    QUESTION_TEMPLATE = "{Question} Answer the question using a single number, word or phrase."

    messages = []

    for i in data:
        message = [{
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": f"file://{i['image_path']}"
                },
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(Question=i['question'])
                }
            ]
        }]
        messages.append(message)

    all_outputs = []  # List to store all answers

    # Process data in batches
    for i in tqdm(range(0, len(messages), BSZ)):
        batch_messages = messages[i:i + BSZ]
        
        # Preparation for inference
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=2048, do_sample=False)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        batch_output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        all_outputs.extend(batch_output_text)
        print(f"Processed batch {i//BSZ + 1}/{(len(messages) + BSZ - 1)//BSZ}")


    final_output = []
    correct_number = 0

    for input_example, model_output in zip(data,all_outputs):
        original_output = model_output
        ground_truth = input_example['ground_truth']
        model_answer = extract_number_answer(original_output)
        
        # Create a result dictionary for this example
        result = {
            'question': input_example,
            'ground_truth': ground_truth,
            'model_output': original_output,
            'extracted_answer': model_answer
        }
        final_output.append(result)
        
        # Count correct answers
        if model_answer is not None and model_answer == ground_truth:
            correct_number += 1

    # Calculate and print accuracy
    accuracy = correct_number / len(data) * 100
    print(f"\nAccuracy: {accuracy:.2f}%")

    # Save results to a JSON file
    output_path = OUTPUT_PATH
    with open(output_path, "w") as f:
        json.dump({
            'accuracy': accuracy,
            'results': final_output
        }, f, indent=2)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    main(args)





