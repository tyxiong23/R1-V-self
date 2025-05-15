from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import re
from argparse import ArgumentParser
import os

from json_utils import load_data
import numpy as np
import torch
import random
import math


def set_seed(seed=5775709):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def extract_number_answer(output_str):
    # Try to find the number within <answer> tags, if can not find, return None
    answer_pattern = r'<answer>\s*(\d+)\s*</answer>'
    match = re.search(answer_pattern, output_str)
    
    if match:
        return int(match.group(1))

    answer_pattern2 = r'<answer>\s*<c(\d+)/>\s*</answer>'
    match2 = re.search(answer_pattern2, output_str)
    
    if match2:
        return int(match2.group(1))

    if "```" in output_str:
        final_answer = output_str.split("```")[-1].strip()

        if " no " in final_answer:
            return 0
        for idx, word in enumerate([' one ', ' two ', ' three ', ' four ', ' five ', ' six ', ' seven ', ' eight ', ' nine ', ' ten ']):
            if word in final_answer:
                return idx + 1
        
        match = re.search(r'\b\d+\b', final_answer)
        if match:
            try:
                num = int(match.group())
                return num
            except:
                pass


    if 'json' in output_str:
        try:
            json_content_str = output_str.split('```json')[1].split('```')[0].strip()
            json_content = json.loads(json_content_str)
            return len(json_content)
        except Exception as e:
            pass
            # print(f"Error parsing JSON: {e}")

    return None


def get_item_id(item):
    return item['question_id']


QUESTION_TEMPLATE = "Please first output bbox coordinates and names of every item in this image in JSON format, and then answer the question: {Question} First output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."

QUESTION_TEMPLATE_NOTHINKING = "Please first output bbox coordinates and names of every item in this image in JSON format, and then answer {Question}"



def batch_inference(batch, model, processor, args):

    messages = []
    for i in batch:
        if args.no_thinking:
            question = QUESTION_TEMPLATE_NOTHINKING.format(Question=i[args.question_key].lower().replace("?", "."))
        else:
            question = QUESTION_TEMPLATE.format(Question=i[args.question_key])

        message = [{
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": f"{os.path.join(args.image_folder, i[args.image_key])}"
                },
                {
                    "type": "text",
                    "text": question
                }
            ]
        }]
        # print(message)
        messages.append(message)
    
    # Preparation for inference
    text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
        
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    do_sample=True if args.temperature > 0 else False
    temperature=args.temperature
    top_p=args.top_p
    # print(do_sample, temperature, top_p)
    generated_ids = model.generate(
        **inputs, 
        use_cache=True, 
        max_new_tokens=args.max_new_tokens, 
        # do_sample=False,
        do_sample=True if args.temperature > 0 else False,
        temperature=args.temperature,
        top_p=args.top_p
    )
        
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    batch_output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    assert len(batch_output_text) == len(batch)   
    return batch_output_text


def get_model_name(model_path):
    lower = model_path.lower()
    if 'qwen2-vl' in lower:
        return 'qwen2_vl'
    elif 'qwen2.5-vl' in lower:
        return 'qwen2_5_vl'
    else:
        raise NotImplementedError(f"Cannot get model name from: {model_path}")


def main(args):

    MODEL_PATH=args.model_path # Qwen2vl-2b-Instruct for original scores
    BSZ=args.batch_size # reduce it if GPU OOM

    ### load data

    questions_all = load_data(os.path.expanduser(args.question_file))

    # for idx, q in enumerate(questions_all):
    #     q['idx'] = idx

    questions = get_chunk(questions_all, args.num_chunks, args.chunk_idx)
    print(f"load {len(questions)} of {len(questions)} for chunk {args.chunk_idx}")
    
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    
    # load previous datas:
    if os.path.exists(answers_file):
        answers = load_data(answers_file)
        print(f"Already answered: load {len(answers)} of {len(questions)} for chunk {args.chunk_idx} from {answers_file}")
        answer_idxs = set([get_item_id(i) for i in answers])
        new_questions = []
        for q in questions:
            if get_item_id(q) not in answer_idxs:
                new_questions.append(q)
        print(f"After filtering: load {len(new_questions)} of {len(questions)} for chunk {args.chunk_idx}")
        if len(new_questions) == 0:
            return
        questions = new_questions
        ans_file = open(answers_file, "a")
    else:
        ans_file = open(answers_file, "w")

    # load model
    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     MODEL_PATH,
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )
    args.model_name = get_model_name(MODEL_PATH)

    qwen2p5 = False
    if args.model_name == 'qwen2_vl':
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            # device_map="auto",
            device_map="cuda"
        )
    elif args.model_name == 'qwen2_5_vl':
        qwen2p5 = True
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda",
        )
    else:
        raise NotImplementedError

    # default processer
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    if qwen2p5:
        processor.tokenizer.padding_side = "left"


    # Process data in batches
    for i in tqdm(range(0, len(questions), BSZ)):
        batch = questions[i:i + BSZ]
        batch_output_text = batch_inference(batch, model, processor, args)

        for data, output_text in zip(batch, batch_output_text):
            data['pred'] = output_text
            pred_answer = extract_number_answer(output_text)
            data['pred_answer'] = pred_answer
            ground_truth = data[args.answer_key]
            data['answer_match'] = pred_answer is not None and pred_answer == ground_truth

            ans_file.write(json.dumps(data) + "\n")
            ans_file.flush()
        
        
        print(f"Processed batch {i//BSZ + 1}/{(len(questions) + BSZ - 1)//BSZ}")




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default=None)
    parser.add_argument("--question-file", type=str, default=None)
    parser.add_argument("--answers-file", type=str, default=None)
    parser.add_argument('--image_key', type=str, default="image_path")
    parser.add_argument('--question_key', type=str, default="question")
    parser.add_argument('--answer_key', type=str, default="answer")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--no_thinking", action="store_true", default=False)
    parser.add_argument("--top_p", type=float, default=None)
    args = parser.parse_args()

    main(args)





