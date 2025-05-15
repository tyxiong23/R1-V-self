import sys
import json
import os

from glob import glob

import json


def load_jsonl(save_path):
    with open(save_path, "r") as f:
        data = []
        error_count = 0
        for line in f.readlines():
            try:
                data.append(json.loads(line))
            except:
                error_count += 1
                pass
        if error_count > 0:
            print('error_count', error_count)       
                # print("error", line)
    return data


def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def load_data(data_path):
    if "jsonl" in data_path:
        data_list = load_jsonl(data_path)
    else:
        data_list = load_json(data_path)
    return data_list


def cal_accuracy(path):
    dataset = load_data(path)
    acc = sum([int(i['answer_match']) for i in dataset]) / len(dataset)
    return round(acc * 100, 2)

if __name__ == "__main__":
    model_dir = sys.argv[1]
    print(model_dir)

    in_jsonl = os.path.join(model_dir, 'all_merge.jsonl')
    out_json = os.path.join(model_dir, 'accuracy.json')

    result_json = {
        "Accuracy": cal_accuracy(in_jsonl)
    }

    print(result_json)

    with open(out_json, 'w') as f:
        json.dump(result_json, f, indent=2)