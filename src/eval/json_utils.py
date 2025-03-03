
import json

def load_jsonl(save_path):
    with open(save_path, "r") as f:
        data = [json.loads(line) for line in f.readlines()]
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

def save_json(out_path, data):
    with open(out_path, 'w') as outf:
        print(f"output {len(data)} -> {out_path}")
        json.dump(data, outf, indent=4, separators=(',', ': '))
