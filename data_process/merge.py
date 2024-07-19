# -*- coding: utf-8 -*-
"""
@Time ： 2024/7/11 16:28
@Auth ： daxuanjie
@File ：merge.py
@IDE ：PyCharm
"""
import json

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = []
        for line in f:
            data.append(json.loads(line))
    return data


if __name__ == '__main__':
    data_llama3 = read_json("llama3_predict.json")
    data_gemma2 = read_jsonl("checkpoint-10.jsonl")
    data_mistral = read_jsonl("mistral_predict.jsonl")
    print(len(data_llama3))
    print(len(data_gemma2))
    print(len(data_mistral))
    data_final = []
    for i in range(len(data_llama3)):
        tmp = {}
        tmp["instruction"] = data_llama3[i]["instruction"]
        tmp["label"] = data_llama3[i]["output"]
        tmp["llama3"] = data_llama3[i]["predict"]
        tmp["gemma2"] = data_gemma2[i]["predict"]
        tmp["mistral"] = data_mistral[i]["predict"]
        data_final.append(tmp)
    with open("case_predict.json", "w", encoding="utf-8") as f:
        json.dump(data_final, f, ensure_ascii=False, indent=4)

