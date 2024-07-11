# -*- coding: utf-8 -*-
"""
@Time ： 2024/5/20 15:54
@Auth ： daxuanjie
@File ：sft_data_pre.py
@IDE ：PyCharm
"""
# sft数据格式转化
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
    data = read_jsonl("GPT-4O_few shot_62.jsonl")
    print(len(data))
    data_final = []
    for i in data:
        tmp = {}
        tmp['instruction'] = "You are a computer operation and maintenance expert\n" + \
                             i["segments"][0]['text'].split("</s> [INST]")[0].split("[/INST]")[0].\
                                 replace("<s> [INST] ","")+"\nPlease give your answer:"
        tmp["instruction"] = tmp["instruction"].replace("no more than 150 words in total.", "no more than 150 words in total. logs:")
        tmp["input"] = ""
        tmp["output"] = "The error category code is:" + \
                        i["segments"][0]['text'].split("</s> [INST]")[1].split("[/INST]")[1].replace("</s>",
                        ".\nThe reasons are as follows:\n") + i["segments"][0]['text'].split("</s> [INST]")[0].split("[/INST]")[1]
        data_final.append(tmp)
    print(len(data_final))
    with open("train_sft_62.json", 'w', encoding='utf-8') as f:
        json.dump(data_final, f, ensure_ascii=False, indent=4)
