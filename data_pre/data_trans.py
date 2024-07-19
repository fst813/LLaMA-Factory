# -*- coding: utf-8 -*-
"""
@Time ： 2024/7/19 9:45
@Auth ： fangsongtao
@File ：data_trans.py
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
    data = read_jsonl("one_by_one_all.jsonl")
    print(len(data))
    data_final = []
    count = 0
    for i in data:
        if str(i["label"]) != i["messages"][3]["content"]:
            # print(i["label"])
            count += 1
        tmp = {}
        tmp['instruction'] = """You are a computer operation and maintenance expert
I will give you some system logs, you need to analyze these system logs.
The hidden error codes and error categories in these logs are as follows: 1. CPU-related errors, 2. Memory and storage-related errors, 3. Other types of errors.
Multiple errors may occur in these logs, this is because a major error triggered a large number of other errors.
Please identify the main error in these logs and the reason for its occurrence.
Please use concise language for your analysis, no more than 150 words in total. logs:
""" + i["Content"] + "\nPlease give your answer:"
        tmp["input"] = ""
        tmp["output"] = str(i["label"])
        tmp["reason"] = i["messages"][1]["content"]
        tmp["predict_code"] = i["messages"][3]["content"]
        tmp["CaseId"] = i["CaseId"]
        data_final.append(tmp)
    print(len(data_final))
    print(count)
    with open("one_by_one_all_request..json", 'w', encoding='utf-8') as f:
        json.dump(data_final, f, ensure_ascii=False, indent=4)
