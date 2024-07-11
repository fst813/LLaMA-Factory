# -*- coding: utf-8 -*-
"""
@Time ： 2024/7/11 15:27
@Auth ： daxuanjie
@File ：dpo_data_trans.py
@IDE ：PyCharm
"""
import json
from tqdm import tqdm

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


if __name__ == '__main__':
    data_final = []
    for item in data:
        good = item["rank"][0]
        bad = item["rank"][-1]
        item["chosen"] = good
        i["rejected"] = bad
        data_final.append(i)
    with open("preference_data.json", "w", encoding="utf-8") as f:
        json.dump(data_final, f, ensure_ascii=False, indent=4)
