# -*- coding: utf-8 -*-
"""
@Time ： 2024/7/11 15:16
@Auth ： fangsongtao
@File ：rank.py
@IDE ：PyCharm
"""

import json
from tqdm import tqdm
import random
from openai import OpenAI

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
llama3_70b_openai_api_base = "http://10.167.193.30:8084/v1"


def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_rank(prompt):
    client = OpenAI(
        api_key=openai_api_key,
        base_url=llama3_70b_openai_api_base,
    )
    chat_response = client.chat.completions.create(
        model="data/llama3_70b",
        messages=[
            {"role": "user", "content": prompt},
        ],
        max_tokens=1000,
        temperature=0,
    )
    return chat_response


if __name__ == '__main__':
    data = read_json("806_predict.json")
    data_final = []
    for item in tqdm(data):
        prompt = item["instruction"]
        try:
           prompt = prompt + "现在已有三个答案，分别是：答案1："+item["predict"][0]+ "答案2："+ item["predict"][1]+"答案3：" + \
           item["predict"][2]+"请根据上述信息对答案进行排序，并给出排序结果，你只需返回排序结果，如\"1,2,3\"，不要返回其他信息。"
           result = get_rank(prompt)
           item["rank"] = [item["predict"][int(result.choices[0].message.content.split(",")[0])],
                           item["predict"][int(result.choices[0].message.content.split(",")[1])],
                           item["predict"][int(result.choices[0].message.content.split(",")[2])]]
        except Exception as e:
            print(e)
            item["rank"] = "error:"+str(e)
            data_final.append(item)
            continue
        data_final.append(item)
    with open("806_rank.json", 'w', encoding='utf-8') as f:
        json.dump(data_final, f, ensure_ascii=False, indent=4)