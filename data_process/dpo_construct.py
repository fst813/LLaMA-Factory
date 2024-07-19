# -*- coding: utf-8 -*-
"""
@Time ： 2024/7/11 15:09
@Auth ： daxuanjie
@File ：dpo_construct.py
@IDE ：PyCharm
"""

import json
from tqdm import tqdm
import random
from openai import OpenAI

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
llama3_openai_api_base = "http://10.167.193.30:8081/v1"
gemma2_openai_api_base = "http://10.167.193.30:8082/v1"
mistral_openai_api_base = "http://10.167.193.30:8083/v1"


def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def get_infer_llama3(prompt):
    client = OpenAI(
        api_key=openai_api_key,
        base_url=llama3_openai_api_base,
    )
    chat_response = client.chat.completions.create(
        model="data/llama3",
        messages=[
            {"role": "user", "content": prompt},
        ],
        max_tokens=1000,
        temperature=0,
    )
    return chat_response


def get_infer_gemma(prompt):
    client = OpenAI(
        api_key=openai_api_key,
        base_url=gemma2_openai_api_base,
    )
    chat_response = client.chat.completions.create(
        model="data/llama3",
        messages=[
            {"role": "user", "content": prompt},
        ],
        max_tokens=1000,
        temperature=0,
    )
    return chat_response


def get_infer_llama3(prompt):
    client = OpenAI(
        api_key=openai_api_key,
        base_url=mistral_openai_api_base,
    )
    chat_response = client.chat.completions.create(
        model="data/llama3",
        messages=[
            {"role": "user", "content": prompt},
        ],
        max_tokens=1000,
        temperature=0,
    )
    return chat_response


if __name__ == '__main__':
    data = read_json("806.json")
    data_final = []
    for item in tqdm(data):
        prompt = item["instruction"]
        try:
            result_llama3 = get_infer_llama3(prompt)
            result_gemma = get_infer_gemma(prompt)
            result_mistral = get_infer_llama3(prompt)
        except Exception as e:
            print(e)
            item["predict"] = "error:" + str(e)
            data_final.append(item)
            continue
        item["predict"] = [result_llama3.choices[0].message.content, result_gemma.choices[0].message.content,
                           result_mistral.choices[0].message.content]
        data_final.append(item)
    with open("806_predict.json", 'w', encoding='utf-8') as f:
        json.dump(data_final, f, ensure_ascii=False, indent=4)
