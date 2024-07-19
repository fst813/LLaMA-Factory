# -*- coding: utf-8 -*-
"""
@Time ： 2024/7/12 15:00
@Auth ： daxuanjie
@File ：case_format_trans.py
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
    all_corect = 0
    llama3_error = 0
    mistral_error = 0
    gemma_error = 0
    for i in range(len(data_llama3)):
        label = data_llama3[i]["output"]
        pre_llama3 = "1"
        pre_mistral = "1"
        pre_gemma = "1"
        all_error = 0
        list_3 = []
        for j in range(3):
            tmp = {}
            tmp_list = []
            # tmp["instruction"] = data_llama3[i]["instruction"]
            # tmp["label"] = data_llama3[i]["output"]
            # tmp["llama3"] = data_llama3[i]["predict"]
            # tmp["gemma2"] = data_gemma2[i]["predict"]
            # tmp["mistral"] = data_mistral[i]["predict"]

            if j == 0:
                tmp["from"] = "human"
                tmp["value"] = data_llama3[i]["instruction"]
                tmp_list.append(tmp.copy())
                tmp["from"] = "gpt"
                tmp["value"] = data_llama3[i]["predict"].split("The reasons are as follows:\n")[-1]
                tmp_list.append(tmp.copy())
                tmp["from"] = "human"
                tmp["value"] = "The error category code for the main error is:"
                tmp_list.append(tmp.copy())
                tmp["from"] = "gpt"
                tmp["value"] = data_llama3[i]["predict"].split("The reasons are as follows:\n")[0][-3]
                pre_llama3 = data_llama3[i]["predict"].split("The reasons are as follows:\n")[0][-3]
                tmp_list.append(tmp.copy())
                tmp = {}
                tmp["conversations"] = tmp_list

                list_3.append(tmp)
            elif j==1:
                tmp["from"] = "human"
                tmp["value"] = data_llama3[i]["instruction"]
                tmp_list.append(tmp.copy())
                tmp["from"] = "gpt"
                tmp["value"] = data_gemma2[i]["predict"].split("The reasons are as follows:\n")[-1]
                tmp_list.append(tmp.copy())
                tmp["from"] = "human"
                tmp["value"] = "The error category code for the main error is:"
                tmp_list.append(tmp.copy())
                tmp["from"] = "gpt"
                tmp["value"] = data_gemma2[i]["predict"].split("The reasons are as follows:\n")[0][-3]
                pre_gemma = data_gemma2[i]["predict"].split("The reasons are as follows:\n")[0][-3]
                tmp_list.append(tmp.copy())
                tmp = {}
                tmp["conversations"] = tmp_list
                list_3.append(tmp)
            else:
                tmp["from"] = "human"
                tmp["value"] = data_llama3[i]["instruction"]
                tmp_list.append(tmp.copy())
                tmp["from"] = "gpt"
                tmp["value"] = data_mistral[i]["predict"].split("The reasons are as follows:\n")[-1]
                tmp_list.append(tmp.copy())
                tmp["from"] = "human"
                tmp["value"] = "The error category code for the main error is:"
                tmp_list.append(tmp.copy())
                tmp["from"] = "gpt"
                tmp["value"] = data_mistral[i]["predict"].split("The reasons are as follows:\n")[0][-3]
                pre_mistral = data_mistral[i]["predict"].split("The reasons are as follows:\n")[0][-3]
                tmp_list.append(tmp.copy())
                tmp = {}
                tmp["conversations"] = tmp_list
                list_3.append(tmp)
        tmp = {}



        if label == pre_llama3 and label==pre_mistral and label==pre_gemma:
            if all_corect >=2:
                continue
            tmp["label"] = label
            tmp["chosen"] = list_3[0]
            tmp["rejected"] = list_3[-1]
            all_corect += 1
            data_final.append(tmp)
        elif label == pre_llama3 and label==pre_mistral:
            if gemma_error > 1:
                continue
            tmp["label"] = label
            tmp["chosen"] = list_3[0]
            tmp["rejected"] = list_3[1]
            gemma_error += 1
            data_final.append(tmp)
        elif label == pre_mistral and label==pre_gemma:
            if llama3_error >=2:
                continue
            tmp["label"] = label
            tmp["chosen"] = list_3[1]
            tmp["rejected"] = list_3[0]
            llama3_error += 1
            data_final.append(tmp)
        elif label == pre_llama3 and label==pre_gemma:
            if mistral_error>=2:
                continue
            tmp["label"] = label
            tmp["chosen"] = list_3[0]
            tmp["rejected"] = list_3[-1]
            data_final.append(tmp)
        else:
            if all_error>=2:
                continue
            tmp["label"] = label
            tmp["chosen"] = list_3[0]
            tmp["rejected"] = list_3[-1]
            all_error += 1
            data_final.append(tmp)


    print(len(data_final))
    with open("case_predict.json", "w", encoding="utf-8") as f:
        json.dump(data_final, f, ensure_ascii=False, indent=4)
