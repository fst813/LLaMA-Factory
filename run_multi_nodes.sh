#!/bin/bash
export train_CUDA_VISIBLE_DEVICES="host0:1,3@host1:0,1,2,3"
#export MODEL_PATH="/data/fst/download_models/ZhipuAI/glm-4-9b-chat/"
export MODEL_PATH="../../download_models/LLM-Research/Meta-Llama-3-8B-Instruct/"
export template="llama3_our"
export lora_output_dir="llama3_pissa_v7_8_0"
export batch_size=3
export gradient_acc_steps=3
export save_steps=250
export eval_CUDA_VISIBLE_DEVICES="0"
export predict_result_dir="${lora_output_dir}_predict"
#train_data_path="data/v7.4.8_test.json"
train_data_path="data/base_v_7_8_0_3500_addtable_prompt.json"
dev_data_path="data/v7.4.8_test.json"
cp $train_data_path data/tmp_train.json
cp $dev_data_path data/tmp_eval.json

envsubst < run_train_template.sh > run_train_multi_nodes.sh
envsubst < run_predict_template.sh > run_predict_single.sh

#chmod +x run_train_multi_nodes.sh
if [ ! -d "$lora_output_dir" ]; then
    mkdir $lora_output_dir
fi
if [ ! -d "logs" ]; then
    mkdir logs
    scp -q -r logs host1:/home/fangsongtao/train_frame/LLaMA-Factory/
fi
# 同步
scp -q run_train_multi_nodes.sh host1:/home/fangsongtao/train_frame/LLaMA-Factory
scp -q data/tmp_train.json host1:/home/fangsongtao/train_frame/LLaMA-Factory/data
scp -q data/dataset_info.json host1:/home/fangsongtao/train_frame/LLaMA-Factory/data


log_train="logs/${lora_output_dir}_$(date +'%Y-%m-%d_%H-%M-%S').log"
log_predict="logs/predict_${lora_output_dir}.log"0603
echo "开始训练"
nohup bash run_train_multi_nodes.sh > $log_train 2>&1 &
echo "开始验证集推理"
nohup bash run_predict_single.sh > $log_predict 2>&1 &
