#model_path="/data2/fst/CodeLlama-13b-Instruct-hf/"
model_path="../../download_models/LLM-Research/Meta-Llama-3-8B-Instruct/"
train_data="data/v7.4.8_test.json"
#train_data="data/train_all_rewrite_random.json"
#dev_data="data/v7.4.2.1_1212_test.json"
dev_data="data/v7.4.8_test.json"
lora_output_dir="llama3_8b_v7_4_9_int8"
template="llama3"
batch_size=1
save_steps=100
train_CUDA_VISIBLE_DEVICES="2"
eval_CUDA_VISIBLE_DEVICES="2"

#以下内容无需修改
predict_result_dir="${lora_output_dir}_predict"
log_train="logs/${lora_output_dir}_$(date +'%Y-%m-%d_%H-%M-%S').log"
log_predict="logs/predict_${lora_output_dir}.log"

if [ ! -d "$lora_output_dir" ]; then
    echo "lora文件夹不存在, 创建文件夹"
    mkdir -p $lora_output_dir
fi

if [ ! -d "logs" ]; then
    echo "logs文件夹不存在, 创建文件夹"
    mkdir -p "logs"
fi

echo "开始训练"
nohup bash run_train.sh $model_path $train_data $lora_output_dir $template $train_CUDA_VISIBLE_DEVICES $batch_size $save_steps > $log_train 2>&1 &


echo "开始验证集推理"
nohup bash run_predict.sh $model_path $dev_data $predict_result_dir $lora_output_dir $template $eval_CUDA_VISIBLE_DEVICES > $log_predict 2>&1 &







