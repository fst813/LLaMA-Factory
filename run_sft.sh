MODEL_PATH="Meta-Llama-3-8B-Instruct/"
train_data_path="train_daxuanjie_62.json"
lora_output_dir="sft_output"
template="llama3"
train_CUDA_VISIBLE_DEVICES="0,1"
batch_size="1"
save_steps="10"
cp $train_data_path data/tmp_train.json
deepspeed --include localhost:$train_CUDA_VISIBLE_DEVICES --master_port 29501 src/train.py \
	--stage sft \
	--preprocessing_num_workers 16 \
	--model_name_or_path $MODEL_PATH \
	--do_train \
	--dataset tmp_train \
	--template $template \
	--finetuning_type lora \
	--lora_target all \
	--output_dir $lora_output_dir \
	--overwrite_cache \
	--overwrite_output_dir \
	--per_device_train_batch_size $batch_size \
	--gradient_accumulation_steps 1 \
	--flash_attn fa2 \
	--quantization_bit 8 \
	--lr_scheduler_type cosine \
	--logging_steps 10 \
	--save_steps $save_steps \
	--cutoff_len 4096 \
	--learning_rate 1e-4 \
	--num_train_epochs 1.0 \
	--plot_loss \
	--fp16 \
	--group_by_length \
	--use_fast_tokenizer False \
	--lora_alpha 16 \
	--lora_rank 8 \
	--lora_dropout 0.1


