MODEL_PATH="Meta-Llama-3-8B-Instruct/"
LORA_PATH="checkpoint-70"
OUTPUT_DIR="llama3_export"
CUDA_VISIBLE_DEVICES=0 llamafactory-cli export \
    --model_name_or_path $MODEL_PATH \
    --template llama3 \
    --finetuning_type lora \
    --export_device auto \
    --lora_target all \
    --adapter_name_or_path $LORA_PATH \
    --export_dir $OUTPUT_DIR \
    --export_legacy_format False\
    --export_size 10 \
