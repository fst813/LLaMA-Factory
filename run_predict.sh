MODEL_PATH="$1"
test_data_path="$2"
OUTPUT_DIR="$3"
CHECKPOINTS_DIR="$4"
template="$5"
echo $MODEL_PATH
echo $OUTPUT_DIR
echo $CHECKPOINTS_DIR
echo $template
cp $test_data_path data/tmp_eval.json
CUDA_VISIBLE_DEVICES="$6" llamafactory-cli train \
    --stage sft \
    --adapter_name_or_path $CHECKPOINTS_DIR \
    --model_name_or_path $MODEL_PATH \
    --do_predict \
    --fp16 \
    --dataset tmp_eval \
    --template $template \
    --cutoff_len 4096 \
    --use_fast_tokenizer False \
    --finetuning_type lora \
    --lora_target all \
    --output_dir $OUTPUT_DIR \
    --per_device_eval_batch_size 1 \
    --max_samples 100000 \
    --num_beams 1 \
    --top_k 1 \
    --top_p 0.75 \
    --temperature 0.1 \
    --max_new_tokens 512 \
    --predict_with_generate
