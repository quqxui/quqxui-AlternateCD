lora_rank=8
lora_trainable="query_key_value,dense,dense_h_to_4h,dense_4h_to_h"
modules_to_save="null"
lora_dropout=0.1
LR=2e-4
model_name_or_path="THUDM/chatglm-6b"   # LLM底座模型路径，或者是huggingface hub上的模型名称

task_dataset=$1
your_exp_name=$2
device=$3

your_data_path="./datasets/$task_dataset/coupling_data"  # 填入数据集所在的文件夹路径

your_checkpopint_path="./experiments/$task_dataset/"  # 填入用来存储模型的路径

peft_path=""  # 如果之前训练过，且存储了peft权重，则设置为peft权重的文件夹路径

CUDA_VISIBLE_DEVICES=$device python src/ft_chatglm_lora/main.py \
    --do_train \
    --train_file $your_data_path/train.json \
    --validation_file $your_data_path/dev.json \
    --cache_dir $your_data_path \
    --prompt_column input \
    --response_column target \
    --overwrite_cache \
    --model_name_or_path $model_name_or_path \
    --output_dir $your_checkpopint_path/$your_exp_name-chatglm-6b-lora-$LR \
    --overwrite_output_dir \
    --max_source_length 828 \
    --max_target_length 196 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --max_steps 1000 \
    --logging_steps 10 \
    --save_steps 100 \
    --learning_rate $LR \
    --lora_rank ${lora_rank} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} 



