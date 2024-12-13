LR=1e-4
model_name_or_path="Qwen/Qwen-7B-Chat" 

task_dataset=$1
your_exp_name=$2
device=$3
max_steps=3000
your_data_path="./datasets/$task_dataset/coupling_data" 

your_checkpopint_path="./experiments/$task_dataset/"  

CUDA_VISIBLE_DEVICES=$device python src/ft_qwen_lora/main.py \
    --do_train \
    --train_file $your_data_path/train.json \
    --model_name_or_path $model_name_or_path \
    --output_dir $your_checkpopint_path/$your_exp_name-qwen-7b-lora-$LR/checkpoint-$max_steps \
    --max_steps $max_steps \

