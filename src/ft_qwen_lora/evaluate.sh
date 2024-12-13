#!/bin/bash

LR=1e-4
model_name_or_path="Qwen/Qwen-7B-Chat"  

task_dataset=$1
your_exp_name=$2
device=$3
contrastive_decoding=$4
batch_size=$5
alpha=$6
max_rate=$7
num_beams=$8
AAC=$9

your_data_path="./datasets/$task_dataset/coupling_data"  
CHECKPOINT="./experiments/$task_dataset/$your_exp_name-qwen-7b-lora-1e-4"  


if [[ $task_dataset == "CHIP-MDCFNPC" || $task_dataset == "CMeIE-V2" || $task_dataset == "CMedCausal" ]]; then
    STEP=3000
else
    STEP=1000
fi
echo "STEP:$STEP"


results_dir="./experiments_cd/qwen_results/$task_dataset/"


CUDA_VISIBLE_DEVICES=$device python src/ft_qwen_lora/main.py \
    --do_predict \
    --test_file $your_data_path/test.json \
    --model_name_or_path $model_name_or_path \
    --output_dir $CHECKPOINT/checkpoint-$STEP \
    --results_dir $results_dir \
    --per_device_eval_batch_size $batch_size \
    --contrastive_decoding $contrastive_decoding \
    --alpha $alpha \
    --max_rate $max_rate \
    --num_beams $num_beams \
    --AAC $AAC \
    