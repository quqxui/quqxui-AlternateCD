lora_rank=8
lora_trainable="query_key_value,dense,dense_h_to_4h,dense_4h_to_h"
modules_to_save="null"
lora_dropout=0.1
LR=2e-4
model_name_or_path="THUDM/chatglm-6b" 
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
CHECKPOINT="./experiments/$task_dataset/$your_exp_name-chatglm-6b-lora-2e-4" 

STEP=1000  

# output_dir="$CHECKPOINT/checkpoint-$STEP"

output_dir="./experiments_cd/chatglm_results/$task_dataset/"

if [ $contrastive_decoding = 'ACD' ];  
then  
echo "contrastive decoding"
fi


CUDA_VISIBLE_DEVICES=$device python src/ft_chatglm_lora/main.py \
    --do_predict \
    --validation_file $your_data_path/dev.json \
    --test_file $your_data_path/test.json \
    --cache_dir $your_data_path \
    --overwrite_cache \
    --prompt_column input \
    --response_column target \
    --model_name_or_path $model_name_or_path \
    --peft_path $CHECKPOINT/checkpoint-$STEP \
    --output_dir $output_dir \
    --overwrite_output_dir \
    --max_source_length 828 \
    --max_target_length 196 \
    --per_device_eval_batch_size $batch_size \
    --predict_with_generate \
    --contrastive_decoding $contrastive_decoding \
    --alpha $alpha \
    --max_rate $max_rate \
    --num_beams $num_beams \
    --AAC $AAC \
