#!/bin/bash

###  generate structure answer from plain text of gt
for task_dataset in  'CMedCausal'  'CMeIE-V2'  'CHIP-MDCFNPC' 'IMCS-V2-SR' 'CMeEE-V2' 'IMCS-V2-NER'
do
python src/evaluation/post_coupling.py datasets/$task_dataset/coupling_data/test.json datasets/$task_dataset/coupling_data/test_struc.json
done

## train
LLM='chatglm' # qwen
for task_dataset in  'CMedCausal'  'CMeIE-V2'  'CHIP-MDCFNPC' 'IMCS-V2-SR' 'CMeEE-V2' 'IMCS-V2-NER'
do
    bash ./src/ft_${LLM}_lora/train.sh $task_dataset  coupling_data 0
    bash ./src/ft_${LLM}_lora/train.sh $task_dataset  identify_data 0
    bash ./src/ft_${LLM}_lora/train.sh $task_dataset  classify_data 0
done

## evaluate
bash ./src/ft_${LLM}_lora/evaluate.sh IMCS-V2-SR coupling_data 0 ACD 2 0.05 0.5 8 True
bash ./src/ft_${LLM}_lora/evaluate.sh IMCS-V2-NER coupling_data 0 ACD 2 0.25 0.5 8 True
bash ./src/ft_${LLM}_lora/evaluate.sh CHIP-MDCFNPC coupling_data 0 ACD 2 0.01 0.4 8 True
bash ./src/ft_${LLM}_lora/evaluate.sh CMedCausal coupling_data 0 ACD 2 0.05 0.4 8 False
bash ./src/ft_${LLM}_lora/evaluate.sh CMeIE-V2 coupling_data 0 ACD 2 0.01 0.5 8 True
bash ./src/ft_${LLM}_lora/evaluate.sh CMeEE-V2 coupling_data 0 ACD 2 0.01 0.4 8 True

