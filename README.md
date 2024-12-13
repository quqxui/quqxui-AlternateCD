
# AlternateCD

## Requirements
    accelerate==0.18.0
    huggingface_hub==0.14.1
    jieba==0.42.1
    nltk==3.8.1
    numpy==1.24.3
    rouge_chinese==1.0.3
    sentencepiece==0.1.99
    torch==1.12.0+cu113
    tqdm==4.65.0

## Datasets
run

    unzip datasets.zip
    unzip transformers-4.27.1.zip

We put preprossed datasets in `datasets/`

The datasets for six tasks have been reorganized based on [PromptCBLUE](https://github.com/michael-wzhu/PromptCBLUE), where the original source data is available for download.



## Train and evaluate

### To generate structure answer from plain text of ground truth
```python
for task_dataset in  'CMedCausal'  'CMeIE-V2'  'CHIP-MDCFNPC' 'IMCS-V2-SR' 'CMeEE-V2' 'IMCS-V2-NER'
do
python src/evaluation/post_coupling.py datasets/$task_dataset/coupling_data/test.json datasets/$task_dataset/coupling_data/test_struc.json
done
```

### To decoupling abilities of LLM
```python
## train
LLM='chatglm' # qwen
for task_dataset in  'CMedCausal'  'CMeIE-V2'  'CHIP-MDCFNPC' 'IMCS-V2-SR' 'CMeEE-V2' 'IMCS-V2-NER'
do
    bash ./src/ft_${LLM}_lora/train.sh $task_dataset  coupling_data 0
    bash ./src/ft_${LLM}_lora/train.sh $task_dataset  identify_data 0
    bash ./src/ft_${LLM}_lora/train.sh $task_dataset  classify_data 0
done
```

### To generate with ALternate Contrastive Decoding
```python
bash ./src/ft_${LLM}_lora/evaluate.sh IMCS-V2-SR coupling_data 0 ACD 2 0.05 0.5 8 True
```

Reproduce the results by runing `bash run.sh`


This code refers to the code of [PromptCBLUE](https://github.com/michael-wzhu/PromptCBLUE) and [self-llm](https://github.com/datawhalechina/self-llm). Thanks their great projects.


## Citation
If you find this project useful in your research, please cite the following paper.

    @inproceedings{xu2024mitigating,
    title={Mitigating Hallucinations of Large Language Models in Medical Information Extraction via Contrastive Decoding},
    author={Xu, Derong and Zhang, Ziheng and Zhu, Zhihong and Lin, Zhenxi and Liu, Qidong and Wu, Xian and Xu, Tong and Zhao, Xiangyu and Zheng, Yefeng and Chen, Enhong},
    booktitle={Findings of the Association for Computational Linguistics: EMNLP 2024},
    pages={7744--7757},
    year={2024}
    }