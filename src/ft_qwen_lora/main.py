from datasets import Dataset
import pandas as pd
import os
os.environ["WANDB_DISABLED"] = "true"
import json
import sys
sys.path.append("./")
sys.path.append("./transformers-4.27.1/src/")
import random
random.seed(4)
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
import copy
import torch
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
import tqdm
import gol
gol._init()

def dataload(file):
    data = []
    with open(file, 'r', encoding='utf-8' ) as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data)
    ds = Dataset.from_pandas(df)
    return ds

def train_lora(model_args):
    def process_func(example):
        MAX_LENGTH = 1024    
        input_ids, attention_mask, labels = [], [], []
        instruction = tokenizer(f"<|im_start|>system\n现在你要扮演一名医疗专家.<|im_end|>\n<|im_start|>user\n" + example["input"] + "<|im_end|>\n", add_special_tokens=False)  
        # instruction = tokenizer(example["input"] , add_special_tokens=False)  
        response = tokenizer("<|im_start|>" + example["target"] + "<|im_end|>", add_special_tokens=False) # <|im_start|> 151644  <|im_end|> 151645
        # response = tokenizer(example["target"], add_special_tokens=False)
        input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  

        if 'classify' in model_args.output_dir or 'identify' in model_args.output_dir:
        # maohao = tokenizer.encode(text='：', add_special_tokens=False) # 5122
        # huanhai = tokenizer.encode(text='\n', add_special_tokens=False) # 198
            indices_ide, indices_cla = [],[]
            flag = 'ide'
            for index, value in enumerate(response["input_ids"]):
                if value == 151644 or value == 151645:
                    continue
                if value == 5122:
                    flag = 'cla'
                    continue
                elif value == 198:
                    flag = 'ide'
                    continue
                if flag == 'ide':
                    indices_ide.append(index)
                elif flag == 'cla':
                    indices_cla.append(index)
            label_mask = []
            if 'classify' in model_args.output_dir:
                indices_mask = indices_ide
            elif 'identify' in model_args.output_dir:
                indices_mask = indices_cla
            for index, value in enumerate(response["input_ids"]):
                if index in indices_mask:
                    label_mask.append(-100)
                else:
                    label_mask.append(value)
            labels = [-100] * len(instruction["input_ids"]) + label_mask + [tokenizer.pad_token_id]
        else:
            labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
        
        if len(input_ids) > MAX_LENGTH: 
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    #  loraConfig
    if model_args.do_train:
        inference_mode=False
    else:
        inference_mode=True
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        target_modules=["c_attn", "c_proj", "w1", "w2"], 
        inference_mode=inference_mode,
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.1
    )

    args = TrainingArguments(
        output_dir=model_args.output_dir,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        logging_steps=10,
        # num_train_epochs=2,
        max_steps=model_args.max_steps,
        gradient_checkpointing=True,
        # save_steps=10,
        save_strategy='no',
        learning_rate=1e-4,
        save_on_each_node=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=False, trust_remote_code=True,padding_side='left') #  padding_side='left' 十分影响性能！！！！！
    tokenizer.pad_token_id = tokenizer.eod_id
    gol.set_value('gol_tokenizer',tokenizer)
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, torch_dtype=torch.half, device_map='auto' )
    model.enable_input_require_grads()

    if model_args.do_train:
        ds = dataload(model_args.train_file)
        tokenized_id = ds.map(process_func, remove_columns=ds.column_names)
        model = get_peft_model(model, config)
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized_id,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
            )
        trainer.train()
        trainer.model.save_pretrained(model_args.output_dir)

    if model_args.do_predict:
        test_data = []
        with open(model_args.test_file, 'r', encoding='utf-8' ) as f:
            for line in f:
                test_data.append(json.loads(line))
        # model = PeftModel.from_pretrained(model, model_args.output_dir)
        model.eval()


        batch_inputs = []
        for i in range(0, len(test_data), model_args.per_device_eval_batch_size):
            batch = test_data[i:i+model_args.per_device_eval_batch_size]
            batch_input = [f"<|im_start|>system\n现在你要扮演一名医疗专家.<|im_end|>\n<|im_start|>user\n" + one_data["input"] + "<|im_end|>\n" for one_data in batch]
            # batch_input = [f"<|im_start|>system\n现在你要扮演一名医疗专家.<|im_end|>\n<|im_start|>user\n" + one_data["input"] + "<|im_end|>\n 请按照指定格式输出，不要输出额外的文本，比如按照如下格式\n  发烧：患有该症状\n咳嗽：患有该症状\n湿疹：没有患有该症状" for one_data in batch]
            # fewshot = random.choices(train_data_str, k=1)
            # fewshot = "我需要你进行信息抽取，获得文本中的实体及其症状判断。这里给出了一些数据样例：" + "\n".join(fewshot) + "\n  下面是待抽取的文本内容："
            # batch_input = [f"<|im_start|>system\n现在你要扮演一名医疗专家.<|im_end|>\n<|im_start|>user\n" + fewshot +  one_data["input"] + "<|im_end|>\n 请按照指定格式输出结构化信息，不要输出额外的文本，比如按照如下格式\n  发烧：患有该症状\n咳嗽：患有该症状\n湿疹：没有患有该症状" for one_data in batch]

            batch_inputs.append(batch_input)
        batch_tokens = [tokenizer(batch_input,padding=True,max_length=828,truncation=True,return_tensors="pt").to(model.device) for batch_input in batch_inputs]
        responses = []
        for ipt in tqdm.tqdm(batch_tokens):
            model_generate = model.generate(**ipt, 
                            # do_sample=True,
                            eos_token_id=tokenizer.eos_token_id, 
                            num_beams=model_args.num_beams,
                            use_cache=True,
                            max_new_tokens=196,
                            # top_k=50,
                            # top_p=0.9,
                            # penalty_alpha=0.6, top_k=4,
                            # temperature=5.95,
                            # repetition_penalty=1.1
                            )
            response = tokenizer.batch_decode(model_generate,skip_special_tokens=False)
            im_start = response[0].rfind('<|im_start|>')
            im_end = response[0].rfind('<|im_end|>')
            # response = tokenizer.batch_decode(model.generate(**ipt, do_sample=False, eos_token_id=tokenizer.eos_token_id, temperature=0.1)[0], skip_special_tokens=True)
            responses += response
        predict_dataset = []
        for data, response in zip(test_data,responses):
            predict_data = copy.deepcopy(data)
            im_start = response.rfind('<|im_start|>')
            im_end = response.rfind('<|im_end|>')
            predict_data['target'] = response[im_start+len('<|im_start|>'):im_end]
            predict_dataset.append(predict_data)
        test_predictions_list = []
        for line in predict_dataset:
            test_predictions_list.append(line)
        with open(model_args.results_dir + 'test_predictions.json', 'w',encoding="utf-8") as f:
            for line in predict_dataset:
                f.write(json.dumps(line,ensure_ascii=False))
                f.write('\n')

        post_coupling_and_evaluation(model_args,test_predictions_list)

def post_coupling_and_evaluation(model_args,test_predictions_list):

    ############################ post_coupling
    from src.evaluation import post_coupling
    from_dir = model_args.results_dir + "test_predictions.json"
    to_dir = model_args.results_dir + "test_cd_struc.json"

    structured_outputs = post_coupling.process_generated_results_coupling(
        from_dir,test_predictions_list
    )
    for key in structured_outputs.keys():
        print(key, len(structured_outputs[key]))

    json.dump(
        structured_outputs,
        open(to_dir, "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2
    )

    ############################ evaluation
    debug_mode = 0
    success_flag = 1
    standard_path = model_args.test_file.replace("test.json","test_struc.json")
    out_path = model_args.results_dir + "test_cd_result.json"
    # 加载金标准文件和预测文件
    dict_pred = structured_outputs
    dict_gt = json.load(
        open(standard_path, "r", encoding="utf-8")
    )
    from src.evaluation.evaluate import report_score,calc_scores
    score_map, success_flag = calc_scores(
        dict_gt, dict_pred, out_path
    )
    score_map = {key: round(value * 100, 2 ) for key, value in score_map.items()}
    print(score_map)
    # report_score(score_map, out_path)



def Args():
    parser = argparse.ArgumentParser(description='Example script with argparse')
    # Add arguments
    parser.add_argument('--train_file', type=str, default=None,)
    parser.add_argument('--output_dir', type=str, default=None,)
    parser.add_argument('--model_name_or_path', type=str, default=None,)
    parser.add_argument('--do_train',  action='store_true')
    parser.add_argument('--max_steps', type=int,default=1000,)

    parser.add_argument('--do_predict', action='store_true')
    parser.add_argument('--test_file', type=str, default=None,)
    parser.add_argument('--results_dir', type=str, default=None,)
    parser.add_argument('--per_device_eval_batch_size', type=int,default=8,)
    parser.add_argument('--contrastive_decoding', type=str, default='normal',)
    parser.add_argument('--alpha', type=float,default=0.1,)
    parser.add_argument('--max_rate', type=float,default=0.1,)
    parser.add_argument('--num_beams', type=int,default=None,)
    parser.add_argument('--AAC', type=bool,default=False,)
    parser.add_argument('--sub_step', type=int,default=1000,)


    model_args = parser.parse_args()
    print(model_args)
    return model_args

if "__main__" == __name__:
    model_args = Args()
    gol.set_value('gol_model_args',model_args)
    gol.set_value('gol_test_file',model_args.test_file)
    train_lora(model_args)
