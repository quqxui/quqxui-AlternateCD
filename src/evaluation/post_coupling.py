# coding=utf-8
# Created by Michael Zhu
# ECNU, 2023

import json
import sys

# from tqdm import tqdm
import re

def process_generated_results_coupling(gt_file=None,test_predictions_list=None):
    structured_output = {
        "CMeEE-V2": [],
        "CMeIE-V2": [],
        "CHIP-MDCFNPC": [],
        "IMCS-V2-NER": [],
        "IMCS-V2-SR": [],
        "CMedCausal": [],
    }
    
    if gt_file is None:
        plain_text_list = test_predictions_list
    else:
        gt_list = []
        for line in open(gt_file, "r", encoding="utf-8"):
            gt_list.append(json.loads(line))
        plain_text_list = gt_list
        
    for line in plain_text_list:

        sample_id_ = line.get("sample_id", "xxxx")
        input = line["input"]
        gen_output = line["target"]
        gen_output = gen_output.replace(":", "：", 100).replace(",", "，", 100).replace(";", "；", 100).replace("(", "（", 100).replace(")", "）", 100)
        # gen_output = line["generated_output"]
        task_dataset = line["task_dataset"]
        task_type = line["task_type"]

        # 选项：
        answer_choices = line["answer_choices"]

        if task_dataset == "CMeIE-V2":
            list_spos = []
            assert isinstance(answer_choices, list)
            list_answer_strs = gen_output.split("\n")

            def wrong_pred():
                tri =   {
                                "predicate": "没有指定类型的三元组",
                                "subject": "没有指定类型的三元组",
                                "object": "没有指定类型的三元组",
                        }
                return tri
            for ans_str in list_answer_strs:
                if gen_output == "没有指定类型的三元组":
                    list_spos.append(wrong_pred())
                    continue
                if len(ans_str.split("：")) == 2:
                    ht, predicate = ans_str.split("：")
                    if len(ht.split("，")) == 2:
                        h,t = ht.split("，")
                        list_spos.append(
                            {
                                    "predicate": predicate,
                                    "subject": h,
                                    "object": t,
                            }
                        )
                    else:
                        list_spos.append(wrong_pred())
                else:
                    list_spos.append(wrong_pred())

            structured_output[f"{task_dataset}"].append(
                {
                    "sample_id": sample_id_,
                    "answer": list_spos,
                }
            )

        elif task_dataset == "CMedCausal":

            list_spos = []
            list_answer_strs = gen_output.split("\n")

            rel = None
            for ans_str in list_answer_strs:
                if gen_output == "给定的文本没有指定关系的三元组" or len(ans_str.split("："))!=2:
                    list_spos.append(
                        {
                                "predicate": "给定的文本没有指定关系的三元组",
                                "subject": "给定的文本没有指定关系的三元组",
                                "object": "给定的文本没有指定关系的三元组",
                        }
                    )
                    break
                # 停经时间的延长，(怀孕,偶尔小腹部隐隐坠痛,因果关系)：条件关系
                ht, rel = ans_str.split("：")
                if rel != "条件关系":
                    h = ht.split("，")[0]
                    t = "，".join(ht.split("，")[1:])
                    triple = {
                        "predicate": rel,
                        "subject": h,
                        "object": t,
                    }
                else:
                    ht = ht.replace('（','').replace('）','')
                    ht_split = ht.split("，")
                    if len(ht_split)!=4:
                        list_spos.append(
                            {
                                "predicate": "给定的文本没有指定关系的三元组",
                                "subject": "给定的文本没有指定关系的三元组",
                                "object": "给定的文本没有指定关系的三元组",
                            }
                        )
                        break
                    tail_triple = {
                                "predicate": ht_split[3],
                                "subject": ht_split[1],
                                "object": ht_split[2],
                        }

                    triple = {
                        "predicate": rel,
                        "subject": ht_split[0],
                        "object": tail_triple,
                    }

                list_spos.append(triple)

            structured_output[f"{task_dataset}"].append(
                {
                    "sample_id": sample_id_,
                    "answer": list_spos,
                }
            )

        elif task_dataset in ["CHIP-MDCFNPC","IMCS-V2-SR", "IMCS-V2-NER", "CMeEE-V2"]:
            # 答案格式：
            #   第一行：引导词
            #    每一行就是 "[症状词]：[阴阳性判断结果]"
            list_answer_strs = gen_output.split("\n")#[1:]
            list_finding_attrs = []
            for ans_str in list_answer_strs:
                if task_dataset == "CMeEE-V2":
                    if gen_output == "上述句子没有指定类型实体":
                        list_finding_attrs.append(
                            {
                                "entity": "上述句子没有指定类型实体",
                                "attr": "上述句子没有指定类型实体"
                            }
                        )
                        continue
                if not len(ans_str.split("：")) == 2:
                    continue

                finding, conclusion = ans_str.split("：")
                if conclusion not in answer_choices:
                    if task_dataset == "CHIP-MDCFNPC":
                        conclusion = "不标注"
                    elif task_dataset == "IMCS-V2-SR":
                        conclusion = "无法根据上下文确定病人是否患有该症状"

                list_finding_attrs.append(
                    {
                        "entity": finding.strip(),
                        "attr": conclusion
                    }
                )
            
            structured_output[f"{task_dataset}"].append(
                {
                    "sample_id": sample_id_,
                    "answer": list_finding_attrs,
                }
            )
        else:
            print("task_dataset: ", task_dataset)
            print("task_type: ", task_type)

            raise ValueError

    return structured_output


if __name__ == "__main__":
    from_dir = sys.argv[1]
    to_dir = sys.argv[2]
    
    structured_outputs = process_generated_results_coupling(
        from_dir, None
    )

    for key in structured_outputs.keys():
        print(key, len(structured_outputs[key]))

    json.dump(
        structured_outputs,
        open(to_dir, "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2
    )


