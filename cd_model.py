#!/usr/bin/env python
# coding=utf-8
from peft import PeftModel
## 
import torch
import copy
class CDModel:
    def __init__(self, pretrain_model, LLM,input_ids):
        ### for CD
        self.pretrain_model = pretrain_model
        import gol
        model_args = gol.get_value('gol_model_args')
        tokenizer = gol.get_value('gol_tokenizer')
        test_file = gol.get_value('gol_test_file')
        self.model_args = model_args
        self.tokenizer = tokenizer

        if model_args.contrastive_decoding in ['ACD','ACD_CD']:
            self.model_cd = PeftModel.from_pretrained(pretrain_model, model_args.peft_path,adapter_name='coupling')
            self.model_cd.load_adapter(model_args.peft_path.replace('coupling','identify').replace('1000',str(model_args.sub_step)), adapter_name='identify')
            self.model_cd.load_adapter(model_args.peft_path.replace('coupling','classify').replace('1000',str(model_args.sub_step)), adapter_name='classify')
            self.alter_pattern = ['ide' for i in range(input_ids.shape[0])]
            ## for plot
            # data_plot = {'cou':[],'ide':[],'cla':[],'next_tokens':[]}
            import json
            with open(test_file, 'r', encoding="utf-8") as f:
                for line in f.readlines():
                    a = json.loads(line)
                    self.answer_choices = a['answer_choices']
                    break
            self.cla_tokens = tokenizer.encode("".join(self.answer_choices))[1:-2] # qwen 不一样 TODO

        elif model_args.contrastive_decoding=='CD':
            self.model_cd = PeftModel.from_pretrained(pretrain_model, model_args.peft_path,adapter_name='coupling')
            if '1000' in model_args.peft_path:
                self.model_cd.load_adapter(model_args.peft_path.replace('1000','500'), adapter_name='coupling_small')
            elif '3000' in model_args.peft_path:
                self.model_cd.load_adapter(model_args.peft_path.replace('3000','2000'), adapter_name='coupling_small')
        elif model_args.contrastive_decoding in ['CAD']:
            import json
            with open(test_file, 'r',encoding='utf-8') as f:
                for line in f.readlines():
                    a = json.loads(line)
                    self.answer_choices = a['answer_choices']
                    break

    def jensen_shannon_divergence(self,p, q):
        import torch.nn.functional as F
        p = p.float()
        q = q.float()
        m = 0.5 * (F.softmax(p,dim=-1) + F.softmax(q,dim=-1))
        kl_p = F.kl_div(F.log_softmax(p,dim=-1), m, reduction='none').sum(-1)
        kl_q = F.kl_div(F.log_softmax(q,dim=-1), m, reduction='none').sum(-1)
        jsd = 0.5 * (kl_p + kl_q)
        return jsd

    def decoding(self, input_ids, model_kwargs, output_attentions, output_hidden_states):

        if self.model_args.contrastive_decoding=='normal':
            # prepare model inputs  
            model_inputs = self.pretrain_model.prepare_inputs_for_generation(input_ids, **model_kwargs)
            # forward pass to get next token
            outputs = self.pretrain_model(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            next_token_logits = outputs.logits[:, -1, :]

        elif self.model_args.contrastive_decoding=='DoLa':
            # prepare model inputs  
            model_inputs = self.pretrain_model.prepare_inputs_for_generation(input_ids, **model_kwargs)
            # forward pass to get next token
            outputs = self.pretrain_model(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=True, 
            )
            dict_outputs = {}
            for i, early_exit_layer in enumerate([0,2,4,6,8,10,12,14,28]):
                lm_logits = self.pretrain_model.lm_head(outputs.hidden_states[early_exit_layer]).permute(1, 0, 2).contiguous()
                dict_outputs[early_exit_layer] = lm_logits

            jsd_max = [0] * input_ids.shape[0]
            layer_max = [0] * input_ids.shape[0]
            logits_mature = outputs.logits[:, -1, :]
            # print(dict_outputs[0].size()) torch.Size([8, 93, 130528])
            for layer, outputs_early_exit in dict_outputs.items():
                jsd_now = self.jensen_shannon_divergence(outputs_early_exit[:, -1, :], logits_mature)
                for i in range(len(jsd_max)):
                    if jsd_max[i] < jsd_now[i]:
                        jsd_max[i] = jsd_now[i]
                        layer_max[i] = layer

            # print(layer_max) #
            logits_early_exit = []
            for i in range(len(layer_max)): #bach size
                logits_early_exit.append(dict_outputs[layer_max[i]][i, -1, :])
            logits_early_exit = torch.stack(logits_early_exit,dim=0)                
            next_token_logits = (1 + self.model_args.alpha) * logits_mature - self.model_args.alpha * logits_early_exit
            next_token_logits[logits_mature < self.model_args.max_rate * torch.max(logits_mature)] = -1000

        elif self.model_args.contrastive_decoding=='CD':
            # prepare model inputs
            model_inputs = self.pretrain_model.prepare_inputs_for_generation(input_ids, **model_kwargs)
            # forward pass to get next token
            self.model_cd.set_adapter('coupling')
            outputs_cou = self.model_cd.base_model(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            self.model_cd.set_adapter('coupling_small')
            outputs_cou_500 = self.model_cd.base_model(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            logits_cou = outputs_cou.logits[:, -1, :] # torch.Size([8, 130528])
            logits_cou_500 = outputs_cou_500.logits[:, -1, :] # torch.Size([8, 130528])

            next_token_logits = (1 + self.model_args.alpha) * logits_cou - self.model_args.alpha * logits_cou_500
            next_token_logits[logits_cou < self.model_args.max_rate * torch.max(logits_cou)] = -1000

            outputs = outputs_cou


        elif self.model_args.contrastive_decoding in ['CAD']:
            # prepare model inputs
            model_inputs = self.pretrain_model.prepare_inputs_for_generation(input_ids, **model_kwargs)

            input_ids_cd = copy.deepcopy(input_ids)
            input_prompts = self.tokenizer.batch_decode(input_ids_cd)
            select_input_prompts = []
            for pr in input_prompts:
                #### CAD
                if self.model_args.contrastive_decoding=='CAD':
                    for ans in self.answer_choices:
                        pr = pr.replace(ans,"")
                    select_input_prompts.append(pr)
                ########
            select_input_ids = self.tokenizer(select_input_prompts)

            for i in range(input_ids_cd.shape[0]):
                sii = select_input_ids['input_ids'][i]
                for j in range(len(input_ids_cd[i]) - len(sii)):
                        input_ids_cd[i][j] = 3
                input_ids_cd[i][-len(sii):] = torch.LongTensor(sii)

            model_inputs_cd = self.pretrain_model.prepare_inputs_for_generation(input_ids_cd, **model_kwargs)
            
            # forward pass to get next token
            outputs = self.pretrain_model(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            outputs_cd = self.pretrain_model(
                **model_inputs_cd,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            logits_cou = outputs.logits[:, -1, :] # torch.Size([8, 130528])
            logits_cd = outputs_cd.logits[:, -1, :] # torch.Size([8, 130528])

            next_token_logits = (1 + self.model_args.alpha) * logits_cou - self.model_args.alpha * logits_cd
            next_token_logits[logits_cou < self.model_args.max_rate * torch.max(logits_cou)] = -1000

        elif self.model_args.contrastive_decoding=='ACD':

            input_ids_cd = copy.deepcopy(input_ids)
            input_prompts = self.tokenizer.batch_decode(input_ids_cd)
            select_input_prompts = []
            for pr in input_prompts:
                select_input_prompts.append(pr[-1]) # CFG 
            select_input_ids = self.tokenizer(select_input_prompts)

            for i in range(input_ids_cd.shape[0]):
                sii = select_input_ids['input_ids'][i]
                for j in range(len(input_ids_cd[i]) - len(sii)):
                        input_ids_cd[i][j] = 3
                input_ids_cd[i][-len(sii):] = torch.LongTensor(sii)

            model_inputs_cd = self.pretrain_model.prepare_inputs_for_generation(input_ids_cd, **model_kwargs)
            self.model_cd.set_adapter('coupling')
            outputs_cd = self.pretrain_model(
                **model_inputs_cd,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            logits_cd = outputs_cd.logits[:, -1, :]

            # prepare model inputs
            model_inputs = self.pretrain_model.prepare_inputs_for_generation(input_ids, **model_kwargs)
            # forward pass to get next token
            if 'coupling' not in self.model_args.peft_path:
                raise
            self.model_cd.set_adapter('coupling')
            outputs_cou = self.model_cd.base_model(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            self.model_cd.set_adapter('classify')
            outputs_cla = self.model_cd.base_model(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            self.model_cd.set_adapter('identify')
            outputs_ide = self.model_cd.base_model(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            logits_cou = outputs_cou.logits[:, -1, :] # torch.Size([8, 130528])
            logits_ide = outputs_ide.logits[:, -1, :]
            logits_cla = outputs_cla.logits[:, -1, :]
            next_token_logits = logits_cla
            outputs = outputs_cla

        elif self.model_args.contrastive_decoding=='ACD_CD':

            input_ids_cd = copy.deepcopy(input_ids)
            input_prompts = self.tokenizer.batch_decode(input_ids_cd)
            select_input_prompts = []
            for pr in input_prompts:
                select_input_prompts.append(pr[-1]) # CFG 
            select_input_ids = self.tokenizer(select_input_prompts)

            for i in range(input_ids_cd.shape[0]):
                sii = select_input_ids['input_ids'][i]
                for j in range(len(input_ids_cd[i]) - len(sii)):
                        input_ids_cd[i][j] = 3
                input_ids_cd[i][-len(sii):] = torch.LongTensor(sii)

            model_inputs_cd = self.pretrain_model.prepare_inputs_for_generation(input_ids_cd, **model_kwargs)
            self.model_cd.set_adapter('coupling')
            outputs_cd = self.pretrain_model(
                **model_inputs_cd,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            logits_cd = outputs_cd.logits[:, -1, :]

            # prepare model inputs
            model_inputs = self.pretrain_model.prepare_inputs_for_generation(input_ids, **model_kwargs)
            # forward pass to get next token
            if 'coupling' not in self.model_args.peft_path:
                raise
            self.model_cd.set_adapter('coupling')
            outputs_cou = self.model_cd.base_model(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            self.model_cd.set_adapter('classify')
            outputs_cla = self.model_cd.base_model(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            self.model_cd.set_adapter('identify')
            outputs_ide = self.model_cd.base_model(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            logits_cou = outputs_cou.logits[:, -1, :] # torch.Size([8, 130528])
            logits_ide = outputs_ide.logits[:, -1, :]
            logits_cla = outputs_cla.logits[:, -1, :]
            jsd_ide = self.jensen_shannon_divergence(logits_cou, logits_ide)
            jsd_cla = self.jensen_shannon_divergence(logits_cou, logits_cla)
            jsd_ide = torch.clamp(jsd_ide, min=0, max=1)
            jsd_cla = torch.clamp(jsd_cla, min=0, max=1)
            next_token_logits = []
            for i in range(len(self.alter_pattern)):
                cou,ide,cla,cd = logits_cou[i],logits_ide[i],logits_cla[i], logits_cd[i]

                max_token = torch.argmax(cou)
                if max_token == 4 or max_token == 12:
                    next_logit = cou
                elif self.alter_pattern[i] == 'cla':
                    if self.model_args.AAC:
                        next_logit = cou + self.model_args.alpha * (cou + (1 - jsd_cla[i]) * cla - jsd_ide[i] * ide - cd)
                    else:
                        next_logit = cou + self.model_args.alpha * (cou + cla - ide - cd)
                elif self.alter_pattern[i] == 'ide':
                    if self.model_args.AAC:
                        next_logit = cou + self.model_args.alpha * (cou + (1 - jsd_ide[i]) * ide - jsd_cla[i] * cla - cd)
                    else:
                        next_logit = cou + self.model_args.alpha * (cou + ide - cla - cd)
                next_logit[cou < self.model_args.max_rate * torch.max(cou)] = -1000
                next_logit[ide < self.model_args.max_rate * torch.max(ide)] = -1000
                next_logit[cla < self.model_args.max_rate * torch.max(cla)] = -1000

                next_token_logits.append(next_logit)
            next_token_logits = torch.stack(next_token_logits,dim=0)
            outputs = outputs_cou
        else:
            raise
        
        return next_token_logits, outputs