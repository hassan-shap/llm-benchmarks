from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_metric, Dataset
import torch
import timeit
import sys
from torch.utils.data import Subset
import random
import fire
import os 
import torch.nn as nn
from datetime import date
import json 
import shutil
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from transformers import BitsAndBytesConfig
from peft import (
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict
)
from peft import prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from peft import PeftModel


def evaluate_model(model_name : str = 'None',
                   base : bool = False,
                   tuned_model_dir : str = '',
                   first_layer_dropped : str = '',
                   layer_drop_pattern : str = 'end',
                   BATCH_SIZE : int = 2,
                   instruction : str = '',
                   dataset_name : str = '',
                   split : str = 'test', 
                   num_shots : int = 0, 
                   top_k : int = 10,
                   save : bool = False, 
                   out_path : str = '', 
                   details : bool = True):


    device = "cuda:0"
    device_map = "cuda:0"

    #Loading tokenizer
    if "Qwen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, pad_token='<|extra_0|>', trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side = 'right', truncation_side='right')

    # Sometime pad_token is not deifned
    if tokenizer.pad_token_id == None:
        tokenizer.pad_token_id = 1
    
    print(f" HERE ARE THE SPECIAL TOKENS {tokenizer.pad_token_id}, {tokenizer.bos_token_id}, {tokenizer.eos_token_id}")
    #### Load the pretrained model and tokenizer ####
    # 16 bit
    # base_model = AutoModelForCausalLM.from_pretrained(model_name)
    # base_model = base_model.to(device = device)

    # 4bit
    bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
    base_model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                quantization_config=bnb_config,
                device_map=device_map,
                pad_token_id=1,
                trust_remote_code=True
           )

    # PEFT LORA LOAD
    if first_layer_dropped > -1:
        # Add pattern logic here

        if "Qwen" in model_name:
            depth = len(base_model.transformer.h)

            if layer_drop_pattern == 'end':
                layers_dropped = list(range(first_layer_dropped, depth-1))
                base_model.transformer.h = nn.ModuleList([block for idx, block in enumerate(base_model.transformer.h) if idx not in layers_dropped])

            if layer_drop_pattern == 'front':
                layers_dropped = list(range(0, first_layer_dropped+1))
                base_model.transformer.h = nn.ModuleList([block for idx, block in enumerate(base_model.transformer.h) if idx not in layers_dropped])
            
            #if layer_drop_pattern == 'equi':

        else:
            depth = len(base_model.base_model.layers)

            if layer_drop_pattern == 'end': 
                layers_dropped = list(range(first_layer_dropped, depth-1))
                base_model.base_model.layers = nn.ModuleList([block for idx, block in enumerate(base_model.base_model.layers) if idx not in layers_dropped])
            
            if layer_drop_pattern == 'front':
                layers_dropped = list(range(0, first_layer_dropped+1))
                base_model.base_model.layers = nn.ModuleList([block for idx, block in enumerate(base_model.base_model.layers) if idx not in layers_dropped])
            
            #if layer_drop_pattern == 'equi':

        print(f"We are removing: {layers_dropped} layers")
        
        if base: 
            model = base_model
        else:
            model = PeftModel.from_pretrained(
                    base_model,
                    tuned_model_dir,
                    torch_dtype=torch.bfloat16,
                )
            del base_model

    if first_layer_dropped == -1:
        model = base_model
        print(f"We are removing: 0 layers and evaluating the base model!")

    model.eval()
    


    # Prepare variables to collect MMLU data
    answers = []
    top_tokens = []
    top_tokens_decoded = []
    top_probs = []
    tf_probs = []
    acc_per_question = []
    confidence = []
    acc = 0
    
    # Load the dataset
    dataset = load_dataset(dataset_name, name = f"{num_shots}_shot")[split]#.select(range(10))
    num_batches = dataset.num_rows// BATCH_SIZE
    stoi = {dataset[0]['target']:dataset[0]['target_str'],dataset[1]['target']:dataset[1]['target_str']}
    s0 = stoi[0]
    s1 = stoi[1] 
    print(s0,s1)

    #### START CLOCK ####
    start_time = timeit.default_timer()
    # Evaluation loop
    for i_b in range(num_batches+1):
        # print(i_b)
        if i_b< num_batches:
            batch = dataset[i_b*BATCH_SIZE:(i_b+1)*BATCH_SIZE]#['input']
        else:
            if dataset.num_rows % BATCH_SIZE > 0:
                batch = dataset[i_b*BATCH_SIZE:]#['input']
            else:
                break

        input_ids = tokenizer(batch['input'], padding=True, truncation=True, return_tensors='pt').input_ids
        input_ids = input_ids.to(device=device)
        #print("DECODED INPUT:", repr(tokenizer.decode(input_ids.tolist()[0], skip_special_tokens=False)))
        #print(input_ids, input_ids.shape)
        
        ### Getting the position of the last non-padding token
        if BATCH_SIZE > 0: 
            ones = tokenizer.pad_token_id * torch.ones_like(input_ids)
            #print(input_ids == ones)
            last_token = input_ids == ones
            row_indices = torch.arange(input_ids.size(0))
            last_token = (torch.sum(last_token, dim = 1) + 1) * (-1)

        # if BATCH_SIZE == 1:
        #     last_token = input_ids[-1]
        #     row_indices = 0
        #     print("Last token", tokenizer.decode(input_ids[-1]), last_token)
        # else:
        #     
        #     ones = torch.ones_like(input_ids)
        #     last_token = input_ids == ones
        #     row_indices = torch.arange(input_ids.size(0))
        #     last_token = (torch.sum(last_token, dim = 1) + 1) * (-1)

        # Extra 1 from BoS and the beginning of each string.
        if tokenizer.bos_token_id == 1:
            last_token+=1

        # print(row_indices, last_token)
        print("Last token is", tokenizer.decode(input_ids[row_indices, last_token]))

        # Generate the model's response
        with torch.no_grad():
            output = model(input_ids)
            outs = output.logits.squeeze()
            if len(batch['input']) == 1:
                outs = outs.unsqueeze(0)

        #print(f"Model output shape: {outs.shape}")

        #print(f"Model output probs on last non-padding token: {softmax(outs[row_indices,last_token,:],dim=1).shape}")

        probabils, out_tokens = torch.topk(softmax(outs[row_indices,last_token,:],dim=1), top_k)
        out_tokens = out_tokens.detach().clone().cpu().tolist()
        probabils = probabils.detach().clone().cpu().tolist()
        top_tokens += out_tokens
        top_probs += probabils

        top_tokens_dec = tokenizer.batch_decode(out_tokens, skip_special_tokens=False)

        top_tokens_decoded += top_tokens_dec
        #print(f"Highest probability tokens are: {top_tokens_dec}, with probabilities: {probabils}")
        
        TF_idx = {
            tokenizer(s0, add_special_tokens=False).input_ids[0]: s0,
            tokenizer(s1, add_special_tokens=False).input_ids[0]: s1,
            }
        TF_idx_conn = {
            tokenizer("_"+s0, add_special_tokens=False).input_ids[-1]: s0,
            tokenizer("_"+s1, add_special_tokens=False).input_ids[-1]: s1,
            }
        TF_idx.update(TF_idx_conn)
        TF_ids_list = list(TF_idx.keys())
                #print(list(tokenizer.decode(adcd_ids_list, skip_special_tokens=False)))
        #print(last_token, "\n", input_ids)
        probs = softmax(outs[row_indices,last_token,:], dim=1)

        tf_predictions = probs[:,TF_ids_list]
        
        # normalize abcd_probs for saving later
        #abcd_predictions = abcd_predictions / abcd_predictions.sum(1).unsqueeze(1)
        tf_probs += tf_predictions.detach().clone().cpu().tolist()
        #print(f"Probabilities: {abcd_predictions} and the sum to 1: {abcd_predictions.sum(1)}")

        # for now dumb measure of confidence
        top_tf_probs, _ = torch.topk(tf_predictions, 2)
        confidence_per_question = torch.abs(top_tf_probs[:,0] - top_tf_probs[:,1])
        confidence += confidence_per_question.detach().clone().cpu().tolist()
        #print(f"shape of top_abcd_probs, {top_abcd_probs}", top_abcd_probs.shape)
        final_prediction = torch.argmax(tf_predictions, dim=1) % 2

        # print(f"Final prediction {final_prediction.data} vs Correct answer: {batch['target']}")
        answers += final_prediction.tolist()
        preds = final_prediction == torch.tensor(batch['target']).to(device=device)
        acc_per_question += preds.tolist()
        acc += torch.sum(preds)
    #print(f"Prediction array {answers}, number of predictions {len(answers)}, Totatl number of correct answers {acc}")
    #print(f"Top tokens: {top_tokens} with probs: {top_probs}. The ABCD probs are: {abcd_probs}")
    acc = acc.item() /  len(answers)
    print(f"Number of correct answers {acc}")

    # convert lists to a single dict

    #### END CLOCK ####
    end_time = timeit.default_timer()
    time_elapsed = start_time - end_time
    if details:
        eval_data = {'model' : model_name,
                    'accuracy' : acc, 
                    'accuracy_per_question' : acc_per_question,
                    'model_answers' : answers,
                    'tf_probs' : tf_probs,
                    'most_probable_tokens' : top_tokens,
                    'most_probable_tokens_decoded' : top_tokens_decoded,
                    'most_probable_tokens_probs' : top_probs}
    else:
        eval_data = {'model' : model_name,
                    'accuracy' : acc}

    # for key in eval_data.keys():
    #     print(f'\n\n====================={key}:=======================\n {eval_data[key]}')

    if save:
        # make directory where to save
        out =  model_name.split("/")[-1]+ "/" + dataset_name.split("/")[-1] + f"_{num_shots}"
        out_dir = os.path.join(out_path, out)
        # delete if dir already exists
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.mkdir(out_dir)
        # actiual saving of the logs
        for key in eval_data.keys():
            T = eval_data[key]
            log_path = os.path.join(out_dir, f"{key}.json")
            print(f"saving {key} to", log_path)
            with open(log_path, 'w') as f:
                json.dump(T, f)


if __name__ == "__main__":
    fire.Fire(evaluate_model)