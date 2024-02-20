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
# from utils.make_prompt import prepend_prompt, pre_process_dataset


def collate_fn(examples):
    prompts = [example['questions']['question']  for example in examples]
    #print(prompts[0])
    answers = [example['questions']['answer'] for example in examples]
    return {'input': prompts, 'target': answers}

model_name = "mistralai/Mistral-7B-v0.1"
base = True
tuned_model_dir = ""
first_layer_dropped = 0
layer_drop_pattern = "cos-sim"
BATCH_SIZE = 64
instruction = ''
dataset_name = "hassansh/boolq_n_shot"
split = 'test' 
num_shots = 0 
top_k = 10
save = True 
out_path = "cossim_results"
details = False

# def evaluate_model(model_name : str = 'None',
#                    base : bool = False,
#                    tuned_model_dir : str = '',
#                    first_layer_dropped : str = '',
#                    num_layer_dropped : str = '',
#                    layer_drop_pattern : str = 'end',
#                    BATCH_SIZE : int = 2,
#                    instruction : str = '',
#                    dataset_name : str = '',
#                    subject_indices : str = '',
#                    split : str = 'test', 
#                    num_shots : int = 0, 
#                    top_k : int = 10,
#                    save : bool = False, 
#                    out_path : str = '', 
#                    details : bool = True):


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

for num_layer_dropped in range(1,25):
    #### START CLOCK ####
    start_time = timeit.default_timer()

    torch.cuda.empty_cache()
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

            if layer_drop_pattern == 'cos-sim':
                if "Llama-2-7b" in model_name:
                    first_layer = [23, 24, 23, 23, 23, 22, 22, 21, 20, 19, 18, 18, 17, 16, 15, 8, 8, 7, 4, 4, 4, 2, 2, 2]
                elif "Llama-2-13b" in model_name:
                    first_layer = [33, 32, 31, 30, 29, 28, 27, 27, 27, 26, 25, 24, 24, 23, 22, 22, 20, 19, 18, 18, 17, 17, 15, 3, 3, 3, 3, 3, 3, 3, 3, 3]
                elif "Llama-2-70b" in model_name:
                    first_layer = [62, 62, 61, 62, 61, 60, 61, 58, 59, 58, 58, 57, 56, 55, 54, 53, 53, 53, 53, 53, 51, 50, 49, 50, 49, 49, 48, 48, 46, 46, 45, 44, 43, 42, 41, 40, 39, 37, 35, 35, 33, 33, 29, 30, 29, 28, 28, 28, 26, 25, 23, 22, 22, 21, 20, 19, 19, 18, 16, 16, 14, 14, 13, 10]
                elif "Mistral-7B" in model_name:
                    first_layer = [26, 24, 23, 23, 23, 22, 21, 21, 20, 20, 19, 17, 17, 16, 13, 13, 11, 8, 8, 6, 6, 5, 4, 3]

                layers_dropped = list(range(first_layer[num_layer_dropped-1], first_layer[num_layer_dropped-1]+ num_layer_dropped))
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
    
    


    # Prepare variables to collect MMLU data boolq
    answers = []
    top_tokens = []
    top_tokens_decoded = []
    top_probs = []
    YN_probs = []
    acc_per_question = []
    confidence = []
    time = []
    acc = 0
 
    # Load the dataset
    boolQ = load_dataset('hassansh/boolq_n_shot')[f'{num_shots}_shot']

    instruction = f"The following are multiple choice questions (with answers) about \n\n"

    # nshot_prompt = prepend_prompt(instruction, nshot_MMLU, n=num_shots, shuffle = False)

    # print(nshot_prompt)

    #processed_MMLU = Dataset.from_dict(pre_process_dataset(instruction, nshot_prompt, MMLU))

    data_loader = DataLoader(boolQ, batch_size=BATCH_SIZE)

    # #### START CLOCK ####
    # start_time = timeit.default_timer()
    # Evaluation loop
    for batch in data_loader:
        #Prepare the prompt with few-shot examples
        # print("STRUCTURE BELOW")
        # for key in example.keys():
        #     print(f"{key}: {example[key]}")

        # print(f"BATCH:{batch} LENGTH: {len(batch['input'])}")

        # print("=======PROMPT TEXT======")
        # print(batch['input'])

        #input_ids = tokenizer(prompt_text, return_tensors='pt').input_ids max_length=2047, padding="max_length",
        #input_ids = tokenizer(batch['input'], padding=True, return_tensors='pt').input_ids

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
        #print("Last token is", tokenizer.decode(input_ids[row_indices, last_token]))

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
        YN_ixs = {tokenizer("No").input_ids[-1] : "No",
                    tokenizer("Yes").input_ids[-1] : "Yes"}        

        YN_ids_list = list(YN_ixs.keys())
        #print(tokenizer.decode(YN_ids_list, skip_special_tokens=False))
        #print(last_token, "\n", input_ids)
        probs = softmax(outs[row_indices,last_token,:], dim=1)

        YN_predictions = probs[:,YN_ids_list]
        
        # normalize abcd_probs for saving later
        #abcd_predictions = abcd_predictions / abcd_predictions.sum(1).unsqueeze(1)
        YN_probs += YN_predictions.detach().clone().cpu().tolist()
        #print(f"Probabilities: {abcd_predictions} and the sum to 1: {abcd_predictions.sum(1)}")

        final_prediction = torch.argmax(YN_predictions, dim=1)

        # print(f"Final prediction {final_prediction.data} vs Correct answer: {batch['target']}")
        answers += final_prediction.tolist()
        preds = final_prediction == torch.tensor(batch['target']).to(device=device)
        acc_per_question += preds.tolist()
        acc += torch.sum(preds)
        # print(f"Number of correct answers {acc}")
    #print(f"Prediction array {answers}, number of predictions {len(answers)}, Totatl number of correct answers {acc}")
    #print(f"Top tokens: {top_tokens} with probs: {top_probs}. The ABCD probs are: {abcd_probs}")
    acc = acc.item() /  len(answers)

    # convert lists to a single dict

    #### END CLOCK ####
    end_time = timeit.default_timer()
    time_elapsed =  end_time - start_time
    if details:
        eval_data = {'model' : model_name,
                    'accuracy' : acc, 
                    'accuracy_per_question' : acc_per_question,
                    'model_answers' : answers,
                    'YN_probs' : YN_probs,
                    'most_probable_tokens' : top_tokens,
                    'most_probable_tokens_decoded' : top_tokens_decoded,
                    'most_probable_tokens_probs' : top_probs,
                    'time' : time_elapsed}
    else:
        eval_data = {'model' : model_name,
                    'accuracy' : acc}

    # for key in eval_data.keys():
    #     print(f'\n\n====================={key}:=======================\n {eval_data[key]}')
    print(model_name, f"time: {time_elapsed}, acc: {acc}")

    if save:
        # make directory where to save
        out_prefix = f"{model_name.split('/')[-1]}"
        out_path_new = os.path.join(out_path, out_prefix)

        out =  f"num_ldrop_{num_layer_dropped}_shots_{num_shots}"
        out_dir = os.path.join(out_path_new, out)
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

    del model

# if __name__ == "__main__":
#     fire.Fire(evaluate_model)
