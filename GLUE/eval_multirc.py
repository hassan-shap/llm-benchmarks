from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_metric
import torch
import sys
from torch.utils.data import Subset
import random
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
from datasets.utils.logging import disable_progress_bar,enable_progress_bar

device = "cuda:0"
device_map = "cuda:0"
instruction = None
BATCH_SIZE = 4
top_k = 10
        

def evaluate_model(model = None,
                   tokenizer = None,
                   dataset_name : str = '',
                   split : str = '', 
                   num_shots : int = 0, 
                   save : bool = False, 
                   out_path : str = '', 
                   details : bool = False):



    # LLama2 does not have a padding token so I pad with BoS

    tokenizer.pad_token_id = 1

    # Load the dataset
    dataset = load_dataset(dataset_name, name = f"{num_shots}_shot")[split]

    # Create the n-shot string to append
    # if num_shots > 0:
    #     dataset_size = len(dataset)
    #     random_indices = random.sample(range(dataset_size), num_shots)
    #     random_subset = Subset(dataset, random_indices)

    #     for i in range(num_shots)

    # Prepare variables to collect MMLU data
    answers = []
    top_tokens = []
    top_tokens_decoded = []
    top_probs = []
    tf_probs = []
    acc_per_question = []
    confidence = []
    acc = 0
    num_batches = dataset.num_rows// BATCH_SIZE

    disable_progress_bar()
    num_paragraphs = 0
    qs_per_paragraph = []
    for i in range(dataset.num_rows):
        if dataset.filter(lambda example: example['question']==i):
            num_paragraphs += 1
            
        else:
            break
    enable_progress_bar()
    print(num_paragraphs)
    acc_paragraph = torch.zeros(num_paragraphs)

    # Evaluation loop
    for i_b in range(num_batches+1):
        # print(i_b)
        if i_b< num_batches:
            batch = dataset[i_b*BATCH_SIZE:(i_b+1)*BATCH_SIZE]
        else:
            if dataset.num_rows % BATCH_SIZE > 0:
                batch = dataset[i_b*BATCH_SIZE:]
            else:
                break

        #input_ids = tokenizer(prompt_text, return_tensors='pt').input_ids
        input_ids = tokenizer(batch['input'], padding=True, return_tensors='pt').input_ids
        input_ids = input_ids.to(device=device)
        #print("DECODED INPUT:", repr(tokenizer.decode(input_ids.tolist()[0], skip_special_tokens=False)))
        #print(input_ids, input_ids.shape)

        ### Getting the position of the last non-padding token
        ones = torch.ones_like(input_ids)
        last_token = input_ids == ones
        row_indices = torch.arange(input_ids.size(0))
        last_token = (torch.sum(last_token, dim = 1) + 1) * (-1)

        # Extra 1 from BoS and the beginning of each string.
        if tokenizer.bos_token_id == 1:
            last_token+=1

        # print(row_indices, last_token)
        # print("Last token", input_ids[row_indices, last_token])

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

        TF_idx = [
            tokenizer(s0, add_special_tokens=False).input_ids[0],
            tokenizer(s1, add_special_tokens=False).input_ids[0],
        ]
        TF_idx_conn = [
            tokenizer("_"+s0, add_special_tokens=False).input_ids[-1],
            tokenizer("_"+s1, add_special_tokens=False).input_ids[-1],
        ]
        TF_ids_list = TF_idx + TF_idx_conn
        

        # print(tokenizer.decode(adcd_ids_list, skip_special_tokens=False))
        # print(last_token, "\n", input_ids)
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
        final_prediction = torch.argmax(outs[row_indices,last_token,:][:,TF_ids_list], dim=1) % 2
        # print(f"Final prediction {final_prediction.data} vs Correct answer: {batch['target']}")
        answers += final_prediction.tolist()
        preds = final_prediction == torch.tensor(batch['target']).to(device=device)
        acc_per_question += preds.tolist()
        acc += torch.sum(preds)
        acc_paragraph[batch['question']] += preds 
        # print(f"Number of correct answers {acc}")
    #print(f"Prediction array {answers}, number of predictions {len(answers)}, Totatl number of correct answers {acc}")
    #print(f"Top tokens: {top_tokens} with probs: {top_probs}. The ABCD probs are: {abcd_probs}")
    acc = acc.item() /  len(answers)
    # convert lists to a single dict

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
        out =  dataset_name.split("/")[-1] + f"_{num_shots}"
        # add day stamp
        today = date.today()
        out = out 
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
    return acc


# Example usage
model_name = "meta-llama/Llama-2-7b-hf"  # Replace with your model's name
#model_name = "mistralai/Mistral-7B-v0.1"
#model_name = "facebook/opt-125m" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"EOS {tokenizer.eos_token_id} BOS {tokenizer.bos_token_id} PAD {tokenizer.pad_token_id}")
############################################################
################### only changing these ####################
#### boolq ########
# dataset_name = "hassansh/boolq_n_shot"  # The boolq dataset name in Hugging Face Datasets
# s0 = "No" #"False"
# s1 = "Yes" #"True"
# #### copa ########
# dataset_name = "hassansh/copa_n_shot"  # The copa dataset name in Hugging Face Datasets
# s0 = "1" #"choice1"
# s1 = "2" #"choice2"
# #### rte ########
# dataset_name = "hassansh/rte_n_shot"  # The rte dataset name in Hugging Face Datasets
# s0 = "False" #"not_entailment"
# s1 = "True" #"entailment"
# # #### multirc ########
dataset_name = "hassansh/multirc_n_shot"  # The rte dataset name in Hugging Face Datasets
s0 = "False" #"not_entailment"
s1 = "True" #"entailment"
# ############################################################
split = "test"  # Specify the dataset split (e.g., 'test')
save = True
out_path = 'gromovand/'
details = True
try:
    os.makedirs(out_path)
    print(f"Directory '{out_path}' created successfully")
except FileExistsError:
    print(f"Directory '{out_path}' already exists")

#### Load the pretrained model and tokenizer ####

#16 bit
# model = AutoModelForCausalLM.from_pretrained(model_name)
# model = model.to(device = device)

# 4bit
bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            quantization_config=bnb_config,
            device_map=device_map
        )
# # #PEFT LORA LOAD
# MODEL_PATH = "/fsx-scaling/ktirumala/andrey_pruning_runs/layer_drop_platypus_13b/dec_25_lora_r_64_layer_start_drop_35"

# blocks_to_remove = [35, 36, 37, 38]
# print(f"We are removing: {blocks_to_remove} layers")
# base_model.base_model.layers = nn.ModuleList([block for idx, block in enumerate(base_model.base_model.layers) if idx not in blocks_to_remove])

# model = PeftModel.from_pretrained(
#         base_model,
#         MODEL_PATH,
#         torch_dtype=torch.bfloat16,
#     )
# del base_model

model.eval()

for n_shot in range(6):
    score = evaluate_model(model,
                        tokenizer,
                        dataset_name = dataset_name, 
                        split = split, 
                        num_shots = n_shot,
                        save = save,
                        out_path = out_path,
                        details = details)

    print(f"Model performance on {n_shot} is {score}")