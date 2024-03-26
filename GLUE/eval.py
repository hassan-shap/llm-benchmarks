
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    logging,
)

import time 
import json

from datasets.utils.logging import disable_progress_bar, enable_progress_bar

# ## Load model
# 
# model_name = "/data/opt-350m"
# model_name = "mistralai/Mistral-7B-Instruct-v0.2"
# model_name = "/data/llama-hf/Llama-2-7b-hf"
model_name = "mistralai/Mistral-7B-v0.1"
# model_name = "meta-llama/Llama-2-7b-hf"

cutoff_len = 4096
torch.cuda.empty_cache()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)
device_map = {"": 0}
# Load model
# base_model = AutoModelForCausalLM.from_pretrained(
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map,
    # cache_dir = "/dev/shm/hassan/.cahce/"
)

# depth = len(base_model.base_model.layers)
# num_removed_layers = 10
# first_layer_dropped = depth - num_removed_layers -1
# layers_dropped = list(range(first_layer_dropped, depth-1))
# print(f"We are removing: {layers_dropped} layers")
# base_model.base_model.layers = torch.nn.ModuleList([block for idx, block in enumerate(base_model.base_model.layers) if idx not in layers_dropped])
# model = base_model
# del base_model

model.eval()

model.config.use_cache = False
model.config.pretraining_tp = 1

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if "Llama-2" in model_name or "Mistral" in model_name:
    print("Doing Llama tokenizer thingy")
    # tokenizer.pad_token_id = tokenizer.bos_token_id
    tokenizer.pad_token = tokenizer.bos_token

tokenizer.padding_side = "right"

bos = tokenizer.bos_token_id
eos = tokenizer.eos_token_id
pad = tokenizer.pad_token_id
print("pre-trained model's BOS EOS and PAD token id:",bos,eos,pad," => It should be 1 2 None")

s0 = "No" #"False"
s1 = "Yes" #"True"
TF_idx = [
    tokenizer(s0, add_special_tokens=False).input_ids[0],
    tokenizer(s1, add_special_tokens=False).input_ids[0],
 ]

TF_idx_conn = [
    tokenizer("_"+s0, add_special_tokens=False).input_ids[-1],
    tokenizer("_"+s1, add_special_tokens=False).input_ids[-1],
 ]
TF_idx_double = TF_idx + TF_idx_conn
idx_TF = [s0, s1]
print(TF_idx_double)
choice_idx = {s0 :0, s1:1}

n_shot_vals = range(6)

disable_progress_bar()

outfname  = "boolq-" + model_name.split('/')[1] + '.json'
out_json = "results/" + outfname

with open(out_json, 'w') as json_file:

    for i_n, n_shot in enumerate(n_shot_vals):
        tic = time.time()
        # if n_shot == 0:
        #     batch_size = 40
        if n_shot <= 1:
            batch_size = 64
        elif n_shot == 2:
            batch_size = 32
        elif n_shot == 3:
            batch_size = 16
        elif n_shot == 4:
            batch_size = 8
        else:
            batch_size = 4

        # dataset = load_dataset("json", data_files={
        #         # 'test' : f"boolq-data/{n_shot}_shot_examples.json"
        #         'test' : f"boolq-data/{n_shot}_shot_examples_small.json"
        #     })

        dataset = load_dataset("hassansh/boolq_n_shot", name = f"{n_shot}_shot")
        dataset = dataset['test'].select(range(10))
        num_batches = dataset.num_rows// batch_size

        acc = 0
        acc_TF = 0

        for i_b in range(num_batches+1):
            # print(i_b)
            if i_b< num_batches:
                ex1 = dataset[i_b*batch_size:(i_b+1)*batch_size]['input']
                answers = dataset[i_b*batch_size:(i_b+1)*batch_size]['output']
            else:
                if dataset.num_rows % batch_size > 0:
                    # print(len(mmlu_dataset['input']),i_b*batch_size)
                    ex1 = dataset[i_b*batch_size:]['input']
                    answers = dataset[i_b*batch_size:]['output']
                else:
                    break

            # input_ids = tokenizer(ex1, padding='max_length', return_tensors='pt', max_length=cutoff_len).input_ids
            input_ids = tokenizer(ex1, padding=True, return_tensors='pt').input_ids
            input_ids = input_ids.to(device=0)

            with torch.no_grad():
                output = model(input_ids)
                answers_2 = output.logits.squeeze()
                if len(ex1) == 1:
                    answers_2 = answers_2.unsqueeze(0)


            ones = torch.ones_like(input_ids)
            last_token = input_ids == ones
            row_indices = torch.arange(input_ids.size(0))
            last_token = (torch.sum(last_token, dim = 1) + 1) * (-1)
            if "Llama-2" in model_name or "Mistral" in model_name:
                last_token+=1

            top_choices_TF = torch.argmax(answers_2[row_indices,last_token,:][:,TF_idx_double],dim=-1) % 2
            # print(top_choices_TF)
            # print(answers)
            # final_prediction = [ tokenizer.decode(TF_idx_double[i])[-1] for i in top_choices_TF]
            acc_TF += torch.sum(top_choices_TF.cpu() == torch.tensor(answers)).item()

            # top_choices = torch.argmax(answers_2[row_indices,last_token,:], dim= -1)
            # print(top_choices)
            # print(tokenizer.decode(top_choices))
            # print(len(answers))
            # top_choices_txt = tokenizer.decode(top_choices).split()
            # print(top_choices_txt)
            # print(len(top_choices_txt))
            for i_a, answer in enumerate(answers):
                # print(i_r, answers[i_r])
                response = tokenizer.decode(torch.argmax(answers_2[i_a,last_token[i_a],:], dim = -1))
                # print(f"({idx_TF[answer]}, {response}): {response == idx_TF[answer]}")
                # print(f"({answer}, {top_choices_TF[i_a]}): {top_choices_TF[i_a] == answer}")
                # print(f"{''.join(['_']*20)}")
                print(response, idx_TF[answer], idx_TF[answer]== response)
                acc += (response == idx_TF[answer])

        acc /= dataset.num_rows
        acc_TF /= dataset.num_rows
        print(n_shot, dataset.num_rows, acc, acc_TF)

        toc = time.time()

        print("Elapsed time: ", toc-tic, " sec")

        eval_data = {'n_shot' : n_shot,
                'accuracy' : acc, 
                'accuracy_TF' : acc_TF, 
                # 'cross_entropy' : cross_entropy, 
                'time' : toc-tic}

        json_file.write(json.dumps(eval_data) + '\n')

# dataset = load_dataset("json", data_files={
#              'test': out_json} )
# dataset.push_to_hub("hassansh/boolq-"+ model_name.split('/')[1])
