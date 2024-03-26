import torch
from datasets import load_dataset
from transformers import (
AutoModelForCausalLM,
AutoTokenizer,
BitsAndBytesConfig,
)
import json
import time 
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from torch.utils.data import DataLoader

def cross_entropy(input, target):
    return -torch.sum(target * torch.log(input), dim = 1)

def KL_divergence(input, target):
    return torch.sum(target * torch.log(target/input), dim = 1)

BATCH_SIZE = 4
num_batches = 40
Temp_list = [2.0,0.5,1.0]

# model_name = "meta-llama/Llama-2-70b-hf"
model_name = "mistralai/Mistral-7B-v0.1"

torch.cuda.empty_cache()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    )
device_map = {"": 0}
# Load model
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map,
    # cache_dir = "/dev/shm/hassan/.cahce/"
    )

base_model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if "Llama-2" in model_name or "Mistral" in model_name:
# tokenizer.pad_token_id = tokenizer.bos_token_id
    tokenizer.pad_token = tokenizer.bos_token

tokenizer.padding_side = "right"

######## dataset import ########
dataset = load_dataset('c4',name = 'en', split='validation', streaming=True)

def collate_fn(examples):
    prompts = [example['text'] for example in examples]
    return {'input': prompts}

data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
device = "cuda" if torch.cuda.is_available() else "cpu"

orig_data_dir = "base_data/"
# orig_fname = orig_data_dir+ f"tf_output_c4_{model_name.split('/')[-1]}.json"
orig_fname = orig_data_dir+ f"output_c4_{model_name.split('/')[-1]}.json"
f = open(orig_fname)
# print(json.load(f))
orig_vec = torch.tensor(json.load(f))
f.close()
# print(orig_vec.shape)

block_size_list = range(24,0,-1)
num_layers = len(base_model.model.layers)
for block_size in block_size_list:
    distance = {}
    for Temp in Temp_list:
        distance[f"{Temp}"] = []
    # print(block_size, " : " )
    tic = time.time()
    for layer in range(num_layers-block_size+1):
        skip_layers = list(range(layer,layer+block_size))
        print(skip_layers)#, end='\r')

        last_token_vec = []
        i = 0 
        for batch in data_loader:
            i += 1
            if i > num_batches:
                break

            cutoff_len = 2000
            #  max_length=cutoff_len
            skip = False
            for text in batch['input']:
                # print(len(text.split(' ')))
                if len(text.split(' '))> cutoff_len:
                    skip = True
                    print(i, ' skipped')

            if skip:
                continue

            tokens = tokenizer(batch['input'], padding = True, truncation=True , return_tensors='pt')
            input_ids = tokens.input_ids.to(device=device)
            # print(input_ids.shape)
            seq_length = input_ids.size(1)
            attention_mask = tokens.attention_mask.to(device=device)

            input_embeds = base_model.model.embed_tokens(input_ids) 

            attention_mask = _prepare_4d_causal_attention_mask(
                        attention_mask, (BATCH_SIZE, seq_length),
                        input_embeds, 0)
            
            ### Getting the position of the last non-padding token
            ones = tokenizer.pad_token_id * torch.ones_like(input_ids)
            last_token = input_ids == ones
            row_indices = torch.arange(input_ids.size(0))
            last_token = (torch.sum(last_token, dim = 1) + 1) * (-1)

            # Extra 1 from BoS and the beginning of each string.
            if tokenizer.bos_token_id == 1:
                last_token+=1

            x = input_embeds
            with torch.no_grad():
                for idx, block in enumerate(base_model.model.layers):
                    if idx not in skip_layers:
                        x = block(x,attention_mask=attention_mask)[0]
                    # x_list.append(x[row_indices,last_token,:].to(torch.float32).cpu())

                x = base_model.model.norm(x)
                x = base_model.lm_head(x)


            last_token_vec.append(x[row_indices,last_token,:])

            # if i % 10 == 0 :
            #     out_dir = "skip_data/"
            #     fname = out_dir+ f"skip_output_c4_{model_name.split('/')[-1]}.json"
            #     last_token_sim_save= torch.cat(last_token_vec, dim = 1).tolist()
            #     with open(fname, 'w') as f:
            #         json.dump(last_token_sim_save, f)

        # print(len(last_token_vec))
        new_vec= torch.cat(last_token_vec, dim = 0).to(torch.float32).cpu()
        # print(new_vec.shape)
        # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        # cos_sim.append(cos(orig_vec,new_vec).tolist())
        # print(new_vec.shape)
        for Temp in Temp_list:
            orig_probs = torch.nn.functional.softmax(orig_vec/Temp, dim = 1)
            distance[f"{Temp}"].append(KL_divergence(torch.nn.functional.softmax(new_vec/Temp,dim=1), orig_probs).tolist())#torch.nn.functional.cross_entropy(new_vec,orig_vec).tolist())
        # print(cos_sim[0].shape)

    out_dir = "skip_data/"
    # fname = out_dir+ f"l_{block_size}_tf_output_c4_{model_name.split('/')[-1]}.json"
    for Temp in Temp_list:
        fname = out_dir+ f"KL_T_{Temp:.1f}_l_{block_size}_output_c4_{model_name.split('/')[-1]}.json"
        with open(fname, 'w') as f:
            json.dump(distance[f"{Temp}"], f)

    toc = time.time()

    # print("last_token_sim:", last_token_sim.shape)
    print(f"Elapsded time: {toc-tic}")

