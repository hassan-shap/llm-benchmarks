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

BATCH_SIZE = 4
num_batches = 10000
step_list = torch.arange(1,25)

# model_name = "/data/llama-hf/Llama-2-7b-hf"
# model_name = "mistralai/Mistral-7B-v0.1"
model_name = "microsoft/phi-2"
local_dir="/data/models"

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
    cache_dir = local_dir,
    pad_token_id=1,
    trust_remote_code = True
)

base_model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token_id == None:
    tokenizer.pad_token_id = 1

tokenizer.padding_side = "right"

######## dataset import ########
dataset = load_dataset('c4',name = 'en', split='validation', streaming=True)
# n_shot = 0
# fname = f'{n_shot}_shot_examples_no_space_100.json'
# mmlu_dataset = load_dataset("json", data_files={
#             'test': '/data/opt-ft/mmlu-data/'+fname,
#         })
# dataset = mmlu_dataset['test']
# print(dataset)

def collate_fn(examples):
    prompts = [example['text'] for example in examples]
    # prompts = [example['input'] for example in examples]
    return {'input': prompts}

data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
device = "cuda" if torch.cuda.is_available() else "cpu"

########

similarity = {}
for step in step_list:
    similarity[f"{step}"] = []

tic = time.time()
# last_token_sim = []
i = 0 
for batch in data_loader:
    i += 1
    if i < num_batches+1:
        print(i, end='\r')
        # print(batch['input'])
        # print(len(batch['input']))
    else:
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
    
    input_embeds = base_model.model.embed_tokens(input_ids) 

    ### Getting the position of the last non-padding token
    ones = tokenizer.pad_token_id * torch.ones_like(input_ids)
    last_token = input_ids == ones
    row_indices = torch.arange(input_ids.size(0))
    last_token = (torch.sum(last_token, dim = 1) + 1) * (-1)


    # Extra 1 from BoS and the beginning of each string.
    if tokenizer.bos_token_id == 1:
        last_token+=1

    # print(row_indices, last_token)
    # print("Last token is", tokenizer.decode(input_ids[row_indices, last_token]))
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    num_layers = len(base_model.model.layers)
    x = input_embeds
    x_list = []
    x_list.append(x[row_indices,last_token,:].to(torch.float32).cpu())
    with torch.no_grad():
        for idx, block in enumerate(base_model.model.layers):
            x = block(x,attention_mask=attention_mask)[0]
            x_list.append(x[row_indices,last_token,:].to(torch.float32).cpu())

    for step in step_list:
        sim_layer = []
        for idx in range(num_layers-step+1):
            # print("cos: ", cos(x_list[idx],x_list[idx+step]).shape)
            sim_layer.append(cos(x_list[idx],x_list[idx+step]))
        sim_layer = torch.stack(sim_layer,dim=1)
        # print("sim_layer: ", sim_layer.shape)
        similarity[f"{step}"].append(sim_layer)


    if i % 100 == 0 :
        last_token_sim = {}
        for step in step_list:
            last_token_sim[f"{step}"]= torch.cat(similarity[f"{step}"], dim=0).tolist()

        out_dir = "data/"
        fname = out_dir+ f"c4_{model_name.split('/')[-1]}_steps.json"
        # fname = out_dir+ f"mmlu_{model_name.split('/')[-1]}_steps.json"
        with open(fname, 'w') as f:
            json.dump(last_token_sim, f)

last_token_sim = {}
for step in step_list:
    last_token_sim[f"{step}"]= torch.cat(similarity[f"{step}"], dim=0).tolist()

out_dir = "data/"
fname = out_dir+ f"c4_{model_name.split('/')[-1]}_steps.json"
# fname = out_dir+ f"mmlu_{model_name.split('/')[-1]}_steps.json"
with open(fname, 'w') as f:
    json.dump(last_token_sim, f)

toc = time.time()

# print("last_token_sim:", last_token_sim.shape)
print(f"Elapsded time: {toc-tic}")

