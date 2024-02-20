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

BATCH_SIZE = 8
num_batches = 4000

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

tic = time.time()
device = "cuda" if torch.cuda.is_available() else "cpu"
last_token_sim = []
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

    if not skip:
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

        similarity = []
        with torch.no_grad():
            x_old = input_embeds
            for idx, block in enumerate(base_model.model.layers):
                # print(x_old.shape)
                # print(x_new[row_indices,last_token,:].shape)
                x_new = block(x_old,attention_mask=attention_mask)[0]
                # similiarity.append(cos(x_new,x_old).cpu())
                similarity.append(cos(x_new[row_indices,last_token,:].to(torch.float32),x_old[row_indices,last_token,:].to(torch.float32)).cpu())
                x_old = x_new
            x = base_model.model.norm(x_new)
            x = base_model.lm_head(x)
        similarity = torch.stack(similarity).squeeze()
        # print("sim:", similarity.shape)
        last_token_sim.append(similarity)

    if i % 10 == 0 :
        out_dir = "data/"
        fname = out_dir+ f"c4_{model_name.split('/')[-1]}_2.json"
        last_token_sim_save= torch.cat(last_token_sim, dim = 1).tolist()
        with open(fname, 'w') as f:
            json.dump(last_token_sim_save, f)

toc = time.time()

# print("last_token_sim:", last_token_sim.shape)
print(f"Elapsded time: {toc-tic}")

