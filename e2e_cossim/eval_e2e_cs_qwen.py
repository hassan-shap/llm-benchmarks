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
num_batches = 40

model_name = "Qwen/Qwen-7B"
# model_name = "microsoft/phi-2"
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

if "Qwen" in model_name:
    tokenizer = AutoTokenizer.from_pretrained(model_name, pad_token='<|extra_0|>', trust_remote_code=True)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side = 'right', truncation_side='right')

if tokenizer.pad_token_id == None:
    tokenizer.pad_token_id = 1

tokenizer.padding_side = "right"

######## dataset import ########
dataset = load_dataset('c4',name = 'en', split='validation', streaming=True)

def collate_fn(examples):
    prompts = [example['text'] for example in examples]
    return {'input': prompts}

data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
device = "cuda" if torch.cuda.is_available() else "cpu"

orig_data_dir = "base_data/"
orig_fname = orig_data_dir+ f"tf_output_c4_{model_name.split('/')[-1]}.json"
# orig_fname = orig_data_dir+ f"output_c4_{model_name.split('/')[-1]}.json"
f = open(orig_fname)
orig_vec = torch.tensor(json.load(f))
f.close()
# print(orig_vec.shape, orig_vec.dtype)

block_size_list = range(24,0,-1)
num_layers = len(base_model.transformer.h)
for block_size in block_size_list:
    cos_sim = []
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

            embed_model = base_model.transformer.wte
            rotary_emb= base_model.transformer.rotary_emb
            input_embeds = embed_model(input_ids) 

            past_key_values_length = 0
            attention_mask = _prepare_4d_causal_attention_mask(
                        attention_mask, (1, seq_length),
                        input_embeds, past_key_values_length,
                        )

            ntk_alpha_list = [1.0]
            rotary_emb._ntk_alpha_cached_list = ntk_alpha_list
            rotary_pos_emb_list = [
                rotary_emb(seq_length, ntk_alpha=ntk_alpha) for ntk_alpha in ntk_alpha_list
            ]            
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
                for idx, block in enumerate(base_model.transformer.h):
                    if idx not in skip_layers:
                        x = block(x, attention_mask=attention_mask, rotary_pos_emb_list=rotary_pos_emb_list )[0]
                    # x_list.append(x[row_indices,last_token,:].to(torch.float32).cpu())

                x = base_model.transformer.ln_f(x)
                # x = base_model.lm_head(x)

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
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_sim.append(cos(orig_vec,new_vec).tolist())

    out_dir = "skip_data/"
    fname = out_dir+ f"l_{block_size}_tf_output_c4_{model_name.split('/')[-1]}.json"
    with open(fname, 'w') as f:
        json.dump(cos_sim, f)

    toc = time.time()

    # print("last_token_sim:", last_token_sim.shape)
    print(f"Elapsded time: {toc-tic}")

