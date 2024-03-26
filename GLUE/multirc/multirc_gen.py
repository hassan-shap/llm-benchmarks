from datasets import load_dataset, concatenate_datasets
import json
import torch
from datasets.utils.logging import disable_progress_bar,enable_progress_bar
import time

disable_progress_bar()

dataset_name = "super_glue"  # The MMLU dataset name in Hugging Face Datasets
dataset = load_dataset(dataset_name, name='multirc')#["train"]
# print(dataset)

labels = dataset["train"].features["label"]
# def create_label_str(batch):
#     return {"label_str": labels.int2str(batch["label"])}
def create_label_str(example):
    return {"label_str": "True" if example["label"]== 1 else "False"}
# print(labels)

# data_train = dataset["train"].map(create_label_str)   
# data_test = dataset["validation"].map(create_label_str)   
# # print(d1)
# # dd = d1.select(range(5)).shuffle()

d1 = dataset["train"].map(create_label_str)   
data_test = dataset["validation"].map(create_label_str)   

# d_all = concatenate_datasets([d1,d2])
train_size = 1000
data_train = d1.select(range(train_size))
# data_test = d2#.select(range(train_size,d_all.num_rows))

print(data_train.num_rows, data_test.num_rows)

# prepare examples
example_prompt = []
ex_inds = torch.randint(0, 80,(6,))
print(ex_inds)
for i, ex in enumerate(ex_inds):
    dd = data_train.filter(lambda example: example['idx']['question']==ex)
    idx = 0
    prompt = "Paragraph: " + dd[idx]['paragraph'] + '\n'
    for idx in [0]:
        prompt += "Question: " + dd[idx]['question'] + '\n'
        prompt += "Candiate answer: " + dd[idx]['answer'] + '\n'
        prompt += "Answer:" + dd[idx]['label_str'] 
        # prompt += "(True/False):" + dd[idx]['label_str'] 
        # prompt += dd[idx]['label_str'] #+ '\n\n'
    example_prompt.append(prompt)


def json_writer(JSON_PATH, num_qs, n_shot_example,seed=10,space=False):
    qlist = {}
    
    with open(JSON_PATH, 'w') as json_file:

        step = 100
        # ind = 0
        # while len(data_test.filter(lambda example: example['idx']['question']==ind))>0:
        for ind in range(num_qs):
            if ind % step == 0:
                print(f"{ind//step}")#, end='\r')

            idx_ex_shuffle = torch.randperm(5)
            init_prompt = ""#Based on the paragraph, determine whether the candidate answer to the question is True or False.\n\n"
            for i in range(n_shot_example):
                init_prompt +=  example_prompt[idx_ex_shuffle[i]] + '\n\n'

            dd = data_test.filter(lambda example: example['idx']['question']==ind)

            init_prompt += "Paragraph: " + dd[0]['paragraph'] + '\n'
            for idx_q in range(dd.num_rows):
                prompt = init_prompt
                prompt += "Question: " + dd[idx_q]['question'] + '\n'
                prompt += "Candidate answer: " + dd[idx_q]['answer'] + '\n'
                prompt += "Answer:"
                # prompt += "(True/False):"
                
                qlist['question'] = ind
                qlist['input'] = prompt
                qlist['target_str'] = dd[idx_q]['label_str']
                qlist['target'] = dd[idx_q]['label']

                 # Serializing json
                json_file.write(json.dumps(qlist) + '\n')
        # print("Done!")
                

tic = time.time()
# print(example_idx)
n_example = 953#data_test.num_rows
n_shot_vals = [2]
seed = 123
for n_shot in range(6):
# for n_shot in n_shot_vals:
    print(f"{n_shot}-shot example is being generated...")
    fname_json = f"multirc-data/{n_shot}_shot_examples.json"
    # fname_json = f"boolq-data/{n_shot}_shot_examples_small.json"
    # JSON_eval = 'sample_eval.json'
    json_writer(fname_json, n_example, n_shot, seed=seed, space= False)
    # json_writer(JSON_eval,example_idx[(n_shot_example+1)*n_test_ex:])

enable_progress_bar ()

fname_dict = {}
for n_shot in range(6):
    fname_dict[f"{n_shot}_shot"] = f"multirc-data/{n_shot}_shot_examples.json"
d_upload = load_dataset("json", data_files=fname_dict)
d_upload.push_to_hub("hassansh/multirc_n_shot")

toc = time.time()
print(f"Elapsed time: {toc-tic} secs")