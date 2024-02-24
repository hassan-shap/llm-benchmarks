torchrun --nproc_per_node=1 --master_port=1234 eval_andrey_fire_multirc.py \
    --model_name mistralai/Mistral-7B-v0.1 \
    --base True \
    --first_layer_dropped 27 \
    --BATCH_SIZE 32 \
    --dataset_name hassansh/boolq_n_shot \
    --num_shots 0 \
    --top_k 10 \
    --save True \
    --out_path gromovand \
    --details False
    # --model_name meta-llama/Llama-2-7b-hf \

