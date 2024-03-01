torchrun --nproc_per_node=1 --master_port=1234 eval_andrey_fire_multirc.py \
    --model_name /data/llama-hf/Llama-2-7b-hf \
    --base True \
    --first_layer_dropped -1 \
    --BATCH_SIZE 8 \
    --dataset_name hassansh/multirc_n_shot \
    --num_shots 0 \
    --top_k 10 \
    --save True \
    --out_path gromovand \
    --details True
    # --model_name mistralai/Mistral-7B-v0.1 \

