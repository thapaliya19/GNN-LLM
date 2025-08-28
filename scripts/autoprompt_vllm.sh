#!/bin/bash

#SBATCH -C a100,gpu                        # constraints; searches for a100 GPUs
#SBATCH --partition=general-gpu            # partition name
#SBATCH --ntasks=8                         # number of CPUs
#SBATCH --nodes=1                          # all CPUs in same node
#SBATCH --gres=gpu:1                       # number of GPUs
#SBATCH --output=prompt-optimize-logs/prompt-optimize-%j.out

set -euxo pipefail  # bail on error, print commands as they run

~/miniconda3/envs/new_llm/bin/python autoprompt.py --use_params --data europe \
    --label_setting ratio \
    --emb_method TTA --backbone gcn --node_text none \
    --seed 1 2 3 \
    --desc_csv /home/ldn24004/autoprompt/datasets/response/text_free/europe/answer_gemma3-1b.csv \
    --topo_csv /home/ldn24004/autoprompt/datasets/response/text_free/brazil/topo_features.csv \
    --llm gemma3-1b --temperature 0.1 \
    --buffer_size 20 --buffer_k 5 \
    --tta_max_iters 10 --entropy_thr 0.5 
