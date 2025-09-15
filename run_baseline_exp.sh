#!/bin/bash
#SBATCH -C gpu                              # a100 or similar GPUs
#SBATCH --partition=general-gpu             # partition name
#SBATCH --nodes=1                           # run on one node
#SBATCH --ntasks=1                          # one task per job
#SBATCH --gres=gpu:1                        # allocate 1 GPU
#SBATCH --cpus-per-task=8                  # CPU cores per task

set -euxo pipefail

# 1) grab the seed from the command-line
seed=$1
dataset=$2
setting=$3
exp_slug=$4
iter=$5
tta_seed=$6
# drop $1 and $2, everything else is node_texts
node_texts=( "${@:7}" )

echo "Running seed=${seed} on $(hostname)"
echo "Node‐texts: ${node_texts[*]}"

mkdir -p ${dataset}_logs_iter_${iter}

nvidia-smi

echo "Activating conda environment..."
source ~/miniconda3-10/etc/profile.d/conda.sh
conda activate base

python autoprompt.py \
  --use_params \
  --data ${dataset} \
  --label_setting ratio \
  --train_num 20 \
  --val_num 30 \
  --emb_method TTA \
  --backbone gcn \
  --node_text ${node_texts[@]} \
  --seed ${seed} \
  --desc_csv /home/ldn24004/autoprompt/runs/${exp_slug}/${dataset}/gemma3-12b/global_rewrite_iter${iter}_${dataset}_gemma3-12b_seed_${tta_seed}.csv \
  --topo_csv /home/ldn24004/autoprompt/datasets/response/${setting}/${dataset}/topo_features.csv \
  --llm gemma3-12b \
  --temperature 0.1 \
  --buffer_size 20 \
  --buffer_k 5 \
  --tta_max_iters 5 \
  --struct_epochs 20 \
  --demos_method combined \
  --tta_topk 0 \
  --desc_mode llm_only \
  --output_dir runs 

EXIT_CODE=$?

# 6) tear down Ollama

exit $EXIT_CODE
