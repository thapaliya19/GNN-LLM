#!/bin/bash
#SBATCH -C gpu,a100
#SBATCH --partition=general-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euxo pipefail

seed=$1
dataset=$2
setting=$3
exp_slug=$4
iter=$5
tta_seed=$6
node_texts=( "${@:7}" )

echo "Running seed=${seed} on $(hostname)"
echo "Node-texts: ${node_texts[*]}"

mkdir -p "${dataset}_logs_iter_${iter}"

nvidia-smi

echo "Activating conda environment..."
# ### NEW: ensure conda shell functions are present
source ~/miniconda3-10/etc/profile.d/conda.sh
# (optional) use a dedicated env instead of base:
# conda activate autoprompt
conda activate base


python autoprompt.py \
  --use_params \
  --data "${dataset}" \
  --label_setting ratio \
  --train_num 20 \
  --val_num 30 \
  --emb_method TTA \
  --backbone gcn \
  --node_text ${node_texts[@]} \
  --seed "${seed}" \
  --desc_csv "/home/ldn24004/autoprompt/runs/${exp_slug}/${dataset}/gemma3-12b/global_rewrite_iter${iter}_${dataset}_gemma3-12b_seed_${tta_seed}.csv" \
  --topo_csv "/home/ldn24004/autoprompt/datasets/response/${setting}/${dataset}/topo_features.csv" \
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


  # --desc_csv "/home/ldn24004/autoprompt/datasets/response/${setting}/${dataset}/answer_gemma3-12b_TANS.csv" \