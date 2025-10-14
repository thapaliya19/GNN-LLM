#!/bin/bash
# ===== run_baseline_one.sh =====
set -euxo pipefail

seed=$1
dataset=$2
setting=$3
exp_slug=$4
iter=$5
tta_seed=$6
# drop first 6 args; remaining are node_texts
node_texts=( "${@:7}" )

export OPENAI_API_KEY="sk-proj-s6bnUu3_"


echo "[one] Running seed=${seed} on $(hostname)"
echo "[one] Node-texts: ${node_texts[*]}"

mkdir -p "${dataset}_logs_iter_${iter}"

nvidia-smi || true

# env
source ~/miniconda3-10/etc/profile.d/conda.sh
conda activate base

python autoprompt.py \
  --use_params \
  --data "${dataset}" \
  --label_setting number \
  --train_num 20 \
  --val_num 30 \
  --emb_method TTA \
  --backbone gcn \
  --node_text "${node_texts[@]}" \
  --seed "${seed}" \
  --topo_csv "/home/ldn24004/autoprompt/datasets/response/${setting}/${dataset}/topo_features.csv" \
  --llm gpt-4o-mini \
  --temperature 0.1 \
  --buffer_size 20 \
  --buffer_k 5 \
  --tta_max_iters 5 \
  --struct_epochs 20 \
  --demos_method combined \
  --tta_topk 0 \
  --desc_mode orig_only \
  --output_dir runs \
  --desc_csv /home/ldn24004/autoprompt/runs_openai/${dataset}/gpt-4o-mini/global_rewrite_iter${iter}_${dataset}_gpt-4o-mini_seed_${tta_seed}.csv 
  # --desc_csv "/home/ldn24004/autoprompt/datasets/response/${setting}/${dataset}/answer_gpt-4o-mini_TANS.csv" 