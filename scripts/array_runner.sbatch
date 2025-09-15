#!/bin/bash
#SBATCH -C gpu
#SBATCH --partition=general-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=1-{{N}}%8          # set N in submitter script
#SBATCH --job-name=grid
#SBATCH --output=grid_logs/%x_%A_%a.out
#SBATCH --error=grid_logs/%x_%A_%a.err

set -euxo pipefail
MANIFEST="scripts/manifest.tsv"
IDX=${SLURM_ARRAY_TASK_ID}

mkdir -p grid_logs ollama_logs

# parse the TSV row (skip header)
LINE=$(sed -n "$((IDX+1))p" "$MANIFEST")
IFS=$'\t' read -r dataset seed ex_block_template sim_method buffer_k setting node_text <<<"$LINE"

# node_text 'title+abstract' → bash array ["title","abstract"]
IFS='+' read -r -a NODE_TEXT_ARR <<< "$node_text"

# optional: unique port per task to avoid Ollama collision
BASE_PORT=11434
PORT=$(( BASE_PORT + IDX ))
export OLLAMA_HOST="http://127.0.0.1:${PORT}"

# start Ollama (comment out if not needed for this run)
ollama serve &> "ollama_logs/ollama_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" &
OLLAMA_PID=$!
sleep 10

source ~/miniconda3-10/etc/profile.d/conda.sh
conda activate base
nvidia-smi

# datasets -> paths
if [[ "$dataset" == "cora" || "$dataset" == "pubmed" ]]; then
  setting_dir="text_rich"
else
  setting_dir="text_free"
fi

python autoprompt.py \
  --use_params \
  --data "$dataset" \
  --label_setting ratio \
  --train_num 20 --val_num 30 \
  --emb_method TTA \
  --backbone gcn \
  --node_text "${NODE_TEXT_ARR[@]}" \
  --seed "$seed" \
  --desc_csv "/home/ldn24004/autoprompt/datasets/response/${setting_dir}/${dataset}/answer_gemma3-12b_TANS.csv" \
  --topo_csv "/home/ldn24004/autoprompt/datasets/response/${setting_dir}/${dataset}/topo_features.csv" \
  --llm gemma3-12b \
  --temperature 0.1 \
  --buffer_size 20 \
  --buffer_k "$buffer_k" \
  --struct_epochs 20 \
  --sim_method "$sim_method" \
  --tta_topk 0 \
  --desc_mode orig_only \
  --output_dir output_ratio_iters \
  --rewrite_all \
  --rewrite_iters 2 \
  --group "grid_${dataset}" \
  --ex_block_template "$ex_block_template"

EXIT_CODE=$?

kill $OLLAMA_PID || true
exit $EXIT_CODE
