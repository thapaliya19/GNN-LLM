#!/bin/bash
#SBATCH -C a100,gpu                              # a100 or similar GPUs
#SBATCH --partition=general-gpu             # partition name
#SBATCH --nodes=1                           # run on one node
#SBATCH --ntasks=1                          # one task per job
#SBATCH --gres=gpu:1                        # allocate 1 GPU
#SBATCH --cpus-per-task=8                  # CPU cores per task
#SBATCH --time=12:00:00                     # time limit per job
#SBATCH --job-name=model_explain
#SBATCH --output=explain_logs/model_explain_seed_%j.out

set -euxo pipefail

# 1) grab the seed from the command-line
dataset=$1
desc_csv=$2
# drop $1 and $2, everything else is node_texts

echo "Using description CSV: $desc_csv"

# 2) make sure logs directories exist
mkdir -p explain_logs

nvidia-smi

# 4) activate your environment
echo "Activating conda environment..."
source ~/miniconda3-10/etc/profile.d/conda.sh
conda activate base

# 5) run your experiment
# python explain_saved_model.py \
#   --use_params \
#   --data ${dataset} \
#   --label_setting ratio \
#   --emb_method TTA \
#   --backbone gcn \
#   --node_text ${node_texts[@]} \
#   --seed ${seed} \
#   --desc_csv ${desc_csv} \
#   --topo_csv /home/ldn24004/autoprompt/datasets/response/${setting}/${dataset}/topo_features.csv \
#   --llm gemma3-12b \
#   --temperature 0.1 \
#   --buffer_size 20 \
#   --buffer_k 5 \
#   --tta_max_iters 5 \
#   --entropy_thr 0.5 \
#   --struct_epochs 20 \
#   --save_model

# python homophily_analysis.py --dataset ${dataset} --data-root ./datasets --desc-csv ${desc_csv}
python cora_text_correlation.py --data-path ./datasets --desc-csv ${desc_csv} --topo-csv /home/ldn24004/autoprompt/datasets/response/text_rich/cora/topo_features.csv

