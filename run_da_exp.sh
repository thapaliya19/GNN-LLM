#!/bin/bash
#SBATCH -C gpu                           # a100 or similar GPUs
#SBATCH --partition=general-gpu             # partition name
#SBATCH --nodes=1                           # run on one node
#SBATCH --ntasks=1                          # one task per job
#SBATCH --gres=gpu:1                        # allocate 1 GPU
#SBATCH --cpus-per-task=8                  # CPU cores per task

set -euxo pipefail

# 1) grab the seed from the command-line
seed=$1
dataset=$2
pt_dataset=$3
src_desc_csv=$4
tgt_desc_csv=$5
# drop $1 and $2, everything else is node_texts
node_texts=( "${@:6}" )

echo "Running seed=${seed} on $(hostname)"
echo "Node‐texts: ${node_texts[*]}"


nvidia-smi

echo "Activating conda environment..."
source ~/miniconda3-10/etc/profile.d/conda.sh
conda activate base

python da.py \
  --use_params \
  --pt_data ${pt_dataset} \
  --data ${dataset} \
  --label_setting ratio \
  --train_ratio 0.0 \
  --emb_method TTA \
  --backbone gcn \
  --node_text ${node_texts[@]} \
  --seed ${seed} \
  --desc_csv ${src_desc_csv} \
  --topo_csv /home/ldn24004/autoprompt/datasets/response/text_free/${pt_dataset}/topo_features.csv \
  --tgt_desc_csv ${tgt_desc_csv} \
  --tgt_topo_csv /home/ldn24004/autoprompt/datasets/response/text_free/${dataset}/topo_features.csv \
  --llm gpt-4o-mini \
  --temperature 0.1 \
  --buffer_size 20 \
  --buffer_k 5 \
  --tta_max_iters 5 \
  --struct_epochs 20 \
  --demos_method combined \
  --tta_topk 0 \
  --desc_mode both \
  --output_dir runs

EXIT_CODE=$?


exit $EXIT_CODE
