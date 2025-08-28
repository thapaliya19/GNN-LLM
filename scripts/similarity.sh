#!/bin/bash
#SBATCH -C gpu                              # a100 or similar GPUs
#SBATCH --partition=general-gpu             # partition name
#SBATCH --nodes=1                           # run on one node
#SBATCH --ntasks=1                          # one task per job
#SBATCH --gres=gpu:1                        # allocate 1 GPU
#SBATCH --cpus-per-task=8                  # CPU cores per task
#SBATCH --time=12:00:00                     # time limit per job
#SBATCH --job-name=model_similarity
#SBATCH --output=similarity_logs/model_similarity_seed_%j.out

set -euxo pipefail

# 1) grab the seed from the command-line
mkdir -p similarity_logs

nvidia-smi

# 4) activate your environment
echo "Activating conda environment..."
source ~/miniconda3-10/etc/profile.d/conda.sh
conda activate base

# 5) run your experiment
python emb_similarity_nodes.py 