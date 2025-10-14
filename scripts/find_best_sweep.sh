#!/bin/bash
#SBATCH --nodes=1                           # run on one node
#SBATCH --ntasks=1                          # one task per job
#SBATCH --cpus-per-task=8                  # CPU cores per task
#SBATCH --time=12:00:00                     # time limit per job

set -euxo pipefail


# 4) activate your environment
echo "Activating conda environment..."
source ~/miniconda3-10/etc/profile.d/conda.sh
conda activate base

# 5) run your experiment
python scripts/find_best_sweep.py