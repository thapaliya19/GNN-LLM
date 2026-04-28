#!/bin/bash
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

source /home/ldn24004/miniconda3-10/etc/profile.d/conda.sh
conda activate tans

echo "=== Experiment Start ==="
echo "Command: $@"
echo "Date: $(date)"
echo "========================"

eval "$@"

echo "=== Experiment End (exit: $?) ==="
echo "Date: $(date)"
exit $?
