#!/bin/bash
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

source /home/ldn24004/miniconda3-10/etc/profile.d/conda.sh
conda activate base

echo "Starting vLLM on $(hostname):8000"
echo "NODE=$(hostname)" > /home/ldn24004/autoprompt/vllm_server_info.txt

python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-72B-Instruct-AWQ \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 \
  --port 8000
