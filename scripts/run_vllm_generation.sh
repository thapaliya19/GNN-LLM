#!/bin/bash
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

cd /home/ldn24004/autoprompt

echo "=== Starting vLLM server (base env) ==="
source /home/ldn24004/miniconda3-10/etc/profile.d/conda.sh
conda activate base

python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-72B-Instruct-AWQ \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --port 8000 &
VLLM_PID=$!

echo "Waiting for vLLM to start..."
for i in $(seq 1 120); do
  curl -s http://localhost:8000/v1/models > /dev/null 2>&1 && break
  sleep 5
  echo "Waiting... (${i}/120)"
done

curl -s http://localhost:8000/v1/models && echo "vLLM ready!" || { echo "vLLM failed to start"; kill $VLLM_PID; exit 1; }

echo "=== Generating descriptions (tans env) ==="
conda activate tans

python scripts/generate_llm_descriptions.py \
  --api_base http://localhost:8000/v1 \
  --model Qwen/Qwen2.5-72B-Instruct-AWQ \
  --datasets cora pubmed \
  --methods tans das tape kea

echo "Shutting down vLLM..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
echo "=== DONE ==="
