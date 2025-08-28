#!/bin/bash
#SBATCH -C a100,gpu
#SBATCH --partition=general-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=tta_seed_%a
#SBATCH --output=cora_logs/tta_seed_%a.out
#SBATCH --error=cora_logs/tta_seed_%a.err
#SBATCH --array=1-30%5
#SBATCH --time=12:00:00

set -euxo pipefail

# derive seed from array index
echo "Running seed=${SLURM_ARRAY_TASK_ID} on $(hostname)"

# create directories for logs and output
mkdir -p cora_logs ollama_logs

# start Ollama server in background
echo "Starting Ollama server..."
ollama serve &> ollama_logs/ollama_seed_${SLURM_ARRAY_TASK_ID}.log &
OLLAMA_PID=$!

# give Ollama time to initialize
echo "Waiting for Ollama (PID=$OLLAMA_PID) to warm up..."
sleep 10

# activate conda environment (adjust path as needed)
echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate new_llm

# run the experiment for this seed
python autoprompt.py \
  --use_params \
  --data cora \
  --label_setting ratio \
  --emb_method TTA \
  --backbone gcn \
  --node_text title abstract \
  --seed ${SLURM_ARRAY_TASK_ID} \
  --desc_csv /home/ldn24004/autoprompt/datasets/response/text_free/cora/answer_gemma3-12b_TANS.csv \
  --topo_csv /home/ldn24004/autoprompt/datasets/response/text_free/cora/topo_features.csv \
  --llm gemma3-12b \
  --temperature 0.1 \
  --buffer_size 20 \
  --buffer_k 5 \
  --tta_max_iters 5 \
  --entropy_thr 0.5

# capture exit code
EXIT_CODE=$?

echo "Experiment finished with exit code $EXIT_CODE"

# shutdown Ollama server
echo "Shutting down Ollama (PID=$OLLAMA_PID)"
kill $OLLAMA_PID || true


exit $EXIT_CODE
