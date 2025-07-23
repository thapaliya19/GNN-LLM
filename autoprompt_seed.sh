#!/bin/bash
#SBATCH -C gpu                              # a100 or similar GPUs
#SBATCH --partition=general-gpu             # partition name
#SBATCH --nodes=1                           # run on one node
#SBATCH --ntasks=1                          # one task per job
#SBATCH --gres=gpu:1                        # allocate 1 GPU
#SBATCH --cpus-per-task=8                  # CPU cores per task
#SBATCH --time=12:00:00                     # time limit per job

set -euxo pipefail

# 1) grab the seed from the command-line
seed=$1
dataset=$2
setting=$3
# drop $1 and $2, everything else is node_texts
node_texts=( "${@:4}" )

echo "Running seed=${seed} on $(hostname)"
echo "Node‐texts: ${node_texts[*]}"

BASE_PORT=11434
PORT=$(( BASE_PORT + seed ))

export OLLAMA_HOST="http://127.0.0.1:${PORT}"
echo "OLLAMA_HOST=${OLLAMA_HOST}"

# 2) make sure logs directories exist
mkdir -p ${dataset}_logs ollama_logs

# 3) start Ollama
echo "Starting Ollama server..."
ollama serve &> ollama_logs/ollama_seed_${seed}.log &
OLLAMA_PID=$!
sleep 10  # give it time to warm up

# 4) activate your environment
echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate new_llm

# 5) run your experiment
python autoprompt.py \
  --use_params \
  --data ${dataset} \
  --label_setting ratio \
  --emb_method TTA \
  --backbone gcn \
  --node_text ${node_texts[@]} \
  --seed ${seed} \
  --desc_csv /home/ldn24004/autoprompt/datasets/response/${setting}/${dataset}/answer_gemma3-12b_TANS.csv \
  --topo_csv /home/ldn24004/autoprompt/datasets/response/${setting}/${dataset}/topo_features.csv \
  --llm gemma3-12b \
  --temperature 0.1 \
  --buffer_size 20 \
  --buffer_k 5 \
  --tta_max_iters 5 \
  --entropy_thr 0.5

EXIT_CODE=$?

# 6) tear down Ollama
echo "Shutting down Ollama (PID=$OLLAMA_PID)"
kill $OLLAMA_PID || true

exit $EXIT_CODE
