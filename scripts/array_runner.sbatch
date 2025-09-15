#!/bin/bash
# This file is the *payload* of each job. Resource flags are passed from the submitter.
# You can add SBATCH defaults here if you want, but CLI args from the submitter override them.

set -euxo pipefail

# Required env vars (exported by the submitter)
: "${DATASET:?}" "${SEED:?}" "${EX_BLOCK_TEMPLATE:?}" "${SIM_METHOD:?}" "${BUFFER_K:?}"
: "${SETTING:?}" "${NODE_TEXT:?}"
: "${OUTPUT_ROOT:?}" "${LOG_ROOT:?}" "${GROUP:?}" "${BUFFER_ALPHA:?}"

# Human-readable slug to isolate outputs
EXP_SLUG="${DATASET}-gemma3-12b-eb:${EX_BLOCK_TEMPLATE}-sim:${SIM_METHOD}-bk:${BUFFER_K}-ba:${BUFFER_ALPHA}-seed:${SEED}"

OUT_DIR="${OUTPUT_ROOT}/${EXP_SLUG}"
RUN_LOG_DIR="${LOG_ROOT}/${EXP_SLUG}"
mkdir -p "${OUT_DIR}" "${RUN_LOG_DIR}" ollama_logs

# Also mirror stdout/err to a per-run file (Slurm will write to --output/--error too)
exec > >(tee -a "${RUN_LOG_DIR}/seed_${SEED}_${SLURM_JOB_ID}.out") 2> >(tee -a "${RUN_LOG_DIR}/seed_${SEED}_${SLURM_JOB_ID}.err" >&2)

# Unique Ollama port per job (avoid collisions)
BASE_PORT=11434
PORT=$(( BASE_PORT + SLURM_JOB_ID % 10000 ))
export OLLAMA_HOST="http://127.0.0.1:${PORT}"
echo "Starting Ollama server..."
ollama serve &> "ollama_logs/ollama_${SLURM_JOB_ID}.log" &
OLLAMA_PID=$!
sleep 10


# Activate your environment (adjust if you use modules instead)
if [[ -f ~/miniconda3-10/etc/profile.d/conda.sh ]]; then
  source ~/miniconda3-10/etc/profile.d/conda.sh
  conda activate base
fi

nvidia-smi || true

# Map dataset to response dir
if [[ "${DATASET}" == "cora" || "${DATASET}" == "pubmed" ]]; then
  setting_dir="text_rich"
else
  setting_dir="text_free"
fi

# Convert NODE_TEXT "title_abstract" → bash array ["title","abstract"]
if [[ "${NODE_TEXT}" == *"_"* ]]; then
  IFS='_' read -r -a NODE_TEXT_ARR <<< "${NODE_TEXT}"
else
  NODE_TEXT_ARR=("${NODE_TEXT}")
fi

python autoprompt.py \
  --use_params \
  --data "${DATASET}" \
  --label_setting ratio \
  --emb_method TTA \
  --backbone gcn \
  --node_text "${NODE_TEXT_ARR[@]}" \
  --seed "${SEED}" \
  --desc_csv "/home/ldn24004/autoprompt/datasets/response/${setting_dir}/${DATASET}/answer_gemma3-12b_TANS.csv" \
  --topo_csv "/home/ldn24004/autoprompt/datasets/response/${setting_dir}/${DATASET}/topo_features.csv" \
  --llm gemma3-12b \
  --buffer_k "${BUFFER_K}" \
  --sim_method "${SIM_METHOD}" \
  --ex_block_template "${EX_BLOCK_TEMPLATE}" \
  --buffer_alpha "${BUFFER_ALPHA}" \
  --rewrite_all \
  --rewrite_iters 2 \
  --struct_epochs 20 \
  --desc_mode orig_only \ 
  --output_dir "${OUT_DIR}" \
  --group "${GROUP}"

CODE=$?
kill ${OLLAMA_PID} || true
exit ${CODE}
