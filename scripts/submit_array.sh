#!/bin/bash
set -euo pipefail

# ---------- User-tunable defaults (Storrs “general”) ----------
MANIFEST="scripts/manifest.tsv"

PARTITION="general-gpu"      # Storrs default
NODES=1
NTASKS=1
CPUS_PER_TASK=8
CONSTRAINTS="a100"        # e.g., a100 or gpu ; leave empty if not required
GRES="gpu:1"               # e.g., gpu:1 ; leave empty if not required
JOB_NAME_PREFIX="exp"    # job name prefix; helps throttling filter
MAX_INFLIGHT=20          # cap number of running+pending jobs from this launcher
SLEEP_BASE=3             # seconds between submissions
# --------------------------------------------------------------

if [[ ! -s "$MANIFEST" ]]; then
  echo "Manifest not found or empty: $MANIFEST" >&2
  exit 1
fi

echo "Using manifest: $MANIFEST"

# Count data rows (minus header)
N=$(awk 'END{print NR-1}' "$MANIFEST")
if (( N <= 0 )); then
  echo "No data rows in manifest (has header only?): $MANIFEST" >&2
  exit 1
fi

echo "Found ${N} experiment configurations in manifest."

# Helpers
count_inflight() {
  # count user's jobs whose name begins with JOB_NAME_PREFIX
  squeue -u "$USER" -h -o '%j' | grep -c "^${JOB_NAME_PREFIX}" || true
}

submit_one() {
  local dataset seed ex_block_template sim_method buffer_k \
        setting node_text \
        output_root log_root group buffer_alpha

  dataset="$1"; seed="$2"; ex_block_template="$3"; sim_method="$4"; buffer_k="$5"
  setting="$6"; node_text="$7"; output_root="$8"; log_root="$9"; group="${10}"; buffer_alpha="${11}"

  # Build the same slug the payload uses (mirror formula)
  local exp_slug="${dataset}-gemma3-12b-eb:${ex_block_template}-sim:${sim_method}-bk:${buffer_k}-ba:${buffer_alpha}-seed:${seed}"
  echo "Submitting: dataset=${dataset}, seed=${seed}, ex_block_template=${ex_block_template}, sim_method=${sim_method}, buffer_k=${buffer_k}, setting=${setting}, node_text=${node_text}, output_root=${output_root}, log_root=${log_root}, group=${group}, buffer_alpha=${buffer_alpha}"
  local out_dir="${output_root}/${exp_slug}"
  local run_log_dir="${log_root}/${exp_slug}"
  mkdir -p "${run_log_dir}" grid_logs

  # Per-run slurm logs (KB suggests unique files; use %j)
  local out_path="${run_log_dir}/seed_${seed}_%j.out"
  local err_path="${run_log_dir}/seed_${seed}_%j.err"

  # Build resource flags
  local RB=()
  RB+=(--partition="${PARTITION}")
  RB+=(--job-name="${JOB_NAME_PREFIX}_${dataset}")
  RB+=(--nodes="${NODES}")
  RB+=(--ntasks="${NTASKS}")
  RB+=(--cpus-per-task="${CPUS_PER_TASK}")
  RB+=(--output="${out_path}")
  RB+=(--error="${err_path}")
  if [[ -n "${GRES}" ]]; then RB+=(--gres="${GRES}"); fi
  if [[ -n "${CONSTRAINTS}" ]]; then RB+=(--constraint="${CONSTRAINTS}"); fi

    # Build a SINGLE --export string (this was the main bug)
  local EXPORTS="ALL"
  EXPORTS+=",DATASET=${dataset}"
  EXPORTS+=",SEED=${seed}"
  EXPORTS+=",EX_BLOCK_TEMPLATE=${ex_block_template}"
  EXPORTS+=",SIM_METHOD=${sim_method}"
  EXPORTS+=",BUFFER_K=${buffer_k}"
  EXPORTS+=",SETTING=${setting}"
  EXPORTS+=",NODE_TEXT=${node_text}"
  EXPORTS+=",OUTPUT_ROOT=${output_root}"
  EXPORTS+=",LOG_ROOT=${log_root}"
  EXPORTS+=",GROUP=${group}"
  EXPORTS+=",BUFFER_ALPHA=${buffer_alpha}"

  # Submit
  local jid
  jid=$(sbatch --parsable "${RB[@]}" --export="${EXPORTS}" scripts/array_runner.sbatch)
  echo "Submitted ${jid} → ${dataset}, seed=${seed}, slug=${exp_slug}"
}

# Read header then iterate rows
{
  read -r header
  i=0
  while IFS=$'\t' read -r dataset seed ex_block_template sim_method buffer_k setting node_text output_root log_root group buffer_alpha
  do
    echo "[$((i+1))/${N}] Preparing to submit: dataset=${dataset}, seed=${seed}, ex_block_template=${ex_block_template}, sim_method=${sim_method}, buffer_k=${buffer_k}, setting=${setting}, node_text=${node_text}, output_root=${output_root}, log_root=${log_root}, group=${group}, buffer_alpha=${buffer_alpha}"
    i=$((i+1))

    submit_one "$dataset" "$seed" "$ex_block_template" "$sim_method" "$buffer_k" \
               "$setting" "$node_text" "$output_root" "$log_root" "$group" "$buffer_alpha"

    echo "Submitted job for dataset=${dataset}, seed=${seed}, ex_block_template=${ex_block_template}, sim_method=${sim_method}, buffer_k=${buffer_k}, setting=${setting}, node_text=${node_text}, output_root=${output_root}, log_root=${log_root}, group=${group}, buffer_alpha=${buffer_alpha}"

    # small randomized backoff to avoid bursts
    sleep $(( SLEEP_BASE + (RANDOM % 3) ))
  done
} < "$MANIFEST"
