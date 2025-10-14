#!/bin/bash
# ===== run_packed_gpu.sbatch =====
#SBATCH -C gpu
#SBATCH --partition=general-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=packgpu
#SBATCH --output=grid_logs/%x_%j.out
#SBATCH --error=grid_logs/%x_%j.err

set -euo pipefail

dataset=$1
pt_dataset=$2
src_desc_csv=$3
tgt_desc_csv=$4
log_file_name=$5
PAR=${6:-4}
shift 6

# ---- parse: seeds until first "--", then node_texts ----
seeds=()
while [[ $# -gt 0 && "$1" != "--" ]]; do seeds+=("$1"); shift; done
[[ $# -gt 0 && "$1" == "--" ]] && shift
node_texts=( "$@" )   # everything after "--" is node_texts

echo "Node: $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "PAR=$PAR   seeds=(${seeds[*]})"
echo "node_texts: ${node_texts[*]}"
nvidia-smi || true


# --- after you parse args and set PAR, seeds, node_texts ---

# Build the list of CPUs we are actually allowed to use in this allocation
allowed_raw=$(taskset -cp $$ 2>/dev/null | awk -F: '{gsub(/ /,"",$2); print $2}')
expand_cpuset() {
  # input like "0-3,8,10-11" -> prints "0 1 2 3 8 10 11"
  IFS=',' read -r -a parts <<< "$1"
  out=()
  for p in "${parts[@]}"; do
    if [[ "$p" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      a=${BASH_REMATCH[1]}; b=${BASH_REMATCH[2]}
      for ((i=a;i<=b;i++)); do out+=("$i"); done
    elif [[ "$p" =~ ^[0-9]+$ ]]; then
      out+=("$p")
    fi
  done
  printf '%s\n' "${out[@]}"
}

mapfile -t ALLOWED_CPUS < <(expand_cpuset "$allowed_raw")
TOTAL_CPUS=${#ALLOWED_CPUS[@]}

if (( TOTAL_CPUS == 0 )); then
  echo "[warn] Could not determine allowed CPUs (got '$allowed_raw'). Will not pin."
fi

# compute per-worker bundles
make_bundle() {
  # args: worker_index(0..PAR-1)
  local idx=$1
  if (( TOTAL_CPUS == 0 )); then
    echo "" ; return 0
  fi
  # round-robin assignment for fairness
  ids=()
  for ((i=idx; i<TOTAL_CPUS; i+=PAR)); do ids+=("${ALLOWED_CPUS[i]}"); done
  # join with commas (taskset accepts comma-separated list)
  local IFS=','; echo "${ids[*]}"
}

pids=()
w=0
for seed in "${seeds[@]}"; do
  # throttle to <= PAR concurrent
  while (( ${#pids[@]} >= PAR )); do
    if wait -n; then :; fi
    alive=(); for pid in "${pids[@]}"; do kill -0 "$pid" 2>/dev/null && alive+=("$pid"); done
    pids=("${alive[@]}")
  done

  bundle=$(make_bundle $w)
  logdir="da_logs/${pt_dataset}-${dataset}/"
  mkdir -p "$logdir"

  echo "Launching seed=${seed} on CPUs ${bundle:-<un-pinned>}"
  if [[ -n "$bundle" ]]; then
    taskset -c "$bundle" \
      bash run_da_exp.sh \
        "${seed}" "${dataset}" "${pt_dataset}" "${src_desc_csv}" "${tgt_desc_csv}" \
        "${node_texts[@]}" \
        >"${logdir}/tta_seed_${seed}.out" \
        2>"${logdir}/tta_seed_${seed}.err" &
  else
    # fallback: no pinning
    bash run_da_exp.sh \
      "${seed}" "${dataset}" "${pt_dataset}" "${src_desc_csv}" "${tgt_desc_csv}" \
      "${node_texts[@]}" \
      >"${logdir}/tta_seed_${seed}.out" \
      2>"${logdir}/tta_seed_${seed}.err" &
  fi

  pids+=( $! )
  w=$(( (w + 1) % PAR ))
done

status=0; for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit $status
