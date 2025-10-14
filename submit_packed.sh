#!/bin/bash
# ===== submit_packed.sh =====
set -euo pipefail

dataset=$1
setting=$2
exp_slug=$3
iter=$4
tta_seed=$5
PAR=${6:-4}
shift 6
node_texts=( "$@" )

# seeds 1..30
mapfile -t seeds < <(seq 1 30)

i=0
while [[ $i -lt ${#seeds[@]} ]]; do
  chunk=( "${seeds[@]:$i:$PAR}" )
  i=$(( i + PAR ))

  sbatch \
    --job-name="${dataset}_pack" \
    --output="grid_logs/${dataset}_pack_%j.out" \
    --error="grid_logs/${dataset}_pack_%j.err" \
    run_packed_gpu.sbatch \
      "${dataset}" "${setting}" "${exp_slug}" "${iter}" "${tta_seed}" "${PAR}" \
      "${chunk[@]}" -- "${node_texts[@]}"

  echo "Submitted packed job for seeds: ${chunk[*]}"
done
