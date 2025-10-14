#!/usr/bin/env bash
# ===== submit_packed.sh =====
set -euo pipefail

# -------- mappings --------
get_src_csv() {
  # Map pt_dataset -> src_desc_csv
  case "$1" in
    usa) echo "runs_openai/usa/gpt-4o-mini/global_rewrite_iter0_usa_gpt-4o-mini_seed_3.csv" ;;
    europe) echo "runs_openai/europe/gpt-4o-mini/global_rewrite_iter0_europe_gpt-4o-mini_seed_4.csv" ;;
    brazil) echo "runs_openai/brazil/gpt-4o-mini/global_rewrite_iter0_brazil_gpt-4o-mini_seed_5.csv" ;;
    *)
      echo "Error: No source CSV mapping found for pt_dataset '$1'" >&2
      exit 1
      ;;
  esac
}

get_tgt_csv() {
  # Map dataset -> tgt_desc_csv
  case "$1" in
    usa) echo "runs_openai/usa/gpt-4o-mini/global_rewrite_iter0_usa_gpt-4o-mini_seed_3.csv" ;;
    europe) echo "runs_openai/europe/gpt-4o-mini/global_rewrite_iter0_europe_gpt-4o-mini_seed_4.csv" ;;
    brazil) echo "runs_openai/brazil/gpt-4o-mini/global_rewrite_iter0_brazil_gpt-4o-mini_seed_5.csv" ;;
    *)
      echo "Error: No target CSV mapping found for dataset '$1'" >&2
      exit 1
      ;;
  esac
}
# --------------------------

dataset=$1
pt_dataset=$2
PAR=${3:-4}
shift 3 || true
node_texts=( "$@" )


# derive CSVs from mappings
src_desc_csv=$(get_src_csv "$pt_dataset")
tgt_desc_csv=$(get_tgt_csv "$dataset")

# seeds 1..30
mapfile -t seeds < <(seq 1 30)

mkdir -p grid_logs

i=0
while [[ $i -lt ${#seeds[@]} ]]; do
  chunk=( "${seeds[@]:$i:$PAR}" )
  i=$(( i + PAR ))

  sbatch \
    --job-name="${dataset}_pack" \
    --output="grid_logs/${dataset}_pack_%j.out" \
    --error="grid_logs/${dataset}_pack_%j.err" \
    run_packed_da_gpu.sbatch \
      "${dataset}" "${pt_dataset}" "${src_desc_csv}" "${tgt_desc_csv}" random_file_name "${PAR}" \
      "${chunk[@]}" -- "${node_texts[@]}"

  echo "Submitted packed job for seeds: ${chunk[*]}"
done
