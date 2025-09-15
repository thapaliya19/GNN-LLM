#!/bin/bash
dataset=$1
setting=$2
exp_slug=$3
iter=$4
tta_seed=$5
node_texts=( "${@:6}" )

for seed in {1..30}; do
  sbatch \
    --job-name=${dataset}_tta_seed_${seed} \
    --output=${dataset}_logs/${exp_slug}/iter_${iter}/tta_seed_${seed}.out \
    --error=${dataset}_logs/${exp_slug}/iter_${iter}/tta_seed_${seed}.err \
    run_baseline_exp.sh ${seed} ${dataset} ${setting} ${exp_slug} ${iter} ${tta_seed} ${node_texts[@]}
  echo "Submitted job for seed=${seed}, dataset=${dataset}, exp_slug=${exp_slug}, setting=${setting}, iter=${iter}, tta_seed=${tta_seed}, node_texts=${node_texts[@]}"
  # sleep 1  # optional: to avoid overwhelming the scheduler with too many jobs at once
done
