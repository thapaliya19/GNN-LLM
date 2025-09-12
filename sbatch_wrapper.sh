#!/bin/bash
dataset=$1
setting=$2
iter=$3
tta_seed=$4
buffer_alpha=$5
node_texts=( "${@:6}" )

for seed in {1..30}; do
  sbatch \
    --job-name=${dataset}_tta_seed_${seed} \
    --output=${dataset}_logs_iter_${iter}/tta_seed_${seed}.out \
    --error=${dataset}_logs_iter_${iter}/tta_seed_${seed}.err \
    run_baseline_exp.sh ${seed} ${dataset} ${setting} ${iter} ${tta_seed} ${buffer_alpha} ${node_texts[@]}
  echo "Submitted job for seed=${seed}, dataset=${dataset}, setting=${setting}, iter=${iter}, tta_seed=${tta_seed}, buffer_alpha=${buffer_alpha}, node_texts=${node_texts[@]}"
  sleep 1  # optional: to avoid overwhelming the scheduler with too many jobs at once
done
