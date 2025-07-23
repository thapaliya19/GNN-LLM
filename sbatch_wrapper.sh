#!/bin/bash
dataset=$1
setting=$2
node_texts=( "${@:3}" )

for seed in 1 2; do
  sbatch \
    --job-name=${dataset}_tta_seed_${seed} \
    --output=${dataset}_logs/tta_seed_${seed}.out \
    --error=${dataset}_logs/tta_seed_${seed}.err \
    autoprompt_seed.sh ${seed} ${dataset} ${setting} ${node_texts[@]}
  echo "Submitted job for seed=${seed}, dataset=${dataset}, setting=${setting}, node_texts=${node_texts[@]}"
    sleep 1  # optional: to avoid overwhelming the scheduler with too many jobs at once
done
