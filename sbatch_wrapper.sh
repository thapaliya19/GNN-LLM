#!/bin/bash
dataset=$1
setting=$2
buffer_alpha=$3
embed_alpha=$4
entropy_thr=$5
node_texts=( "${@:6}" )

for seed in {1..30}; do
  sbatch \
    --job-name=${dataset}_tta_seed_${seed} \
    --output=${dataset}_logs_iter_0/tta_seed_${seed}.out \
    --error=${dataset}_logs_iter_0/tta_seed_${seed}.err \
    run_baseline_exp.sh ${seed} ${dataset} ${setting} ${buffer_alpha} ${embed_alpha} ${entropy_thr} ${node_texts[@]}
  echo "Submitted job for seed=${seed}, dataset=${dataset}, setting=${setting}, buffer_alpha=${buffer_alpha}, embed_alpha=${embed_alpha}, entropy_thr=${entropy_thr}, node_texts=${node_texts[@]}"
  sleep 1  # optional: to avoid overwhelming the scheduler with too many jobs at once
done
