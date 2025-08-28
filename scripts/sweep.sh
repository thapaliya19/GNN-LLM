#!/bin/bash

#SBATCH -C gpu # constraints; searches for a100 gpus
#SBATCH --partition=general-gpu # partition name
#SBATCH --ntasks=8 # number of cpus
#SBATCH --nodes=1 # all cpus in same node
#SBATCH --gres=gpu:1 # number of gpus
#SBATCH --job-name=sweep             # job name
#SBATCH --output=sweep-logs/sweep-%j.out

# Number of agents to run
NUM_AGENTS=2
SWEEP_ID=thapaliya19/autoprompt/vnt6ibmx # Replace with your actual sweep ID

# Run agents in parallel
for i in $(seq 1 $NUM_AGENTS); do
  nohup wandb agent $SWEEP_ID > agent_$SLURM_JOB_ID_$i.log 2>&1 &
done

# Wait for all agents to finish (optional)
wait
