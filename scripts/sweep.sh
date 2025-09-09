#!/bin/bash

#SBATCH -C gpu                           # constraints; searches for a100 gpus
#SBATCH --partition=general-gpu         # partition name
#SBATCH --ntasks=8                      # number of cpus
#SBATCH --nodes=1                       # all cpus in same node
#SBATCH --gres=gpu:1                    # number of gpus
#SBATCH --job-name=sweep               # job name
#SBATCH --output=sweep-logs/%j/sweep-%j.out  # unique log directory per job

# Number of agents to run
NUM_AGENTS=4
SWEEP_ID=thapaliya19/autoprompt/89s59aec  # Replace with your actual sweep ID

# Create a unique log directory for this job
LOG_DIR="sweep-logs/$SLURM_JOB_ID"
mkdir -p "$LOG_DIR"

# Run agents in parallel with job-specific logs
for i in $(seq 1 $NUM_AGENTS); do
  nohup wandb agent "$SWEEP_ID" > "$LOG_DIR/agent_${SLURM_JOB_ID}_$i.log" 2>&1 &
done

# Wait for all agents to finish
wait
