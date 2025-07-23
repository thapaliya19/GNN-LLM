#!/bin/bash
#SBATCH --job-name=add_node_idx
#SBATCH --output=add_node_idx.%A.out
#SBATCH --error=add_node_idx.%A.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --partition=general

# Load Python module if required
# module load python/3.8

# 4) activate your environment
echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate new_llm

# Directory containing your CSVs
INPUT_DIR=/home/ldn24004/autoprompt/output/europe/gemma3-12b

# Run the in-place processing script
srun python process_csvs.py \
    --input_dir "$INPUT_DIR" \
    --workers "$SLURM_CPUS_PER_TASK"
