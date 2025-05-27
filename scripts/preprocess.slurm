#!/usr/bin/env bash
#SBATCH --partition=gpu-H100
#SBATCH --job-name=preprocess # Name of the process
#SBATCH --gres=gpu:1 # Number of GPUs
#SBATCH --cpus-per-gpu=2 # Number of CPU cores (2 is reasonable)
#SBATCH --mem-per-gpu=16GB # RAM memory needed (8-16GB)
#SBATCH --time=1-00:00 # 10 hours
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asudupe008@ikasle.ehu.eus
#SBATCH --output=.slurm/preprocess_out.log
#SBATCH --error=.slurm/preprocess_err.log
#SBATCH --chdir=/home/andoni.sudupe/mHubert_finetune

# Activate virtual environment
conda activate hubert

# Run the evaluation script with command line arguments
srun python -u /home/andoni.sudupe/mHubert_finetune/scripts/preprocess.py