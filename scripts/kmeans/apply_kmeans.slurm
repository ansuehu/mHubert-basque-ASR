#!/usr/bin/env bash
#SBATCH --partition=gpu-H100
#SBATCH --job-name=mhubert-extract-features # Name of the process
#SBATCH --gres=gpu:1 # Number of GPUs
#SBATCH --cpus-per-gpu=2 # Number of CPU cores (increased for dataset processing)
#SBATCH --mem-per-gpu=16GB # RAM memory needed (8-16GB)
#SBATCH --time=12:00:00 # 12 hours
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asudupe008@ikasle.ehu.eus
#SBATCH --output=.slurm/apply-kmeans-out.log
#SBATCH --error=.slurm/apply-kmeans-err.log
#SBATCH --chdir=/home/andoni.sudupe/mHubert_finetune

# Activate virtual environment
conda activate hubert

# Run the feature extraction script with command line arguments for HuggingFace dataset
srun python -u scripts/kmeans/apply_kmeans.py