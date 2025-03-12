#!/usr/bin/env bash
#SBATCH --partition=gpu-H100
#SBATCH --job-name=mHubert-Finetune-CTC # Name of the process
#SBATCH --gres=gpu:1 # Number of GPUs
#SBATCH --cpus-per-gpu=2 # Number of CPU cores (2 is reasonable)
#SBATCH --mem-per-gpu=16GB # RAM memory needed (8-16GB)
#SBATCH --time=1:00:00 # 1 hour
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asudupe008@ikasle.ehu.eus
#SBATCH --output=.slurm/stdout-%N_%j.log
#SBATCH --error=.slurm/stderr-%N_%j.log

source ~/envs/myenv/bin/activate

srun python ./scripts/finetune_hubert.py
