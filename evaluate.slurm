#!/usr/bin/env bash
#SBATCH --partition=gpu-H100
#SBATCH --job-name=evaluate-mhubert # Name of the process
#SBATCH --gres=gpu:1 # Number of GPUs
#SBATCH --cpus-per-gpu=2 # Number of CPU cores (2 is reasonable)
#SBATCH --mem-per-gpu=16GB # RAM memory needed (8-16GB)
#SBATCH --time=10:00:00 # 10 hours
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asudupe008@ikasle.ehu.eus
#SBATCH --output=.slurm/evaluate-out-%N_%j.log
#SBATCH --error=.slurm/evaluate-error-%N_%j.log
#SBATCH --chdir=/home/andoni.sudupe/mHubert_finetune

# Activate virtual environment
source ~/envs/myenv/bin/activate

# Run the evaluation script with command line arguments
srun python -u ./evaluate_hubert.py \
    --model_path "/home/andoni.sudupe/mHubert_finetune/checkpoints/mHubert-basque-ASR-30ep/checkpoint-302700" \
    --data_dir "/home/andoni.sudupe/mHubert_finetune/data/preprocessed_data" \
    --datasets test_cv test_parl test_oslr \