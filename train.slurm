#!/usr/bin/env bash
#SBATCH --partition=gpu-H100
#SBATCH --job-name=mHubert-Finetune # Name of the process
#SBATCH --gres=gpu:1 # Number of GPUs
#SBATCH --cpus-per-gpu=2 # Number of CPU cores (2 is reasonable)
#SBATCH --mem-per-gpu=16GB # RAM memory needed (8-16GB)
#SBATCH --time=0 # No time limit
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asudupe008@ikasle.ehu.eus
#SBATCH --output=.slurm/stdout-%N_%j.log
#SBATCH --error=.slurm/stderr-%N_%j.log
#SBATCH --chdir=/home/andoni.sudupe/mHubert_finetune

# Activate virtual environment
source ~/envs/myenv/bin/activate

# Choose one of the following commands based on your needs:

# Option 1: Continue training from a checkpoint
srun python -u ./finetune_hubert.py \
    --model_name "/home/andoni.sudupe/mHubert_finetune/checkpoints/mHubert-basque-ASR-30ep/checkpoint-146000" \
    --continue_training \
    --data_dir "data/preprocessed_data" \
    --output_dir "checkpoints/mHubert-basque-ASR-30ep" \
    --epochs 30 \
    --batch_size 8

# Option 2: Start new training (commented out)
# srun python -u ./finetune_hubert.py \
#     --model_name "utter-project/mHuBERT-147" \
#     --data_dir "data/preprocessed_data" \
#     --output_dir "checkpoints/mHubert-basque-ASR-new" \
#     --epochs 30 \
#     --batch_size 8