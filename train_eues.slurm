#!/usr/bin/env bash
#SBATCH --partition=gpu-H100
#SBATCH --job-name=mHubert-eues # Name of the process
#SBATCH --gres=gpu:1 # Number of GPUs
#SBATCH --cpus-per-gpu=2 # Number of CPU cores (2 is reasonable)
#SBATCH --mem-per-gpu=16GB # RAM memory needed (8-16GB)
#SBATCH --time=10-00:00 # No time limit
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asudupe008@ikasle.ehu.eus
#SBATCH --output=.slurm/mHubert-eues.out.log
#SBATCH --error=.slurm/mHubert-eues.err.log
#SBATCH --chdir=/home/andoni.sudupe/mHubert_finetune

# Option 1: Continue training from a checkpoint
# srun python -u ./finetune_hubert.py \
#     --model_name "/home/andoni.sudupe/mHubert_finetune/checkpoints/mHubert-basque-ASR-30ep/checkpoint-302700" \
#     --continue_training \
#     --data_dir "/home/andoni.sudupe/mHubert_finetune/data/composite_eu/preprocessed_data" \
#     --output_dir "/home/andoni.sudupe/mHubert_finetune/checkpoints/mHubert-basque-ASR-30ep" \
#     --epochs 60 \
#     --batch_size 64
    # --model_name "/home/andoni.sudupe/mHubert_finetune/checkpoints/mHubert-basque-ASR-30ep" \

# Option 2: Start new training
# srun python -u ./finetune_hubert.py \
#     --model_name "utter-project/mHuBERT-147" \
#     --data_dir "data/composite_eues/preprocessed_data" \
#     --output_dir "checkpoints/mHubert-eues-ASR" \
#     --epochs 50 \
#     --batch_size 32 \
#     --dataloader_num_workers 4

# srun python -u ./finetune_hubert.py \
#     --model_name "/home/andoni.sudupe/mHubert_finetune/checkpoints/mHubert-basque-ASR-30ep/checkpoint-302700" \
#     --data_dir "/home/andoni.sudupe/mHubert_finetune/data/composite_eu/preprocessed_data" \
#     --output_dir "/home/andoni.sudupe/mHubert_finetune/checkpoints/mHubert-basque-ASR-continue" \
#     --epochs 20 \
#     --batch_size 64 \
#     --dataloader_num_workers 1

srun python -u ./finetune_hubert.py \
    --model_name "/home/andoni.sudupe/mHubert_finetune/checkpoints/mHubert-eues-ASR/checkpoint-153000" \
    --data_dir "/home/andoni.sudupe/mHubert_finetune/data/composite_eues/preprocessed_data" \
    --output_dir "/home/andoni.sudupe/mHubert_finetune/checkpoints/mHubert-eues-ASR" \
    --epochs 35 \
    --batch_size 16 \
    --dataloader_num_workers 1

# srun python -u ./finetune_hubert.py \
#     --model_name "/home/andoni.sudupe/mHubert_finetune/checkpoints/mHubert-eues-ASR/checkpoint-153000" \
#     --continue_training \
#     --data_dir "/home/andoni.sudupe/mHubert_finetune/data/composite_eues/preprocessed_data" \
#     --output_dir "/home/andoni.sudupe/mHubert_finetune/checkpoints/mHubert-eues-ASR" \
#     --epochs 50 \
#     --batch_size 16 \
#     --dataloader_num_workers 1