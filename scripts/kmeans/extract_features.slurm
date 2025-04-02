#!/usr/bin/env bash
#SBATCH --partition=gpu-H100
#SBATCH --job-name=mhubert-extract-features # Name of the process
#SBATCH --gres=gpu:1 # Number of GPUs
#SBATCH --cpus-per-gpu=2 # Number of CPU cores (increased for dataset processing)
#SBATCH --mem-per-gpu=16GB # RAM memory needed (8-16GB)
#SBATCH --time=12:00:00 # 12 hours
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asudupe008@ikasle.ehu.eus
#SBATCH --output=.slurm/extract-features-out-%N_%j.log
#SBATCH --error=.slurm/extract-features-error-%N_%j.log
#SBATCH --chdir=/home/andoni.sudupe/mHubert_finetune

# Activate virtual environment
source ~/envs/myenv/bin/activate

# Run the feature extraction script with command line arguments for HuggingFace dataset
srun python -u ./extract_features_huggingface.py \
    --model_name "/home/andoni.sudupe/mHubert_finetune/checkpoints/mHubert-basque-ASR-30ep/checkpoint-302700" \
    --dataset_path "data/raw_data" \
    --output_file "hubert_basque_features.npy" \
    --batch_size 16 \
    --local_files_only