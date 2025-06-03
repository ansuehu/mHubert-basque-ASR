#!/usr/bin/env bash
#SBATCH --partition=gpu-H100
#SBATCH --job-name=mhubert-extract-features # Name of the process
#SBATCH --gres=gpu:1 # Number of GPUs
#SBATCH --cpus-per-gpu=2 # Number of CPU cores (increased for dataset processing)
#SBATCH --mem-per-gpu=16GB # RAM memory needed (8-16GB)
#SBATCH --time=12:00:00 # 12 hours
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asudupe008@ikasle.ehu.eus
#SBATCH --output=.slurm/extract-features-out.log
#SBATCH --error=.slurm/extract-features-error.log
#SBATCH --chdir=/home/andoni.sudupe/mHubert_finetune

# Run the feature extraction script with command line arguments for HuggingFace dataset
python -u scripts/kmeans/extract_features.py \
    --model_name "Ansu/mHubert-basque-k1000-L9" \
    --dataset_path "data/composite_eu/preprocessed_data" \
    --batch_size 1 \
    --feat_file "data/composite_eu/hubert_basque_features.npy" \
    --len_file "data/composite_eu/hubert_basque_features.len"

