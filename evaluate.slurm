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



# Run the evaluation script with command line arguments
# srun python -u ./evaluate_hubert.py \
#     --model_path "/home/andoni.sudupe/mHubert_finetune/checkpoints/mHubert-basque-ASR-continue/checkpoint-121080" \
#     --data_dir "/data/aholab/asr/corpus/Faktoria_Gaitu/asr/manifests/2025-01/2025-01_clean_norm.json" \
#     --vocab_path "basque_vocab.json" \
#     --manifest True \
#     --datasets train \
#     --num_samples 5

srun python -u ./evaluate_hubert.py \
    --model_path "/home/andoni.sudupe/mHubert_finetune/checkpoints/mHubert-eues-ASR/checkpoint-151000" \
    --data_dir "/home/andoni.sudupe/mHubert_finetune/data/composite_eues/preprocessed_data" \
    --vocab_path "vocab.json" \
    --datasets test_cv_es test_cv_eu test_mls_es test_oslr_es test_oslr_eu test_parl_es test_parl_eu test_vp_es \
    --num_samples 5