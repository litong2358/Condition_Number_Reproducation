#!/bin/bash
#SBATCH --job-name=cn_train_seed1
#SBATCH --output=logs/train_seed1_%j.out
#SBATCH --error=logs/train_seed1_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=DGXA100
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cn

export WANDB_MODE=offline

cd /hpcstor6/scratch01/t/tong.li003/projects/Condition_Number_Reproducation

echo "===== Start training at $(date) ====="
echo "===== Hostname: $(hostname) ====="
echo "===== WANDB_MODE: $WANDB_MODE ====="
nvidia-smi

echo ""
echo "===== Running main.py ====="
python main.py --config configs/resnet18_cars.yaml --seed 1

echo ""
echo "===== End at $(date) ====="
