#!/bin/bash
#SBATCH --job-name=cn_seed2
#SBATCH --output=logs/seed2_%j.out
#SBATCH --error=logs/seed2_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=DGXA100
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cn

export WANDB_MODE=offline

cd /hpcstor6/scratch01/t/tong.li003/projects/Condition_Number_Reproducation

echo "===== SEED 2: Start training at $(date) ====="
python main.py --config configs/resnet18_cars.yaml --seed 2

echo ""
echo "===== SEED 2: Start eval at $(date) ====="
python eval_rir.py \
    --config configs/resnet18_cars.yaml \
    --ckpt results/imagenet_cars/classification/pt_imagenet/2/checkpoints/last.ckpt \
    --output_dir results/imagenet_cars/classification/pt_imagenet/2/ \
    --skip_test_acc

echo ""
echo "===== End at $(date) ====="
