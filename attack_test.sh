#!/bin/bash
#SBATCH --job-name=cn_pgdtest
#SBATCH --output=logs/pgdtest_%j.out
#SBATCH --error=logs/pgdtest_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=DGXA100
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cn
export WANDB_MODE=offline
cd /hpcstor6/scratch01/t/tong.li003/projects/Condition_Number_Reproducation

echo "===== PGD TEST (small) at $(date) ====="
# 小参数:n=3, steps=5, restarts=1, eot=2 — 快速验证不报错
python attack_rir.py \
    --config configs/resnet18_cars.yaml \
    --ckpt results/imagenet_cars/classification/pt_imagenet/1/checkpoints/last.ckpt \
    --output_dir results/attack_pgdtest/ \
    --k 50 --n 3 \
    --eps 0.0314 --alpha 0.0078 \
    --pgd_steps 5 --pgd_restarts 1 --eot 2 \
    --attacks clean,random_noise,pgd
echo "===== Test end at $(date) ====="
