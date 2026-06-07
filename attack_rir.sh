#!/bin/bash
#SBATCH --job-name=cn_pgd
#SBATCH --output=logs/pgd_%j.out
#SBATCH --error=logs/pgd_%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=DGXA100
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cn
export WANDB_MODE=offline

cd /hpcstor6/scratch01/t/tong.li003/projects/Condition_Number_Reproducation

echo "===== PGD attack at $(date) ====="
nvidia-smi | head -15
echo ""

python attack_rir.py \
    --config configs/resnet18_cars.yaml \
    --ckpt results/imagenet_cars/classification/pt_imagenet/1/checkpoints/last.ckpt \
    --output_dir results/attack_pgd_seed1/ \
    --k 100 --n 20 \
    --eps 0.0314 --alpha 0.0078 \
    --pgd_steps 40 --pgd_restarts 3 --eot 4 \
    --attacks clean,random_noise,pgd

echo ""
echo "===== End at $(date) ====="
