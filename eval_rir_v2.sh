#!/bin/bash
#SBATCH --job-name=cn_eval_v2
#SBATCH --output=logs/eval_v2_%j.out
#SBATCH --error=logs/eval_v2_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=DGXA100
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cn
export WANDB_MODE=offline

cd /hpcstor6/scratch01/t/tong.li003/projects/Condition_Number_Reproducation

echo "===== Eval v2 (no .eval()) Start at $(date) ====="

python eval_rir_v2.py \
    --config configs/resnet18_cars.yaml \
    --ckpt results/imagenet_cars/classification/pt_imagenet/1/checkpoints/last.ckpt \
    --output_dir results/imagenet_cars/classification/pt_imagenet/1_v2/ \
    --skip_test_acc

echo "===== End at $(date) ====="
