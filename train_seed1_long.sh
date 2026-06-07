#!/bin/bash
#SBATCH --job-name=cn_s1_long
#SBATCH --output=logs/s1_long_%j.out
#SBATCH --error=logs/s1_long_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=DGXA100
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cn
export WANDB_MODE=offline

cd /hpcstor6/scratch01/t/tong.li003/projects/Condition_Number_Reproducation

echo "===== SEED 1 LONG Start at $(date) ====="
nvidia-smi

python main.py --config configs/resnet18_cars.yaml --seed 1

echo ""
echo "===== Eval ====="
python eval_rir.py \
    --config configs/resnet18_cars.yaml \
    --ckpt results/imagenet_cars/classification/pt_imagenet/1/checkpoints/last.ckpt \
    --output_dir results/imagenet_cars/classification/pt_imagenet/1/ \
    --skip_test_acc

echo "===== End at $(date) ====="
