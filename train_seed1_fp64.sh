#!/bin/bash
#SBATCH --job-name=cn_s1_fp64
#SBATCH --output=logs/s1_fp64_%j.out
#SBATCH --error=logs/s1_fp64_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=DGXA100
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cn
export WANDB_MODE=offline

cd /hpcstor6/scratch01/t/tong.li003/projects/Condition_Number_Reproducation

echo "===== SEED 1 FP64 Start at $(date) ====="
nvidia-smi

# 备份原 seed 1 (短版结果在 1_short,现在的 1 是 short 的副本)
# 训练会写到 results/.../1/,我们先备份
cp -r results/imagenet_cars/classification/pt_imagenet/1 \
      results/imagenet_cars/classification/pt_imagenet/1_fp32_backup 2>/dev/null

# 训练 (Cars 长度,127 iter/epoch x 3 = 381 iter,但 fp64)
python main.py --config configs/resnet18_cars.yaml --seed 1

echo ""
echo "===== Eval at $(date) ====="
python eval_rir.py \
    --config configs/resnet18_cars.yaml \
    --ckpt results/imagenet_cars/classification/pt_imagenet/1/checkpoints/last.ckpt \
    --output_dir results/imagenet_cars/classification/pt_imagenet/1_fp64/ \
    --skip_test_acc

echo "===== End at $(date) ====="
