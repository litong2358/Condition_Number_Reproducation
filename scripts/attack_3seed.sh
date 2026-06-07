#!/bin/bash
#SBATCH --job-name=cn_atk3
#SBATCH --output=logs/atk3_%j.out
#SBATCH --error=logs/atk3_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=DGXA100
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cn
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

cd /hpcstor6/scratch01/t/tong.li003/projects/Condition_Number_Reproducation

for SEED in 1 2 3; do
    echo ""
    echo "############################################################"
    echo "##### SEED $SEED at $(date)"
    echo "############################################################"
    
    python -u attacks/attack_rir.py \
        --config configs/resnet18_cars.yaml \
        --ckpt results/imagenet_cars/classification/pt_imagenet/${SEED}/checkpoints/last.ckpt \
        --output_dir results/attack_3seed_seed${SEED}/ \
        --k 100 --n 20 \
        --eps 0.0314 --alpha 0.0078 \
        --pgd_steps 40 --pgd_restarts 3 --eot 4 \
        --attacks random_noise,pgd
done

echo ""
echo "===== All done at $(date) ====="
