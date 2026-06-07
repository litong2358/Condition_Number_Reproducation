#!/bin/bash
#SBATCH --job-name=cn_vmmrdb23
#SBATCH --output=logs/vmmrdb23_%j.out
#SBATCH --error=logs/vmmrdb23_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=DGXA100
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cn
export WANDB_MODE=offline

cd /hpcstor6/scratch01/t/tong.li003/projects/Condition_Number_Reproducation

for SEED in 2 3; do
    echo ""
    echo "############################################################"
    echo "##### VMMRdb attack on SEED $SEED at $(date)"
    echo "############################################################"
    
    python attack_vmmrdb.py \
        --config configs/resnet18_cars.yaml \
        --ckpt results/imagenet_cars/classification/pt_imagenet/${SEED}/checkpoints/last.ckpt \
        --vmmrdb_path /hpcstor6/scratch01/t/tong.li003/datasets/vmmrdb \
        --output_dir results/attack_vmmrdb_seed${SEED}/ \
        --k 100 --n 20 --probe_epochs 30
done

echo ""
echo "===== All done at $(date) ====="
