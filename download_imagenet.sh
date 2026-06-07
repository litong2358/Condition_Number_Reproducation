#!/bin/bash
#SBATCH --job-name=dl_imagenet
#SBATCH --output=logs/dl_imagenet_%j.out
#SBATCH --error=logs/dl_imagenet_%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=Intel6240
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

cd /hpcstor6/scratch01/t/tong.li003/datasets/imagenet_1k

echo "=== Start download at $(date) ==="
echo "=== Working dir: $(pwd) ==="

# 1. validation set (6.3GB,大约 10-30 分钟)
echo ""
echo "=== [1/2] Downloading validation set (6.3 GB) ==="
wget -c "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"
echo "Val download exit code: $?"

# 2. training set (138GB,大约 1-6 小时)
echo ""
echo "=== [2/2] Downloading training set (138 GB) ==="
wget -c "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar"
echo "Train download exit code: $?"

echo ""
echo "=== Download finished at $(date) ==="
ls -lh

# 算 MD5 校验,验证下载完整性
echo ""
echo "=== MD5 Verification ==="
md5sum ILSVRC2012_img_val.tar
md5sum ILSVRC2012_img_train.tar
