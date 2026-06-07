#!/bin/bash
#SBATCH --job-name=extract_imgnet
#SBATCH --output=logs/extract_%j.out
#SBATCH --error=logs/extract_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=Intel6240
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8

IMAGENET_DIR=/hpcstor6/scratch01/t/tong.li003/datasets/imagenet_1k
cd $IMAGENET_DIR

echo "=== Start extraction at $(date) ==="
echo "=== Working dir: $(pwd) ==="

# ===== Step 1: 解压 devkit =====
echo ""
echo "=== [1/4] Extracting devkit (small) ==="
tar -xzf ILSVRC2012_devkit_t12.tar.gz
echo "Devkit extracted. Files:"
ls -la ILSVRC2012_devkit_t12/

# ===== Step 2: 解压 train 第一层 =====
echo ""
echo "=== [2/4] Extracting train (level 1: 1000 tar files) ==="
echo "Start: $(date)"
mkdir -p train
cd train
tar -xf ../ILSVRC2012_img_train.tar
echo "Level 1 done: $(date)"
ls *.tar | wc -l
echo "Number of class tars (should be 1000)"

# ===== Step 3: 解压 train 第二层 =====
echo ""
echo "=== [3/4] Extracting train (level 2: per-class) ==="
echo "Start: $(date)"
# Loop:对每个 nXXX.tar 解压到对应文件夹,然后删除 tar
for f in n*.tar; do
    class_name="${f%.tar}"  # 去掉 .tar 后缀
    mkdir -p "$class_name"
    tar -xf "$f" -C "$class_name"
    rm "$f"  # 删 .tar 省空间
done
echo "Level 2 done: $(date)"

# 验证
n_classes=$(ls -d n*/ 2>/dev/null | wc -l)
echo "Total class dirs: $n_classes (should be 1000)"

# 抽样查一个类有多少图
sample_class=$(ls -d n*/ | head -1 | tr -d '/')
n_imgs=$(ls $sample_class/ | wc -l)
echo "Example: $sample_class has $n_imgs images (typically ~1300)"

cd $IMAGENET_DIR

# ===== Step 4: 解压 + 整理 val =====
echo ""
echo "=== [4/4] Extracting val and organizing by class ==="
echo "Start: $(date)"
mkdir -p val
cd val
tar -xf ../ILSVRC2012_img_val.tar
n_val=$(ls *.JPEG | wc -l)
echo "Val images extracted: $n_val (should be 50000)"

# 用 Python 用 meta.mat 整理 val 文件
cd $IMAGENET_DIR
python << 'PYEOF'
import scipy.io as sio
import os
import shutil

devkit_dir = "ILSVRC2012_devkit_t12"
val_dir = "val"

# 读 meta.mat 得到 (ILSVRC_ID -> WNID) 的映射
meta = sio.loadmat(os.path.join(devkit_dir, "data", "meta.mat"))
synsets = meta['synsets']
id_to_wnid = {}
for i in range(1000):  # 前 1000 个是叶子节点
    ilsvrc_id = int(synsets[i][0][0][0][0])
    wnid = synsets[i][0][1][0]
    id_to_wnid[ilsvrc_id] = wnid

# 读 val 的标签
val_label_file = os.path.join(devkit_dir, "data", "ILSVRC2012_validation_ground_truth.txt")
with open(val_label_file, 'r') as f:
    val_labels = [int(line.strip()) for line in f]

print(f"Loaded {len(val_labels)} val labels")
print(f"Loaded {len(id_to_wnid)} class mappings")

# 移动每张 val 图到对应类文件夹
moved = 0
for i, label_id in enumerate(val_labels):
    img_name = f"ILSVRC2012_val_{i+1:08d}.JPEG"
    wnid = id_to_wnid[label_id]
    
    src = os.path.join(val_dir, img_name)
    dst_dir = os.path.join(val_dir, wnid)
    dst = os.path.join(dst_dir, img_name)
    
    if os.path.exists(src):
        os.makedirs(dst_dir, exist_ok=True)
        shutil.move(src, dst)
        moved += 1

print(f"Moved {moved} val images")

# 验证
class_dirs = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]
print(f"Val class dirs: {len(class_dirs)} (should be 1000)")
PYEOF

echo "Val done: $(date)"

# ===== 最终验证 =====
echo ""
echo "=== FINAL VERIFICATION ==="
echo "Train dir:"
ls -d train/n*/ 2>/dev/null | wc -l
echo "Val dir:"
ls -d val/n*/ 2>/dev/null | wc -l

echo ""
echo "=== End at $(date) ==="
