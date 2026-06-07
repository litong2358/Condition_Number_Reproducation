"""
Preprocess VMMRdb Most_Stolen_Cars into image_folder format (train/val split).

Input:
  vmmrdb_raw/Dataset/Most_Stolen_Cars/<class>/*.jpg

Output:
  vmmrdb/train/<class>/*.jpg  (80%)
  vmmrdb/val/<class>/*.jpg    (20%)
"""

import os
import shutil
import random
from pathlib import Path

random.seed(42)

SRC = Path("/hpcstor6/scratch01/t/tong.li003/datasets/vmmrdb_raw/Dataset/Most_Stolen_Cars")
DST = Path("/hpcstor6/scratch01/t/tong.li003/datasets/vmmrdb")

train_ratio = 0.8

classes = sorted([d.name for d in SRC.iterdir() if d.is_dir()])
print(f"Found {len(classes)} classes: {classes}\n")

train_total, val_total = 0, 0

for cls in classes:
    imgs = sorted([f for f in (SRC / cls).iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    random.shuffle(imgs)
    
    n_train = int(len(imgs) * train_ratio)
    train_imgs = imgs[:n_train]
    val_imgs = imgs[n_train:]
    
    # Create dirs
    (DST / "train" / cls).mkdir(parents=True, exist_ok=True)
    (DST / "val" / cls).mkdir(parents=True, exist_ok=True)
    
    for img in train_imgs:
        shutil.copy2(img, DST / "train" / cls / img.name)
    for img in val_imgs:
        shutil.copy2(img, DST / "val" / cls / img.name)
    
    train_total += len(train_imgs)
    val_total += len(val_imgs)
    print(f"  {cls}: {len(train_imgs)} train, {len(val_imgs)} val")

print(f"\n=== Done ===")
print(f"Total: {len(classes)} classes, {train_total} train, {val_total} val")

# Verify
train_classes = len(list((DST / "train").iterdir()))
val_classes = len(list((DST / "val").iterdir()))
print(f"Verify: train {train_classes} classes, val {val_classes} classes")
