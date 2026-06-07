"""
Preprocess Stanford Cars from flat structure to by-class folder structure.
"""

import os
import shutil
import scipy.io as sio
from pathlib import Path

# === Paths ===
SRC_DIR = "/hpcstor6/scratch01/t/tong.li003/datasets/cars_torchvision/stanford_cars"
DST_DIR = "/hpcstor6/scratch01/t/tong.li003/datasets/cars"

src = Path(SRC_DIR)
dst = Path(DST_DIR)

# === Load class names ===
meta = sio.loadmat(src / "devkit" / "cars_meta.mat")
class_names = [name[0] for name in meta["class_names"][0]]
print(f"Loaded {len(class_names)} class names")
print(f"Examples: {class_names[:3]}")
print()

# Clean class names for use as folder names (replace / and spaces)
def clean_name(name):
    return name.replace(" ", "_").replace("/", "_")

clean_class_names = [clean_name(n) for n in class_names]

# === Process Train Set ===
print("=== Processing train set ===")
train_annos = sio.loadmat(src / "devkit" / "cars_train_annos.mat")["annotations"][0]
print(f"Train annotations: {len(train_annos)}")

# Create all class folders for train
train_dst = dst / "train"
for cls_name in clean_class_names:
    (train_dst / cls_name).mkdir(parents=True, exist_ok=True)

# Copy images
count = 0
for anno in train_annos:
    class_id = int(anno[4][0][0]) - 1  # 1-indexed → 0-indexed
    fname = anno[5][0]
    
    src_img = src / "cars_train" / fname
    dst_img = train_dst / clean_class_names[class_id] / fname
    
    if src_img.exists():
        shutil.copy2(src_img, dst_img)
        count += 1
    else:
        print(f"WARNING: missing {src_img}")

print(f"Copied {count} train images to {train_dst}")
print()

# === Process Test/Val Set ===
print("=== Processing test set (will be 'val') ===")
test_annos = sio.loadmat(src / "cars_test_annos_withlabels.mat")["annotations"][0]
print(f"Test annotations: {len(test_annos)}")

val_dst = dst / "val"
for cls_name in clean_class_names:
    (val_dst / cls_name).mkdir(parents=True, exist_ok=True)

count = 0
for anno in test_annos:
    class_id = int(anno[4][0][0]) - 1
    fname = anno[5][0]
    
    src_img = src / "cars_test" / fname
    dst_img = val_dst / clean_class_names[class_id] / fname
    
    if src_img.exists():
        shutil.copy2(src_img, dst_img)
        count += 1
    else:
        print(f"WARNING: missing {src_img}")

print(f"Copied {count} val images to {val_dst}")
print()

# === Verify ===
print("=== Verification ===")
train_classes = [d for d in (train_dst).iterdir() if d.is_dir()]
val_classes = [d for d in (val_dst).iterdir() if d.is_dir()]
print(f"Train classes: {len(train_classes)}")
print(f"Val classes: {len(val_classes)}")

train_total = sum(len(list(d.iterdir())) for d in train_classes)
val_total = sum(len(list(d.iterdir())) for d in val_classes)
print(f"Train total images: {train_total}")
print(f"Val total images: {val_total}")

print()
print(f"Expected: 196 classes, 8144 train, 8041 val")
print(f"Got: {len(train_classes)} classes, {train_total} train, {val_total} val")

if len(train_classes) == 196 and train_total == 8144 and val_total == 8041:
    print("\n[OK] ALL CORRECT!")
else:
    print("\n[FAIL] Numbers don't match. Investigate.")
