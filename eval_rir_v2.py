"""
独立评估脚本:加载训练好的 checkpoint,正确算 RIR。

修复 main.py 的 bug — 不再用 model.feature_extractor 两次,
而是用一个原始未训练的 PretrainedFeatureExtractor 作为 θ_0。
"""

import os
import sys
import copy
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.option import load_yaml
from utils.log import log_and_save_avg_condition_numbers
from utils.evaluate_imgnet import evaluate_model
from models.model import Model
from models.feature_extractor import PretrainedFeatureExtractor
from dataset import DataModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True, help="Path to .ckpt file")
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--n', type=int, default=20)
    parser.add_argument('--skip_test_acc', action='store_true',
                       help="Skip recomputing test accuracy (use existing)")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === 1. Load config ===
    config = load_yaml(args.config)
    model_params = config['model']
    data_params = config['data']

    # === 2. Load checkpoint into model ===
    print(f"Loading checkpoint from: {args.ckpt}")
    model = Model(model_params).to(device, dtype=torch.double)
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    # model.eval()  # disabled to match main.py
    print("Checkpoint loaded.")

    # === 3. 关键修复:创建一个独立的、未训练的 θ_0 ===
    print("Creating original (pre-trained, NOT immunized) feature extractor as θ_0...")
    original_feature_extractor = PretrainedFeatureExtractor(dataset="imagenet").to(device, dtype=torch.double)
    # original_feature_extractor.eval()  # disabled
    for p in original_feature_extractor.parameters():
        p.requires_grad = False
    print("θ_0 initialized.")

    # === 4. 初始化 data module ===
    print("Initializing data module...")
    data_module = DataModule(
        batch_size=data_params.get("batch_size", 64),
        dataset_name=data_params.get("dataset_name", None),
        d1_path=data_params.get("d1_path", None),
        d2_path=data_params.get("d2_path", None),
        d2_name=data_params.get("d2_name", None),
    )
    data_module.prepare_data()
    data_module.setup()
    _ = data_module.train_dataloader()  # 初始化 train_loader
    print("Data module ready.")

    # === 5. 算 RIR(正确的 θ_0 vs θ_I)===
    print(f"\n=== Computing RIR with k={args.k}, n={args.n} ===")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 删掉旧的(错误的)ratios.txt
    old_ratios = os.path.join(args.output_dir, "ratios.txt")
    if os.path.exists(old_ratios):
        print(f"Removing old (buggy) ratios.txt")
        os.remove(old_ratios)
    old_std = os.path.join(args.output_dir, "ratios.txt_std")
    if os.path.exists(old_std):
        os.remove(old_std)
    
    with torch.no_grad():
        log_and_save_avg_condition_numbers(
            data_module,
            original_feature_extractor,   # ⭐ θ_0 (correct!)
            model.feature_extractor,      # θ_I
            k=args.k, n=args.n,
            output_dir=args.output_dir,
            device=device,
            train=True,
        )
    
    # === 6. 显示结果 ===
    print("\n=== RESULTS ===")
    with open(os.path.join(args.output_dir, "ratios.txt"), 'r') as f:
        line = f.read().strip()
    parts = line.split()
    r1_mean, r2_mean, rir_mean = float(parts[0]), float(parts[1]), float(parts[2])
    
    with open(os.path.join(args.output_dir, "ratios.txt_std"), 'r') as f:
        line = f.read().strip()
    parts = line.split()
    r1_std, r2_std, rir_std = float(parts[0]), float(parts[1]), float(parts[2])
    
    print(f"")
    print(f"  Eq.(17)(i):  κ(H_H(θ_I))/κ(H_H(θ_0)) = {r2_mean:.3f} ± {r2_std:.3f}")
    print(f"     Paper:                            = 2.386 ± 0.442")
    print(f"")
    print(f"  Eq.(17)(ii): κ(H_P(θ_I))/κ(H_P(θ_0)) = {r1_mean:.3f} ± {r1_std:.3f}")
    print(f"     Paper:                            = 0.699 ± 0.062")
    print(f"")
    print(f"  RIR (i)/(ii):                        = {rir_mean:.3f} ± {rir_std:.3f}")
    print(f"     Paper:                            = 3.467 ± 0.358")
    
    # === 7. (可选)重新算 Test Acc ===
    if not args.skip_test_acc:
        print("\n=== Recomputing Test Accuracy (optional, slow) ===")
        test_acc = evaluate_model(model, data_params['d1_path'])
        print(f"Test Accuracy: {test_acc:.2f}%")
        with open(os.path.join(args.output_dir, "test_acc.txt"), 'w') as f:
            f.write(f"Test Accuracy: {test_acc:.2f}%\n")
    

if __name__ == "__main__":
    main()
