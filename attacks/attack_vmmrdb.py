"""
Natural distribution shift attack on CN immunization using VMMRdb.

Defender immunized on Stanford Cars (studio photos).
Attacker uses VMMRdb (real street photos) — different distribution.

Two evaluations:
  1. RIR on VMMRdb features (vs Stanford Cars baseline)
  2. Linear-probing accuracy: theta_0 vs theta_I on VMMRdb
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from utils.option import load_yaml
from utils.loss import condition_number
from models.model import Model
from models.feature_extractor import PretrainedFeatureExtractor
from dataset import DataModule
from timm.data import create_dataset
from torch.utils.data import DataLoader
import timm


def build_vmmrdb_loader(vmmrdb_path, split, batch_size=64):
    """Build a dataloader for VMMRdb using the same transforms as CN."""
    dataset = create_dataset("image_folder", root=vmmrdb_path, split=split)
    # Use timm default ImageNet transforms (same as Stanford Cars in CN)
    from timm.data import create_transform
    transform = create_transform(
        input_size=224, is_training=(split == 'train'),
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
    )
    dataset.transform = transform
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'),
                        num_workers=4, drop_last=False)
    return loader, len(dataset.reader.class_to_idx) if hasattr(dataset, 'reader') else 10


@torch.no_grad()
def compute_rir_vmmrdb(feature_extractor_ori, feature_extractor_immu,
                       vmmrdb_loader, imagenet_dm, k=100, n=20, device='cuda'):
    """
    Compute RIR where the HARMFUL task (D_H) is VMMRdb instead of Stanford Cars.
    D_P (ImageNet) stays the same.
    """
    import statistics
    
    # Collect VMMRdb samples into a buffer
    vmmrdb_imgs = []
    for X, _ in vmmrdb_loader:
        vmmrdb_imgs.append(X)
        if sum(x.size(0) for x in vmmrdb_imgs) >= k * n:
            break
    vmmrdb_buffer = torch.cat(vmmrdb_imgs, dim=0)
    
    ratio_P_list, ratio_H_list, rir_list = [], [], []
    
    for i in range(n):
        # ImageNet sample (D_P)
        (X1, _), (_, _) = imagenet_dm.get_sampled_data(k, train=True)
        X1 = X1.to(device, dtype=torch.double)
        
        # VMMRdb sample (D_H) -- random k from buffer
        idx = torch.randperm(vmmrdb_buffer.size(0))[:k]
        X2 = vmmrdb_buffer[idx].to(device, dtype=torch.double)
        
        f1_0 = feature_extractor_ori(X1)
        f1_I = feature_extractor_immu(X1)
        f2_0 = feature_extractor_ori(X2)
        f2_I = feature_extractor_immu(X2)
        
        f1_0f = f1_0.view(f1_0.size(0), -1)
        f2_0f = f2_0.view(f2_0.size(0), -1)
        f1_If = f1_I.view(f1_I.size(0), -1)
        f2_If = f2_I.view(f2_I.size(0), -1)
        
        lc1_0 = condition_number(f1_0f.T @ f1_0f)
        lc2_0 = condition_number(f2_0f.T @ f2_0f)
        lc1_I = condition_number(f1_If.T @ f1_If)
        lc2_I = condition_number(f2_If.T @ f2_If)
        
        ratio_P_list.append(torch.exp(lc1_I - lc1_0).item())
        ratio_H_list.append(torch.exp(lc2_I - lc2_0).item())
        rir_list.append(torch.exp(lc2_I - lc2_0 - lc1_I + lc1_0).item())
    
    return {
        'ratio_P': (statistics.mean(ratio_P_list), statistics.stdev(ratio_P_list)),
        'ratio_H': (statistics.mean(ratio_H_list), statistics.stdev(ratio_H_list)),
        'rir': (statistics.mean(rir_list), statistics.stdev(rir_list)),
    }


def linear_probe(feature_extractor, train_loader, val_loader, num_classes,
                 device='cuda', epochs=30, lr=0.01):
    """
    Train a linear classifier on top of frozen feature_extractor.
    Returns final val accuracy.
    """
    feature_extractor.eval()
    for p in feature_extractor.parameters():
        p.requires_grad = False
    
    # Get feature dim
    with torch.no_grad():
        X_sample, _ = next(iter(train_loader))
        X_sample = X_sample.to(device, dtype=torch.double)
        feat = feature_extractor(X_sample)
        feat_dim = feat.view(feat.size(0), -1).size(1)
    
    classifier = nn.Linear(feat_dim, num_classes).to(device, dtype=torch.double)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9)
    
    # Train
    for epoch in range(epochs):
        classifier.train()
        for X, y in train_loader:
            X = X.to(device, dtype=torch.double)
            y = y.to(device)
            with torch.no_grad():
                feat = feature_extractor(X).view(X.size(0), -1)
            logits = classifier(feat)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Evaluate
    classifier.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device, dtype=torch.double)
            y = y.to(device)
            feat = feature_extractor(X).view(X.size(0), -1)
            logits = classifier(feat)
            pred = logits.argmax(dim=1)
            correct += (pred == y.to(device)).sum().item()
            total += y.size(0)
    
    return 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--vmmrdb_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--n', type=int, default=20)
    parser.add_argument('--probe_epochs', type=int, default=30)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_yaml(args.config)
    
    # === Load theta_I ===
    print("Loading immunized model (theta_I)...")
    model = Model(config['model']).to(device, dtype=torch.double)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    feature_extractor_immu = model.feature_extractor
    feature_extractor_immu.eval()
    for p in feature_extractor_immu.parameters():
        p.requires_grad = False
    
    # === Create theta_0 ===
    print("Creating original model (theta_0)...")
    feature_extractor_ori = PretrainedFeatureExtractor(dataset="imagenet").to(device, dtype=torch.double)
    feature_extractor_ori.eval()
    for p in feature_extractor_ori.parameters():
        p.requires_grad = False
    
    # === ImageNet data module (for D_P) ===
    print("Setting up ImageNet data...")
    imagenet_dm = DataModule(
        batch_size=64, dataset_name='imagenet',
        d1_path=config['data']['d1_path'],
        d2_path=config['data']['d2_path'],
        d2_name='cars',
    )
    imagenet_dm.prepare_data()
    imagenet_dm.setup()
    _ = imagenet_dm.train_dataloader()
    
    # === VMMRdb loaders ===
    print("Setting up VMMRdb data...")
    vmmrdb_train, n_cls = build_vmmrdb_loader(args.vmmrdb_path, 'train', batch_size=64)
    vmmrdb_val, _ = build_vmmrdb_loader(args.vmmrdb_path, 'val', batch_size=64)
    print(f"VMMRdb: {n_cls} classes")
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    
    # === Eval 1: RIR on VMMRdb ===
    print("\n=== [1/2] Computing RIR on VMMRdb (natural shift) ===")
    rir_result = compute_rir_vmmrdb(
        feature_extractor_ori, feature_extractor_immu,
        vmmrdb_train, imagenet_dm, k=args.k, n=args.n, device=device
    )
    results['rir'] = rir_result
    print(f"  RIR (VMMRdb) = {rir_result['rir'][0]:.3f} +/- {rir_result['rir'][1]:.3f}")
    print(f"  (Stanford Cars baseline was ~1.3)")
    
    # === Eval 2: Linear probing ===
    print("\n=== [2/2] Linear probing on VMMRdb ===")
    print("  Probing theta_0 (original)...")
    acc_0 = linear_probe(feature_extractor_ori, vmmrdb_train, vmmrdb_val, n_cls,
                         device=device, epochs=args.probe_epochs)
    print(f"  theta_0 accuracy: {acc_0:.2f}%")
    
    print("  Probing theta_I (immunized)...")
    acc_I = linear_probe(feature_extractor_immu, vmmrdb_train, vmmrdb_val, n_cls,
                         device=device, epochs=args.probe_epochs)
    print(f"  theta_I accuracy: {acc_I:.2f}%")
    
    results['probe'] = {'acc_0': acc_0, 'acc_I': acc_I}
    
    # === Summary ===
    print("\n" + "="*60)
    print("ATTACK SUMMARY (VMMRdb natural shift)")
    print("="*60)
    print(f"\nRIR on VMMRdb:        {rir_result['rir'][0]:.3f} +/- {rir_result['rir'][1]:.3f}")
    print(f"  (i)  kappa_H ratio: {rir_result['ratio_H'][0]:.3f}")
    print(f"  (ii) kappa_P ratio: {rir_result['ratio_P'][0]:.3f}")
    print(f"\nLinear probing accuracy on VMMRdb:")
    print(f"  theta_0 (no immunization): {acc_0:.2f}%")
    print(f"  theta_I (immunized):       {acc_I:.2f}%")
    print(f"  Accuracy gap:              {acc_0 - acc_I:.2f}%")
    print(f"\nInterpretation:")
    if acc_0 - acc_I < 2.0:
        print(f"  Immunization has LITTLE effect on VMMRdb (gap < 2%)")
        print(f"  -> Attack SUCCEEDS: immunization fails under distribution shift")
    else:
        print(f"  Immunization still reduces attacker accuracy by {acc_0-acc_I:.1f}%")
    
    # Save
    with open(os.path.join(args.output_dir, "vmmrdb_attack.txt"), 'w') as f:
        f.write(f"RIR_VMMRdb {rir_result['rir'][0]:.4f} {rir_result['rir'][1]:.4f}\n")
        f.write(f"ratio_H {rir_result['ratio_H'][0]:.4f}\n")
        f.write(f"ratio_P {rir_result['ratio_P'][0]:.4f}\n")
        f.write(f"probe_acc_theta0 {acc_0:.4f}\n")
        f.write(f"probe_acc_thetaI {acc_I:.4f}\n")
        f.write(f"acc_gap {acc_0 - acc_I:.4f}\n")


if __name__ == "__main__":
    main()
