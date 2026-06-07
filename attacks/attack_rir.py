"""
Adversarial attacks on CN immunization (Stanford Cars input).
Mirrors IMMA Ex2: clean / random_noise / PGD-40.

Attack point: Cars input X_H
Objective: make immunized features' condition number match the
           original model's, i.e., erase the immunization.
PGD maximizes:  -|log_kappa(theta_I) - log_kappa(theta_0)|
"""

import os, sys, argparse, statistics
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from utils.option import load_yaml
from utils.loss import condition_number
from models.model import Model
from models.feature_extractor import PretrainedFeatureExtractor
from dataset import DataModule


def log_kappa(feat):
    f = feat.view(feat.size(0), -1)
    return condition_number(f.T @ f)   # returns log condition number


@torch.no_grad()
def eval_rir(X1, X2, fe_ori, fe_immu):
    """Compute (ratio_P, ratio_H, RIR) on given X1(ImageNet), X2(Cars)."""
    f1_0 = fe_ori(X1);  f1_I = fe_immu(X1)
    f2_0 = fe_ori(X2);  f2_I = fe_immu(X2)
    lc1_0 = log_kappa(f1_0); lc1_I = log_kappa(f1_I)
    lc2_0 = log_kappa(f2_0); lc2_I = log_kappa(f2_I)
    ratio_P = torch.exp(lc1_I - lc1_0).item()
    ratio_H = torch.exp(lc2_I - lc2_0).item()
    rir = torch.exp(lc2_I - lc2_0 - lc1_I + lc1_0).item()
    return ratio_P, ratio_H, rir


def attack_random_noise(X2, eps):
    return (X2 + torch.empty_like(X2).uniform_(-eps, eps)).detach()


def attack_pgd(X2, fe_ori, fe_immu, eps, alpha, n_steps, n_restarts, eot_n):
    """PGD: maximize -|log_kappa_immu - log_kappa_ori| over delta."""
    X2 = X2.detach()
    best_X = X2.clone()
    best_loss = -float('inf')

    for _ in range(n_restarts):
        delta = torch.empty_like(X2).uniform_(-eps, eps)
        delta.requires_grad_(True)

        last_loss = -float('inf')
        for step in range(n_steps):
            grad_accum = torch.zeros_like(X2)
            loss_accum = 0.0
            valid = 0
            for _ in range(eot_n):
                X_adv = X2 + delta
                lc_immu = log_kappa(fe_immu(X_adv))
                lc_ori  = log_kappa(fe_ori(X_adv))
                # ATTACK: minimize kappa_H ratio (immu/ori) -> RIR drops -> immunization fails
                # PGD maximizes loss, so loss = -(lc_immu - lc_ori) minimizes (lc_immu - lc_ori)
                loss = -(lc_immu - lc_ori)
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                g = torch.autograd.grad(loss, delta, retain_graph=False)[0]
                if torch.isnan(g).any():
                    continue
                grad_accum += g
                loss_accum += loss.item()
                valid += 1
            if valid == 0:
                break
            grad_accum /= valid
            last_loss = loss_accum / valid
            delta = (delta.detach() + alpha * grad_accum.sign())
            delta = torch.clamp(delta, -eps, eps).requires_grad_(True)

        if last_loss > best_loss:
            best_loss = last_loss
            best_X = (X2 + delta.detach()).clone()

    return best_X.detach()


def run_attack(name, data_module, fe_ori, fe_immu, args, device):
    rp_l, rh_l, rir_l = [], [], []
    for i in range(args.n):
        (X1, _), (X2, _) = data_module.get_sampled_data(args.k, train=True)
        X1 = X1.to(device, dtype=torch.double)
        X2 = X2.to(device, dtype=torch.double)

        if name == 'clean':
            X2a = X2
        elif name == 'random_noise':
            X2a = attack_random_noise(X2, args.eps)
        elif name == 'pgd':
            X2a = attack_pgd(X2, fe_ori, fe_immu, args.eps, args.alpha,
                             args.pgd_steps, args.pgd_restarts, args.eot)
        else:
            raise ValueError(name)

        rp, rh, rir = eval_rir(X1, X2a, fe_ori, fe_immu)
        rp_l.append(rp); rh_l.append(rh); rir_l.append(rir)
        if (i + 1) % 5 == 0:
            print(f"  [{name}] {i+1}/{args.n}: RIR={rir:.3f}")

    return {
        'ratio_P': (statistics.mean(rp_l), statistics.stdev(rp_l)),
        'ratio_H': (statistics.mean(rh_l), statistics.stdev(rh_l)),
        'rir': (statistics.mean(rir_l), statistics.stdev(rir_l)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--k', type=int, default=100)
    p.add_argument('--n', type=int, default=20)
    p.add_argument('--eps', type=float, default=8/255)
    p.add_argument('--alpha', type=float, default=2/255)
    p.add_argument('--pgd_steps', type=int, default=40)
    p.add_argument('--pgd_restarts', type=int, default=3)
    p.add_argument('--eot', type=int, default=4)
    p.add_argument('--attacks', default='clean,random_noise,pgd')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_yaml(args.config)

    print("Loading theta_I...")
    model = Model(config['model']).to(device, dtype=torch.double)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    fe_immu = model.feature_extractor.eval()
    for q in fe_immu.parameters():
        q.requires_grad = False

    print("Creating theta_0...")
    fe_ori = PretrainedFeatureExtractor(dataset="imagenet").to(device, dtype=torch.double).eval()
    for q in fe_ori.parameters():
        q.requires_grad = False

    print("Setting up data...")
    dm = DataModule(batch_size=config['data'].get('batch_size', 64),
                    dataset_name=config['data'].get('dataset_name'),
                    d1_path=config['data'].get('d1_path'),
                    d2_path=config['data'].get('d2_path'),
                    d2_name=config['data'].get('d2_name'))
    dm.prepare_data(); dm.setup(); _ = dm.train_dataloader()

    os.makedirs(args.output_dir, exist_ok=True)
    all_res = {}
    for name in args.attacks.split(','):
        print(f"\n{'='*50}\nAttack: {name}\n{'='*50}")
        all_res[name] = run_attack(name, dm, fe_ori, fe_immu, args, device)

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"{'Attack':<15}{'(i) kH':<18}{'(ii) kP':<18}{'RIR':<18}")
    print(f"{'Paper':<15}{'2.386':<18}{'0.699':<18}{'3.467':<18}")
    for name in args.attacks.split(','):
        r = all_res[name]
        print(f"{name:<15}"
              f"{r['ratio_H'][0]:.3f}+/-{r['ratio_H'][1]:.3f}    "
              f"{r['ratio_P'][0]:.3f}+/-{r['ratio_P'][1]:.3f}    "
              f"{r['rir'][0]:.3f}+/-{r['rir'][1]:.3f}")

    with open(os.path.join(args.output_dir, "attack_summary.txt"), 'w') as f:
        for name in args.attacks.split(','):
            r = all_res[name]
            f.write(f"{name} rir {r['rir'][0]:.4f} {r['rir'][1]:.4f} "
                    f"kH {r['ratio_H'][0]:.4f} kP {r['ratio_P'][0]:.4f}\n")
    print(f"\nSaved to {args.output_dir}/attack_summary.txt")


if __name__ == "__main__":
    main()
