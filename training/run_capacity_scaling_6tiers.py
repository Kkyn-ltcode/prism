"""Experiment E: Extended parameter scaling with 6 data points."""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import sys
from pathlib import Path
import json
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from torch_geometric.loader import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dataset_qm9 import get_qm9_splits
from models.eq_gnn import GaussianRBF, compute_edge_mean_angles, N_RBF
from models.classical_baselines import PaiNNModel
from training.run_table1_qm9_benchmark import resolve_data_dir

# Scale Configurations (6 data points)
# `d_h` = head_hidden dimension inside the edge MLP
# `hidden_features` = node embedding dimension
# `painn_h` = PaiNN hidden channels (separate param budget)

SCALES = {
    "XS":  {"d_h": 4,   "head_hidden": 4,   "hidden_features": 16,  "painn_h": 8},
    "S":   {"d_h": 8,   "head_hidden": 8,   "hidden_features": 24,  "painn_h": 16},
    "M":   {"d_h": 16,  "head_hidden": 16,  "hidden_features": 32,  "painn_h": 32},
    "L":   {"d_h": 32,  "head_hidden": 32,  "hidden_features": 48,  "painn_h": 64},
    "XL":  {"d_h": 64,  "head_hidden": 64,  "hidden_features": 64,  "painn_h": 128},
    "XXL": {"d_h": 128, "head_hidden": 128, "hidden_features": 128, "painn_h": 64},
}

TARGET_UNITS = {"homo": "eV", "lumo": "eV", "gap": "eV", "mu": "Debye", "alpha": "Bohr³", "u0": "eV"}
TARGET_DISPLAY = {"homo": "HOMO", "lumo": "LUMO", "gap": "Gap", "mu": "Dipole", "alpha": "Polariz.", "u0": "U₀"}

SAVE_PATH = PROJECT_ROOT / "results" / "experiment_e_param_scaling.json"


# Scalable Model Definitions

class ScalablePRISMHead(nn.Module):
    def __init__(self, in_features, head_hidden, n_rbf=N_RBF, omega_0=30.0):
        super().__init__()
        self.feature_proj = nn.Linear(in_features, 2)
        self.omega_0 = omega_0
        input_dim = 4 + n_rbf + 1
        n_outputs = 8
        self.linear1 = nn.Linear(input_dim, head_hidden)
        with torch.no_grad():
            self.linear1.weight.uniform_(-1 / input_dim, 1 / input_dim)
            if self.linear1.bias is not None:
                self.linear1.bias.uniform_(-1 / input_dim, 1 / input_dim)
        self.linear2 = nn.Linear(head_hidden, n_outputs, bias=False)
        bound2 = math.sqrt(6.0 / head_hidden) / self.omega_0
        with torch.no_grad():
            self.linear2.weight.uniform_(-bound2, bound2)

    def forward(self, h_i, h_j, rbf_features, angle):
        h_i_proj = self.feature_proj(h_i)
        h_j_proj = self.feature_proj(h_j)
        x = torch.cat([h_i_proj, h_j_proj, rbf_features, angle], dim=-1)
        h = torch.sin(self.omega_0 * self.linear1(x))
        return torch.tanh(self.linear2(h))


class ScalableClassicalHead(nn.Module):
    def __init__(self, in_features, head_hidden, n_rbf=N_RBF):
        super().__init__()
        self.feature_proj = nn.Linear(in_features, 2)
        input_dim = 4 + n_rbf + 1
        n_outputs = 8
        self.linear1 = nn.Linear(input_dim, head_hidden)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(head_hidden, n_outputs, bias=False)

    def forward(self, h_i, h_j, rbf_features, angle):
        h_i_proj = self.feature_proj(h_i)
        h_j_proj = self.feature_proj(h_j)
        x = torch.cat([h_i_proj, h_j_proj, rbf_features, angle], dim=-1)
        h = self.act(self.linear1(x))
        return torch.tanh(self.linear2(h))


class ScalableLayer(nn.Module):
    def __init__(self, in_features, head_hidden, head_type="prism", n_rbf=N_RBF):
        super().__init__()
        self.rbf = GaussianRBF(n_rbf=n_rbf)
        if head_type == "prism":
            self.head_electronic = ScalablePRISMHead(in_features, head_hidden, n_rbf)
            self.head_topological = ScalablePRISMHead(in_features, head_hidden, n_rbf)
        else:
            self.head_electronic = ScalableClassicalHead(in_features, head_hidden, n_rbf)
            self.head_topological = ScalableClassicalHead(in_features, head_hidden, n_rbf)
        self.coord_gate = nn.Linear(2, 1)
        msg_dim = 7 * 2
        self.node_mlp = nn.Sequential(
            nn.Linear(in_features + msg_dim, in_features),
            nn.SiLU(),
            nn.Linear(in_features, in_features)
        )

    def forward(self, h, x, edge_index):
        row, col = edge_index[0], edge_index[1]
        rel_x = x[row] - x[col]
        d_ij = torch.sum(rel_x ** 2, dim=-1, keepdim=True)
        rbf_features = self.rbf(d_ij)
        angles = compute_edge_mean_angles(x, edge_index)
        out_e = self.head_electronic(h[row], h[col], rbf_features, angles)
        out_t = self.head_topological(h[row], h[col], rbf_features, angles)
        msg_e = torch.cat([out_e[:, :3], out_e[:, 4:]], dim=-1)
        msg_t = torch.cat([out_t[:, :3], out_t[:, 4:]], dim=-1)
        m_ij = torch.cat([msg_e, msg_t], dim=-1)
        phi_x_raw = torch.cat([out_e[:, 3:4], out_t[:, 3:4]], dim=-1)
        phi_x = torch.tanh(self.coord_gate(phi_x_raw))
        coord_msg = rel_x * phi_x
        x_update = torch.zeros_like(x)
        x_update.index_add_(0, row, coord_msg)
        degrees = torch.zeros((x.size(0), 1), device=x.device)
        degrees.index_add_(0, row, torch.ones((row.size(0), 1), device=x.device))
        x_new = x + x_update / degrees.clamp(min=1.0)
        m_i = torch.zeros((h.size(0), m_ij.size(1)), device=h.device)
        m_i.index_add_(0, row, m_ij)
        h_new = h + self.node_mlp(torch.cat([h, m_i], dim=-1))
        return h_new, x_new


class ScalableModel(nn.Module):
    def __init__(self, head_type="prism", in_features=11, hidden_features=16,
                 head_hidden=4, num_layers=2, out_dim=1, n_rbf=N_RBF):
        super().__init__()
        self.embedding = nn.Linear(in_features, hidden_features)
        self.layers = nn.ModuleList([
            ScalableLayer(hidden_features, head_hidden, head_type, n_rbf)
            for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, out_dim)
        )

    def forward(self, h, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        h = self.embedding(h)
        for layer in self.layers:
            h, x = layer(h, x, edge_index)
        num_graphs = int(batch.max().item()) + 1
        out_pool = torch.zeros((num_graphs, h.size(1)), device=h.device)
        out_pool.index_add_(0, batch, h)
        counts = torch.zeros((num_graphs, 1), device=h.device)
        counts.index_add_(0, batch, torch.ones((h.size(0), 1), device=h.device))
        out_pool = out_pool / counts.clamp(min=1.0)
        return self.head(out_pool)


# Training

def create_model(model_type, cfg):
    if model_type == "prism":
        return ScalableModel(head_type="prism",
                             hidden_features=cfg["hidden_features"],
                             head_hidden=cfg["head_hidden"])
    elif model_type == "twin":
        return ScalableModel(head_type="classical",
                             hidden_features=cfg["hidden_features"],
                             head_hidden=cfg["head_hidden"])
    elif model_type == "painn":
        return PaiNNModel(in_features=11, hidden_channels=cfg["painn_h"])


def train_eval(model_type, cfg, target, seed,
               num_train=1000, num_val=250, num_test=500,
               epochs=15, batch_size=16, lr=1e-3,
               device="cpu", data_dir=None):
    """Train one model with target standardization, return (test_mae, param_count, training_time_sec)."""
    resolved_data_dir = resolve_data_dir(data_dir)
    dev = torch.device(device)

    train_set, val_set, test_set, (mean, std) = get_qm9_splits(
        root=resolved_data_dir, target=target,
        num_train=num_train, num_val=num_val, num_test=num_test, seed=seed,
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    torch.manual_seed(seed)
    model = create_model(model_type, cfg).to(dev)
    total_params = count_parameters(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.MSELoss()

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(dev)
            optimizer.zero_grad()
            y_norm = (batch.y.view(-1) - mean) / std
            pred_norm = model(batch.x.float(), batch.pos.float(), batch.edge_index, batch.batch).view(-1)
            loss = criterion(pred_norm, y_norm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        scheduler.step()
    train_time = time.time() - t0

    model.eval()
    total_mae = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(dev)
            pred_norm = model(batch.x.float(), batch.pos.float(), batch.edge_index, batch.batch).view(-1)
            pred = pred_norm * std + mean
            total_mae += torch.abs(pred - batch.y.view(-1)).sum().item()
            total_samples += batch.num_graphs

    test_mae = total_mae / total_samples
    return test_mae, total_params, train_time


def compute_stats(maes_a, maes_b):
    a, b = np.array(maes_a), np.array(maes_b)
    diff = a - b
    try:
        _, w_p = stats.wilcoxon(a, b)
    except ValueError:
        w_p = 1.0
    _, t_p = stats.ttest_rel(a, b)
    n = len(a)
    d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0.0
    return {
        "mean_diff": float(np.mean(diff)),
        "std_diff": float(np.std(diff, ddof=1)),
        "cohens_d": float(d),
        "wilcoxon_p": float(w_p),
        "ttest_p": float(t_p),
        "n_seeds": n,
        "significant_005": bool(t_p < 0.05),
        "significant_001": bool(t_p < 0.01),
    }


# Main

def main():
    parser = argparse.ArgumentParser(description="Experiment E: Extended Parameter Scaling (6 points, 100k+)")
    parser.add_argument("--targets", nargs="+", default=["homo", "lumo", "gap"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--scales", nargs="+", default=list(SCALES.keys()),
                        help="Subset of scales to run: XS S M L XL XXL")
    parser.add_argument("--models", nargs="+", default=["prism", "twin", "painn"],
                        help="Models to evaluate")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--num-train", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use: cpu, cuda, etc.")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Optional explicit path to QM9 dataset")
    parser.add_argument("--force", action="store_true",
                        help="Force recomputation even if scale/model exists in checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    args = parser.parse_args()

    selected_scales = {k: SCALES[k] for k in args.scales if k in SCALES}

    # Dry run: just print param counts
    if args.dry_run:
        print("=" * 90)
        print("  DRY RUN — Parameter Counts per Scale")
        print("=" * 90)
        for name, cfg in selected_scales.items():
            prism = create_model("prism", cfg)
            twin = create_model("twin", cfg)
            painn = create_model("painn", cfg)
            print(f"  {name:4s}: PRISM={count_parameters(prism):>9,} | "
                  f"Twin={count_parameters(twin):>9,} | "
                  f"PaiNN={count_parameters(painn):>9,}  "
                  f"(d_h={cfg['d_h']}, hidden={cfg['hidden_features']})")
        print(f"\n  Targets: {args.targets}")
        print(f"  Seeds:   {args.seeds}")
        print(f"  Epochs:  {args.epochs}")
        print(f"  Device:  {args.device}")
        n_runs = len(selected_scales) * len(args.targets) * len(args.models) * len(args.seeds)
        print(f"  Total runs: {n_runs}")
        return

    # Load checkpoint if exists
    results = {}
    if SAVE_PATH.exists() and not args.force:
        with open(SAVE_PATH) as f:
            results = json.load(f)
        print(f"Loaded checkpoint: {SAVE_PATH} ({len(results)} scales)")
    elif args.force:
        print("--force enabled: Re-running all selected models and scales from scratch.")

    total_start = time.time()

    for scale_name, cfg in selected_scales.items():
        if scale_name not in results:
            results[scale_name] = {"config": cfg}

        print(f"\n{'='*80}")
        print(f"  SCALE: {scale_name} (d_h={cfg['d_h']}, hidden={cfg['hidden_features']})")
        print(f"{'='*80}")

        for target in args.targets:
            if target not in results[scale_name]:
                results[scale_name][target] = {}

            print(f"\n  Target: {TARGET_DISPLAY.get(target, target)} ({TARGET_UNITS[target]})")

            for model_type in args.models:
                # Skip if already computed with same or more seeds (unless --force)
                existing = results[scale_name][target].get(model_type)
                if not args.force and existing and len(existing.get("maes", [])) >= len(args.seeds):
                    print(f"    {model_type:6s} already done ({existing['params']:,}p, "
                          f"MAE={existing['mean_mae']:.4f}±{existing['std_mae']:.4f})")
                    continue

                print(f"    {model_type:6s} training...", end="", flush=True)
                maes = []
                param_count = 0
                total_time = 0
                for seed in args.seeds:
                    mae, p_count, t_sec = train_eval(
                        model_type, cfg, target, seed,
                        num_train=args.num_train, epochs=args.epochs,
                        batch_size=args.batch_size,
                        device=args.device, data_dir=args.data_dir
                    )
                    maes.append(mae)
                    param_count = p_count
                    total_time += t_sec

                mean_mae = float(np.mean(maes))
                std_mae = float(np.std(maes, ddof=1)) if len(maes) > 1 else 0.0

                results[scale_name][target][model_type] = {
                    "mean_mae": mean_mae,
                    "std_mae": std_mae,
                    "maes": [float(m) for m in maes],
                    "params": param_count,
                    "total_train_time_s": round(total_time, 1),
                    "mean_time_per_seed_s": round(total_time / len(args.seeds), 1),
                }
                print(f" MAE={mean_mae:.4f}±{std_mae:.4f} ({param_count:,}p, "
                      f"{total_time/len(args.seeds):.0f}s/seed)")

            # Compute PRISM vs Twin stats
            prism_data = results[scale_name][target].get("prism")
            twin_data = results[scale_name][target].get("twin")
            if prism_data and twin_data and len(prism_data["maes"]) >= 2:
                results[scale_name][target]["stats_prism_vs_twin"] = compute_stats(
                    prism_data["maes"], twin_data["maes"]
                )

            # Checkpoint after each target
            with open(SAVE_PATH, "w") as f:
                json.dump(results, f, indent=2)

    total_time = time.time() - total_start

    # Summary Table
    print(f"\n\n{'='*120}")
    print(f"  EXPERIMENT E: EXTENDED PARAMETER SCALING RESULTS")
    print(f"  Total wall time: {total_time/60:.1f} minutes")
    print(f"{'='*120}")

    for target in args.targets:
        print(f"\n  Target: {TARGET_DISPLAY.get(target, target)} ({TARGET_UNITS[target]})")
        print(f"  {'Scale':<6} {'Params':>9} │ {'PRISM MAE':>12} │ {'Twin MAE':>12} │ {'Δ(%)':>7} │ {'p-value':>9} │ {'Cohen d':>8} │ {'PaiNN MAE':>12} {'PaiNN p':>9}")
        print(f"  {'─'*6} {'─'*9} │ {'─'*12} │ {'─'*12} │ {'─'*7} │ {'─'*9} │ {'─'*8} │ {'─'*12} {'─'*9}")

        for scale_name in SCALES:
            if scale_name not in results or target not in results[scale_name]:
                continue
            t = results[scale_name][target]
            prism = t.get("prism", {})
            twin = t.get("twin", {})
            painn = t.get("painn", {})
            st = t.get("stats_prism_vs_twin", {})

            if not prism or not twin:
                continue

            p_mae = prism["mean_mae"]
            t_mae = twin["mean_mae"]
            pct = (t_mae - p_mae) / t_mae * 100 if t_mae > 0 else 0
            p_str = f"{p_mae:.4f}±{prism['std_mae']:.4f}"
            t_str = f"{t_mae:.4f}±{twin['std_mae']:.4f}"
            painn_str = f"{painn['mean_mae']:.4f}±{painn['std_mae']:.4f}" if painn else "—"
            pval = st.get("ttest_p", 1.0)
            cd = st.get("cohens_d", 0)
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""

            print(f"  {scale_name:<6} {prism['params']:>9,} │ {p_str:>12} │ {t_str:>12} │ {pct:>+5.1f}% │ {pval:>8.4f}{sig} │ {cd:>+7.2f} │ {painn_str:>12}")

    # Trend Analysis
    print(f"\n  Scaling Trend (does periodic advantage widen with capacity?):")
    for target in args.targets:
        gaps = []
        param_counts = []
        for scale_name in SCALES:
            if scale_name not in results or target not in results[scale_name]:
                continue
            t = results[scale_name][target]
            p = t.get("prism", {}).get("mean_mae")
            tw = t.get("twin", {}).get("mean_mae")
            if p is not None and tw is not None:
                gaps.append(tw - p)
                param_counts.append(t["prism"]["params"])

        if len(gaps) >= 2:
            first_gap = gaps[0]
            last_gap = gaps[-1]
            first_pct = gaps[0] / (gaps[0] + results[list(SCALES.keys())[0]][target]["prism"]["mean_mae"]) * 100
            diff = last_gap - first_gap
            trend = "WIDENS" if diff > 0.01 else "NARROWS" if diff < -0.01 else "PERSISTS"
            print(f"    {TARGET_DISPLAY.get(target, target):6s}: "
                  f"delta@{param_counts[0]:,}p = {first_gap:.4f} -> "
                  f"delta@{param_counts[-1]:,}p = {last_gap:.4f}  ->  {trend}")

    print(f"\n  Results saved to: {SAVE_PATH}")


if __name__ == "__main__":
    main()
