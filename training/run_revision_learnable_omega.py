"""
Revision Experiment: Learnable ω₀ for Periodic-EGNN
=====================================================
Addresses: R1 Q-1, R3 W-2, EIC Q-1 (3 reviewers independently ask this)
  "Did you attempt making ω₀ a learnable parameter?
   Did it converge to physically meaningful frequencies?"

Strategy:
  - Modify TrigHead to make ω₀ an nn.Parameter (per-layer)
  - Compare against fixed ω₀ = 30.0 baseline
  - Report: final ω₀ values, convergence, MAE comparison

Models (all at ~2,527 params):
  1. Periodic-EGNN (fixed ω₀ = 30.0) — baseline
  2. Periodic-EGNN (learnable ω₀, init=30.0) — per-head learnable
  3. Periodic-EGNN (learnable ω₀, init=10.0) — different init
  4. Classical Twin (tanh) — reference

Targets: HOMO, LUMO, Gap
Seeds: 42, 43, 44 (3 seeds for mean ± std)

Expected runtime:
  - 4 models × 3 targets × 3 seeds × ~8s = ~5 min total
  - (All classical, no PQC overhead)

Usage:
  cd /Users/kkyn/Documents/Work/Quantum/qgnn
  .pixi/envs/default/bin/python training/run_revision_learnable_omega.py
  .pixi/envs/default/bin/python training/run_revision_learnable_omega.py --dry-run

Output:
  results/revision_learnable_omega.json
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from datasets.dataset_qm9 import get_qm9_splits
from models.eq_gnn import GaussianRBF, compute_edge_mean_angles, N_RBF
from models.egnn_baseline import count_parameters
from models.classical_twin import ClassicalTwinModelV2
from models.classical_trig_twin import PRISMModelV2, TrigTwinModelV2


TARGET_UNITS = {"homo": "eV", "lumo": "eV", "gap": "eV"}
SAVE_PATH = PROJECT_ROOT / "results" / "revision_learnable_omega.json"


# =============================================================================
# Learnable ω₀ TrigHead
# =============================================================================

class LearnableOmegaTrigHead(nn.Module):
    """
    SIREN head with learnable ω₀ (per-head scalar parameter).

    Identical to TrigHead except ω₀ is an nn.Parameter that is
    optimized via backpropagation alongside all other weights.

    This addresses the reviewer concern: "Does ω₀ converge to
    physically meaningful frequencies?"
    """
    def __init__(self, in_features, n_q_layers=2, n_rbf=N_RBF, omega_0_init=30.0):
        super().__init__()
        self.feature_proj = nn.Linear(in_features, 2)

        # ω₀ as a LEARNABLE parameter (1 scalar per head)
        self.log_omega_0 = nn.Parameter(torch.tensor(math.log(omega_0_init)))
        # Using log-space to ensure ω₀ > 0 during optimization

        input_dim = 4 + n_rbf + 1
        n_outputs = 8
        hidden = 4

        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, n_outputs, bias=False)

        # SIREN init with initial ω₀
        self._siren_init(omega_0_init)

    def _siren_init(self, omega_0):
        n_in1 = self.linear1.in_features
        with torch.no_grad():
            self.linear1.weight.uniform_(-1.0 / n_in1, 1.0 / n_in1)
            if self.linear1.bias is not None:
                self.linear1.bias.uniform_(-1.0 / n_in1, 1.0 / n_in1)
        n_in2 = self.linear2.in_features
        bound2 = math.sqrt(6.0 / n_in2) / omega_0
        with torch.no_grad():
            self.linear2.weight.uniform_(-bound2, bound2)

    @property
    def omega_0(self):
        return torch.exp(self.log_omega_0)

    def forward(self, h_row, h_col, rbf_features, angles):
        h_i_proj = self.feature_proj(h_row)
        h_j_proj = self.feature_proj(h_col)
        inputs = torch.cat([h_i_proj, h_j_proj, rbf_features, angles], dim=-1)
        x = torch.sin(self.omega_0 * self.linear1(inputs))
        return torch.tanh(self.linear2(x))


class LearnableOmegaLayer(nn.Module):
    """GNN layer with learnable ω₀ SIREN heads."""
    def __init__(self, in_features, n_q_layers=2, n_rbf=N_RBF, omega_0_init=30.0):
        super().__init__()
        self.rbf = GaussianRBF(n_rbf=n_rbf)
        self.head_electronic = LearnableOmegaTrigHead(
            in_features, n_q_layers=n_q_layers, n_rbf=n_rbf, omega_0_init=omega_0_init)
        self.head_topological = LearnableOmegaTrigHead(
            in_features, n_q_layers=n_q_layers, n_rbf=n_rbf, omega_0_init=omega_0_init)
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


class LearnableOmegaModel(nn.Module):
    """Full Periodic-EGNN with learnable ω₀."""
    def __init__(self, in_features=11, hidden_features=16, num_layers=2, out_dim=1,
                 n_q_layers=2, n_rbf=N_RBF, omega_0_init=30.0):
        super().__init__()
        self.embedding = nn.Linear(in_features, hidden_features)
        self.layers = nn.ModuleList([
            LearnableOmegaLayer(hidden_features, n_q_layers=n_q_layers,
                                n_rbf=n_rbf, omega_0_init=omega_0_init)
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

    def get_omega_values(self):
        """Extract all learned ω₀ values from the model."""
        omegas = {}
        for i, layer in enumerate(self.layers):
            omegas[f"layer{i}_electronic"] = layer.head_electronic.omega_0.item()
            omegas[f"layer{i}_topological"] = layer.head_topological.omega_0.item()
        return omegas


# =============================================================================
# Training
# =============================================================================

def train_and_evaluate(model, target, seed, num_train=1000, num_val=250, num_test=500,
                       epochs=15, batch_size=16, lr=1e-3, track_omega=False):
    """Train a model and return test MAE + optional omega trajectory."""
    device = torch.device("cpu")

    train_set, val_set, test_set, (mean, std) = get_qm9_splits(
        root="./data/QM9", target=target,
        num_train=num_train, num_val=num_val, num_test=num_test, seed=seed
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    torch.manual_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.MSELoss()

    omega_trajectory = []

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            y_norm = (batch.y - mean) / std
            pred = model(batch.x, batch.pos, batch.edge_index, batch.batch)
            loss = criterion(pred, y_norm)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Track ω₀ values during training
        if track_omega and hasattr(model, 'get_omega_values'):
            omegas = model.get_omega_values()
            omega_trajectory.append({"epoch": epoch, **omegas})

    model.eval()
    total_mae = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred_norm = model(batch.x, batch.pos, batch.edge_index, batch.batch)
            pred = pred_norm * std + mean
            total_mae += torch.abs(pred - batch.y).sum().item()
            total_samples += batch.num_graphs

    result = {"test_mae": total_mae / total_samples}
    if track_omega:
        result["omega_trajectory"] = omega_trajectory
        if hasattr(model, 'get_omega_values'):
            result["final_omegas"] = model.get_omega_values()
    return result


def main():
    parser = argparse.ArgumentParser(description="Learnable ω₀ Experiment")
    parser.add_argument("--targets", nargs="+", default=["homo", "lumo", "gap"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        args.targets = ["homo"]
        args.seeds = [42]
        args.epochs = 3
        print("🏃 DRY RUN: 1 seed, 3 epochs, HOMO only\n")

    # ── Model configurations ──
    model_configs = {
        "periodic_fixed_30": {
            "label": "PRISM (ω₀=30, fixed)",
            "factory": lambda: PRISMModelV2(
                in_features=11, hidden_features=16, num_layers=2,
                n_q_layers=2, out_dim=1, head_type="siren", omega_0=30.0),
            "track_omega": False,
        },
        "periodic_learnable_30": {
            "label": "PRISM (ω₀=30, learnable)",
            "factory": lambda: LearnableOmegaModel(
                in_features=11, hidden_features=16, num_layers=2,
                n_q_layers=2, out_dim=1, omega_0_init=30.0),
            "track_omega": True,
        },
        "periodic_learnable_10": {
            "label": "PRISM (ω₀=10, learnable)",
            "factory": lambda: LearnableOmegaModel(
                in_features=11, hidden_features=16, num_layers=2,
                n_q_layers=2, out_dim=1, omega_0_init=10.0),
            "track_omega": True,
        },
        "classical_twin": {
            "label": "Classical Twin (tanh)",
            "factory": lambda: ClassicalTwinModelV2(
                in_features=11, hidden_features=16, num_layers=2,
                n_q_layers=2, out_dim=1),
            "track_omega": False,
        },
    }

    # ── Verify parameter counts ──
    print("=" * 70)
    print("LEARNABLE ω₀ EXPERIMENT")
    print("=" * 70)

    for key, cfg in model_configs.items():
        m = cfg["factory"]()
        p = count_parameters(m)
        # Note: learnable ω₀ adds 4 params (1 per head × 2 heads × 2 layers)
        print(f"  {cfg['label']:45s}  {p:,} params")
        del m

    # ── Load checkpoint ──
    if SAVE_PATH.exists():
        with open(SAVE_PATH) as f:
            results = json.load(f)
    else:
        results = {"metadata": {
            "description": "Learnable omega_0 experiment for Periodic-EGNN",
            "targets": args.targets, "seeds": args.seeds, "epochs": args.epochs,
        }}

    # ── Run all experiments ──
    for target in args.targets:
        if target not in results:
            results[target] = {}

        print(f"\n{'='*60}")
        print(f"TARGET: {target.upper()}")
        print(f"{'='*60}")

        for model_key, cfg in model_configs.items():
            if model_key not in results[target]:
                results[target][model_key] = {"maes": {}, "omegas": {}}

            print(f"\n  Model: {cfg['label']}")

            for seed in args.seeds:
                seed_key = str(seed)
                if seed_key in results[target][model_key]["maes"]:
                    cached = results[target][model_key]["maes"][seed_key]
                    print(f"    Seed {seed}: {cached:.4f} (cached)")
                    continue

                t0 = time.time()
                model = cfg["factory"]()
                res = train_and_evaluate(
                    model, target, seed,
                    epochs=args.epochs,
                    track_omega=cfg["track_omega"],
                )
                elapsed = time.time() - t0

                results[target][model_key]["maes"][seed_key] = res["test_mae"]
                if "final_omegas" in res:
                    results[target][model_key]["omegas"][seed_key] = res["final_omegas"]
                if "omega_trajectory" in res:
                    results[target][model_key].setdefault("trajectories", {})[seed_key] = res["omega_trajectory"]

                omega_str = ""
                if "final_omegas" in res:
                    vals = list(res["final_omegas"].values())
                    omega_str = f" | ω₀ → [{min(vals):.1f}, {max(vals):.1f}]"

                print(f"    Seed {seed}: {res['test_mae']:.4f} ({elapsed:.1f}s){omega_str}")

                with open(SAVE_PATH, "w") as f:
                    json.dump(results, f, indent=2)

            # Summary stats
            maes = list(results[target][model_key]["maes"].values())
            results[target][model_key]["mean"] = float(np.mean(maes))
            results[target][model_key]["std"] = float(np.std(maes))
            print(f"    >> Mean: {np.mean(maes):.4f} ± {np.std(maes):.4f}")

            # Omega summary
            if results[target][model_key].get("omegas"):
                all_omegas = {}
                for seed_key, omegas in results[target][model_key]["omegas"].items():
                    for k, v in omegas.items():
                        all_omegas.setdefault(k, []).append(v)
                print(f"    >> Final ω₀:")
                for k, vals in sorted(all_omegas.items()):
                    print(f"       {k}: {np.mean(vals):.2f} ± {np.std(vals):.2f}")

    with open(SAVE_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # ── Summary table ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for target in args.targets:
        print(f"\n  {target.upper()} (eV):")
        print(f"  {'Model':45s} | {'MAE':>20s}")
        print(f"  {'─'*45}─┼{'─'*22}")
        for model_key, cfg in model_configs.items():
            if model_key in results[target]:
                m = results[target][model_key].get("mean", 0)
                s = results[target][model_key].get("std", 0)
                n = len(results[target][model_key].get("maes", {}))
                print(f"  {cfg['label']:45s} | {m:.4f} ± {s:.4f} ({n}s)")

    print(f"\n💾 Results saved to: {SAVE_PATH}")
    print("✅ Learnable ω₀ experiment complete!")


if __name__ == "__main__":
    main()
