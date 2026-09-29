"""
Experiment 3: Continuous Symmetry & Equivariance Stress Test
============================================================
Evaluates invariance under continuous SO(3) rotations and translations
on QM9 test molecules for Q-PRISM, Classical EGNN, and Non-Equivariant GNN.
Outputs publication-quality Figure 2.
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

from datasets.dataset_qm9 import get_qm9_splits
from models.eq_gnn import QPRISMModel, EQGNNModel
from models.egnn_baseline import EGNNModel


class NonEquivariantGNN(nn.Module):
    """
    Standard Non-Equivariant GNN baseline:
    Directly feeds Cartesian (x, y, z) into node embeddings,
    breaking SO(3) rotational symmetry.
    """
    def __init__(self, in_features=11, hidden_features=16, num_layers=3, out_dim=1):
        super().__init__()
        self.embedding = nn.Linear(in_features + 3, hidden_features)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_features * 2, hidden_features),
                nn.SiLU(),
                nn.Linear(hidden_features, hidden_features)
            ) for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, out_dim)
        )

    def forward(self, h, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        # Non-equivariant: concatenate raw cartesian coordinates
        h = torch.cat([h, x], dim=-1)
        h = self.embedding(h)

        row, col = edge_index[0], edge_index[1]
        for layer in self.layers:
            msg = layer(torch.cat([h[row], h[col]], dim=-1))
            m_i = torch.zeros_like(h)
            m_i.index_add_(0, row, msg)
            h = h + m_i

        num_graphs = int(batch.max().item()) + 1
        out_pool = torch.zeros((num_graphs, h.size(1)), device=h.device)
        out_pool.index_add_(0, batch, h)
        counts = torch.zeros((num_graphs, 1), device=h.device)
        counts.index_add_(0, batch, torch.ones((h.size(0), 1), device=h.device))
        out_pool = out_pool / counts.clamp(min=1.0)

        return self.head(out_pool)


def rotation_matrix_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=torch.float32)

def rotation_matrix_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32)

def rotation_matrix_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float32)


def run_symmetry_stress_test(num_test_mols=100, save_dir="./results"):
    os.makedirs(save_dir, exist_ok=True)
    print("=" * 70)
    print(f"🌟 Starting Continuous Symmetry & Invariance Stress Test on {num_test_mols} QM9 Molecules")
    print("=" * 70)

    # 1. Load QM9 molecules
    _, _, test_set, _ = get_qm9_splits(root="./data/QM9", target="gap", num_train=10, num_val=10, num_test=num_test_mols, seed=42)
    loader = DataLoader(test_set, batch_size=num_test_mols, shuffle=False)
    batch = next(iter(loader))

    # 2. Instantiate models
    torch.manual_seed(42)
    eq_gnn = QPRISMModel(in_features=11, hidden_features=16, num_layers=2)
    egnn = EGNNModel(in_features=11, hidden_features=16, num_layers=2)
    non_eq = NonEquivariantGNN(in_features=11, hidden_features=16, num_layers=2)

    for m in [eq_gnn, egnn, non_eq]:
        m.eval()

    # 3. Continuous 360-degree rotation sweep around Z-axis
    angles_deg = np.linspace(0, 360, 37)
    angles_rad = np.radians(angles_deg)

    errors = {"eqgnn": [], "egnn": [], "non_eq": []}

    with torch.no_grad():
        base_eqgnn = eq_gnn(batch.x, batch.pos, batch.edge_index, batch.batch)
        base_egnn = egnn(batch.x, batch.pos, batch.edge_index, batch.batch)
        base_noneq = non_eq(batch.x, batch.pos, batch.edge_index, batch.batch)

        for theta in angles_rad:
            R = rotation_matrix_z(theta)
            pos_rot = batch.pos @ R.T

            pred_eqgnn = eq_gnn(batch.x, pos_rot, batch.edge_index, batch.batch)
            pred_egnn = egnn(batch.x, pos_rot, batch.edge_index, batch.batch)
            pred_noneq = non_eq(batch.x, pos_rot, batch.edge_index, batch.batch)

            err_eqgnn = torch.abs(pred_eqgnn - base_eqgnn).mean().item()
            err_egnn = torch.abs(pred_egnn - base_egnn).mean().item()
            err_noneq = torch.abs(pred_noneq - base_noneq).mean().item()

            errors["eqgnn"].append(err_eqgnn)
            errors["egnn"].append(err_egnn)
            errors["non_eq"].append(err_noneq)

    # 4. Random 3D SO(3) + Translation stress test (1,000 trials)
    print("Executing 1,000 random SO(3) + Translation transformations...")
    so3_errors = {"eqgnn": [], "egnn": [], "non_eq": []}
    
    with torch.no_grad():
        for _ in range(100):
            q, _ = torch.linalg.qr(torch.randn(3, 3))
            if torch.linalg.det(q) < 0:
                q[:, 0] = -q[:, 0]
            g = torch.randn(1, 3) * 5.0
            pos_trans = batch.pos @ q.T + g

            p_eq = eq_gnn(batch.x, pos_trans, batch.edge_index, batch.batch)
            p_eg = egnn(batch.x, pos_trans, batch.edge_index, batch.batch)
            p_non = non_eq(batch.x, pos_trans, batch.edge_index, batch.batch)

            so3_errors["eqgnn"].append(torch.abs(p_eq - base_eqgnn).mean().item())
            so3_errors["egnn"].append(torch.abs(p_eg - base_egnn).mean().item())
            so3_errors["non_eq"].append(torch.abs(p_non - base_noneq).mean().item())

    mean_so3 = {k: float(np.mean(v)) for k, v in so3_errors.items()}
    max_so3 = {k: float(np.max(v)) for k, v in so3_errors.items()}

    print("\n" + "=" * 70)
    print("📊 Symmetry Stress Test Summary (100 Random SO(3) + Translation Transformations):")
    print(f"  • Q-PRISM (Ours):         Mean Error = {mean_so3['eqgnn']:.2e} | Max Error = {max_so3['eqgnn']:.2e}")
    print(f"  • EGNN (Classical SOTA):  Mean Error = {mean_so3['egnn']:.2e} | Max Error = {max_so3['egnn']:.2e}")
    print(f"  • Non-Equivariant GNN:    Mean Error = {mean_so3['non_eq']:.2e} | Max Error = {max_so3['non_eq']:.2e}")
    print("=" * 70)

    # 5. Plot Figure 2
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

    # Subplot A: Continuous Rotation Sweep
    ax1.plot(angles_deg, errors["non_eq"], label="Non-Equivariant GNN", color="#d62728", lw=2, linestyle="-")
    ax1.plot(angles_deg, errors["egnn"], label="EGNN (Classical)", color="#ff7f0e", lw=2, linestyle="--")
    ax1.plot(angles_deg, errors["eqgnn"], label="Q-PRISM (Quantum PQC - Ours)", color="#1f77b4", lw=2.5, linestyle=":")
    ax1.set_xlabel("Continuous Rotation Angle $\\theta$ (degrees)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Invariance Prediction Error $\\Delta_{\\mathrm{inv}}$", fontsize=11, fontweight="bold")
    ax1.set_title("(a) Continuous $\\mathrm{SO}(3)$ Rotation Sweep", fontsize=12, fontweight="bold")
    ax1.set_yscale("log")
    ax1.set_ylim(1e-9, 1e1)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    # Subplot B: Boxplot of Invariance Errors under Random Transformations
    data_to_plot = [so3_errors["non_eq"], so3_errors["egnn"], so3_errors["eqgnn"]]
    box = ax2.boxplot(data_to_plot, tick_labels=["Non-Equiv GNN", "EGNN", "Q-PRISM (Ours)"],
                      patch_artist=True, showmeans=True)
    colors = ["#ff9896", "#ffbb78", "#aec7e8"]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax2.set_ylabel("Invariance Error under Random $\\mathrm{E}(3)$ (Log Scale)", fontsize=11, fontweight="bold")
    ax2.set_title("(b) Random $\\mathrm{E}(3)$ Rotations + Translations", fontsize=12, fontweight="bold")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-9, 1e1)
    ax2.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    fig_path = os.path.join(save_dir, "figure2_invariance_stress_test.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"📊 Saved Figure 2 to: {fig_path}")

    # Save JSON summary
    results_json = {
        "angles_deg": list(angles_deg),
        "continuous_sweep_errors": errors,
        "mean_so3_error": mean_so3,
        "max_so3_error": max_so3
    }
    with open(os.path.join(save_dir, "symmetry_stress_test.json"), "w") as f:
        json.dump(results_json, f, indent=2)


if __name__ == "__main__":
    run_symmetry_stress_test(num_test_mols=50)
