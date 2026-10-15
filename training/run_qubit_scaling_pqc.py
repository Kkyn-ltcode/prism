"""Experiment A: Multi-qubit scaling benchmark across 4 to 12 qubits."""

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
import pennylane as qml
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from torch_geometric.loader import DataLoader
from datasets.dataset_qm9 import get_qm9_splits
from models.eq_gnn import GaussianRBF, compute_edge_mean_angles, N_RBF
from models.egnn_baseline import count_parameters

# Configuration

QUBIT_VALUES = [4, 6, 8, 10, 12]
DEFAULT_TARGETS = ["gap"]  # Focused on gap; extend if desired

TARGET_UNITS = {"homo": "eV", "lumo": "eV", "gap": "eV", "mu": "Debye"}
TARGET_DISPLAY = {"homo": "HOMO", "lumo": "LUMO", "gap": "Gap", "mu": "Dipole"}

SAVE_PATH = PROJECT_ROOT / "results" / "experiment_a_qubit_scaling.json"
DRYRUN_PATH = PROJECT_ROOT / "results" / "experiment_a_qubit_scaling_dryrun.json"
TABLE_PATH = PROJECT_ROOT / "results" / "experiment_a_qubit_scaling_table.tex"
PAPER_TABLE_PATH = PROJECT_ROOT / "paper" / "experiment_a_qubit_scaling_table.tex"
FIG_PATH = PROJECT_ROOT / "results" / "figure_qubit_scaling.png"
PAPER_FIG_PATH = PROJECT_ROOT / "paper" / "figures" / "figure_qubit_scaling.png"

DEFAULT_PROTOCOL = {
    "num_train": 1000,
    "num_val": 250,
    "num_test": 500,
    "batch_size": 16,
    "lr": 1e-3,
    "epochs": 15,
}

# Hardware timing constants (nanoseconds)
T2E_ARM_NS = 380e3      # Arm qubit T2E coherence
T2E_IBM_NS = 50e3       # IBM Transmon T2 coherence
GATE_1Q_ARM_NS = 5      # Single-qubit gate time, Arm
GATE_2Q_ARM_NS = 17     # Two-qubit gate time, Arm
GATE_1Q_IBM_NS = 25     # Single-qubit gate time, IBM
GATE_2Q_IBM_NS = 300    # Two-qubit gate time, IBM


# ═════════════════════════════════════════════
# NQ-Aware Quantum Circuit & Head
# (imported from benchmark_multiqubit_scaling)
# ═════════════════════════════════════════════

def _create_qnode_nq(nq):
    """Create a QNode for an nq-qubit PQC with default.qubit + backprop."""
    dev = qml.device("default.qubit", wires=nq)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights, dist_weights):
        K = dist_weights.shape[2]
        n_layers = weights.shape[0]
        feat_dim = nq  # nq/2 features per node × 2 nodes

        if inputs.ndim == 1:
            features = [inputs[f] for f in range(feat_dim)]
            rbf = inputs[feat_dim:feat_dim + K]
            angle = inputs[feat_dim + K]
        else:
            features = [inputs[:, f] for f in range(feat_dim)]
            rbf = inputs[:, feat_dim:feat_dim + K]
            angle = inputs[:, feat_dim + K]

        for l in range(n_layers):
            for q in range(nq):
                qml.RY(features[q], wires=q)
            for q in range(nq):
                if inputs.ndim == 1:
                    phase = sum(dist_weights[l, q, k] * rbf[k] for k in range(K))
                else:
                    phase = sum(dist_weights[l, q, k] * rbf[:, k] for k in range(K))
                qml.RZ(phase, wires=q)
            for q in range(nq):
                qml.RY(angle, wires=q)
            for q in range(nq):
                qml.RX(weights[l, q, 0], wires=q)
                qml.RY(weights[l, q, 1], wires=q)
                qml.RZ(weights[l, q, 2], wires=q)
            for q in range(nq):
                qml.CNOT(wires=[q, (q + 1) % nq])

        observables = []
        for q in range(nq):
            observables.append(qml.expval(qml.PauliZ(q)))
        for q in range(nq):
            observables.append(qml.expval(qml.PauliZ(q) @ qml.PauliZ((q + 1) % nq)))
        return observables

    return circuit


class QuantumHeadNQ(nn.Module):
    """Quantum head generalized to nq qubits."""
    def __init__(self, in_features, nq, n_q_layers=2, n_rbf=N_RBF):
        super().__init__()
        self.nq = nq
        self.feature_proj = nn.Linear(in_features, nq // 2)
        qnode = _create_qnode_nq(nq)
        weight_shapes = {
            "weights": (n_q_layers, nq, 3),
            "dist_weights": (n_q_layers, nq, n_rbf),
        }
        self.q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)
        with torch.no_grad():
            self.q_layer.dist_weights.uniform_(0.05, 0.2)

    def forward(self, h_row, h_col, rbf_features, angles):
        h_i_proj = self.feature_proj(h_row)
        h_j_proj = self.feature_proj(h_col)
        q_inputs = torch.cat([h_i_proj, h_j_proj, rbf_features, angles], dim=-1)
        return self.q_layer(q_inputs)  # [E, 2*nq]


# NQ-Aware Classical Heads (parameter-matched)

def _compute_head_hidden(nq, n_q_layers=2, n_rbf=N_RBF):
    """Compute the MLP hidden dim that matches the PQC parameter count."""
    pqc_params = n_q_layers * nq * (3 + n_rbf)
    input_dim = nq + n_rbf + 1
    n_outputs = 2 * nq
    denominator = input_dim + 1 + n_outputs
    # Round to nearest int; at least 1
    hidden = max(1, round(pqc_params / denominator))
    return hidden


class PRISMHeadNQ(nn.Module):
    """PRISM head (sin → tanh) generalized to nq qubits, parameter-matched."""
    def __init__(self, in_features, nq, n_q_layers=2, n_rbf=N_RBF, omega_0=30.0):
        super().__init__()
        self.nq = nq
        self.omega_0 = omega_0
        self.feature_proj = nn.Linear(in_features, nq // 2)

        input_dim = nq + n_rbf + 1
        n_outputs = 2 * nq
        hidden = _compute_head_hidden(nq, n_q_layers, n_rbf)

        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, n_outputs, bias=False)

        # SIREN init
        with torch.no_grad():
            self.linear1.weight.uniform_(-1.0 / input_dim, 1.0 / input_dim)
            if self.linear1.bias is not None:
                self.linear1.bias.uniform_(-1.0 / input_dim, 1.0 / input_dim)
            bound2 = math.sqrt(6.0 / hidden) / omega_0
            self.linear2.weight.uniform_(-bound2, bound2)

    def forward(self, h_row, h_col, rbf_features, angles):
        h_i_proj = self.feature_proj(h_row)
        h_j_proj = self.feature_proj(h_col)
        x = torch.cat([h_i_proj, h_j_proj, rbf_features, angles], dim=-1)
        h = torch.sin(self.omega_0 * self.linear1(x))
        return torch.tanh(self.linear2(h))


class ClassicalHeadNQ(nn.Module):
    """Classical twin head (SiLU → tanh) generalized to nq qubits, parameter-matched."""
    def __init__(self, in_features, nq, n_q_layers=2, n_rbf=N_RBF):
        super().__init__()
        self.nq = nq
        self.feature_proj = nn.Linear(in_features, nq // 2)

        input_dim = nq + n_rbf + 1
        n_outputs = 2 * nq
        hidden = _compute_head_hidden(nq, n_q_layers, n_rbf)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_outputs, bias=False),
        )

    def forward(self, h_row, h_col, rbf_features, angles):
        h_i_proj = self.feature_proj(h_row)
        h_j_proj = self.feature_proj(h_col)
        x = torch.cat([h_i_proj, h_j_proj, rbf_features, angles], dim=-1)
        return torch.tanh(self.mlp(x))


# ═════════════════════════════════════════════
# NQ-Aware Layer & Model (works for all 3 types)
# ═════════════════════════════════════════════

class NQLayer(nn.Module):
    """Message-passing layer parameterized by qubit count and head type."""
    def __init__(self, in_features, nq, head_type="qprism", n_q_layers=2, n_rbf=N_RBF):
        super().__init__()
        self.nq = nq
        self.rbf = GaussianRBF(n_rbf=n_rbf)

        if head_type == "qprism":
            self.head_electronic = QuantumHeadNQ(in_features, nq, n_q_layers, n_rbf)
            self.head_topological = QuantumHeadNQ(in_features, nq, n_q_layers, n_rbf)
        elif head_type == "prism":
            self.head_electronic = PRISMHeadNQ(in_features, nq, n_q_layers, n_rbf)
            self.head_topological = PRISMHeadNQ(in_features, nq, n_q_layers, n_rbf)
        elif head_type == "twin":
            self.head_electronic = ClassicalHeadNQ(in_features, nq, n_q_layers, n_rbf)
            self.head_topological = ClassicalHeadNQ(in_features, nq, n_q_layers, n_rbf)
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

        self.coord_gate = nn.Linear(2, 1)
        msg_per_head = 2 * nq - 1  # all outputs except one used for coord
        msg_dim = msg_per_head * 2
        self.node_mlp = nn.Sequential(
            nn.Linear(in_features + msg_dim, in_features),
            nn.SiLU(),
            nn.Linear(in_features, in_features),
        )

    def forward(self, h, x, edge_index):
        row, col = edge_index[0], edge_index[1]
        rel_x = x[row] - x[col]
        d_ij = torch.sum(rel_x ** 2, dim=-1, keepdim=True)
        rbf_features = self.rbf(d_ij)
        angles = compute_edge_mean_angles(x, edge_index)

        out_e = self.head_electronic(h[row], h[col], rbf_features, angles)
        out_t = self.head_topological(h[row], h[col], rbf_features, angles)

        # Split: one output per head used for coordinate gate, rest for message
        coord_idx = self.nq - 1
        msg_e = torch.cat([out_e[:, :coord_idx], out_e[:, coord_idx + 1:]], dim=-1)
        msg_t = torch.cat([out_t[:, :coord_idx], out_t[:, coord_idx + 1:]], dim=-1)
        m_ij = torch.cat([msg_e, msg_t], dim=-1)

        phi_x_raw = torch.cat([out_e[:, coord_idx:coord_idx + 1],
                                out_t[:, coord_idx:coord_idx + 1]], dim=-1)
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


class NQModel(nn.Module):
    """Full model parameterized by qubit count and head type."""
    def __init__(self, head_type="qprism", in_features=11, hidden_features=16,
                 num_layers=2, out_dim=1, nq=4, n_q_layers=2, n_rbf=N_RBF):
        super().__init__()
        self.embedding = nn.Linear(in_features, hidden_features)
        self.layers = nn.ModuleList([
            NQLayer(hidden_features, nq, head_type, n_q_layers, n_rbf)
            for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, out_dim),
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

def train_eval(model, target, seed, protocol):
    """Train one model → return (test_mae, wall_time)."""
    device = torch.device("cpu")
    data_dir = str(PROJECT_ROOT / "data" / "QM9")
    train_set, val_set, test_set, (mean, std) = get_qm9_splits(
        root=data_dir, target=target,
        num_train=protocol["num_train"],
        num_val=protocol["num_val"],
        num_test=protocol["num_test"],
        seed=seed,
    )
    train_loader = DataLoader(train_set, batch_size=protocol["batch_size"], shuffle=True)
    test_loader = DataLoader(test_set, batch_size=protocol["batch_size"], shuffle=False)

    torch.manual_seed(seed)
    model = model.to(device)
    epochs = protocol["epochs"]
    optimizer = torch.optim.Adam(model.parameters(), lr=protocol["lr"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.MSELoss()

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.pos, batch.edge_index, batch.batch)
            loss = criterion(pred, (batch.y - mean) / std)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        scheduler.step()
    wall_time = time.time() - t0

    model.eval()
    total_abs, n = 0.0, 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.pos, batch.edge_index, batch.batch)
            pred_orig = pred * std + mean
            total_abs += (pred_orig - batch.y).abs().sum().item()
            n += batch.y.numel()

    return total_abs / n, wall_time


# Statistics


def compute_paired_stats(maes_a, maes_b):
    a, b = np.array(maes_a), np.array(maes_b)
    n = len(a)
    diff = a - b
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0
    try:
        _, wilcoxon_p = stats.wilcoxon(a, b)
    except ValueError:
        wilcoxon_p = 1.0
    _, ttest_p = stats.ttest_rel(a, b)
    return {
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "cohens_d": float(cohens_d),
        "wilcoxon_p": float(wilcoxon_p),
        "ttest_p": float(ttest_p),
        "n": n,
    }


def bh_fdr(pvalues, alpha=0.05):
    n = len(pvalues)
    if n == 0:
        return []
    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    adjusted = [0.0] * n
    prev = 1.0
    for rank_minus1 in range(n - 1, -1, -1):
        orig_idx, p = indexed[rank_minus1]
        rank = rank_minus1 + 1
        adj = min(prev, p * n / rank)
        adjusted[orig_idx] = adj
        prev = adj
    return [(pvalues[i], adjusted[i], adjusted[i] < alpha) for i in range(n)]


# Hardware coherence budget

def compute_hw_budget(nq, n_q_layers=2, n_heads=2):
    """Compute circuit timing and coherence ratios."""
    # Gates per PQC evaluation (both heads combined)
    single_gates = n_heads * n_q_layers * 6 * nq  # 3 rotations × 2 (feature + variational)
    two_q_gates = n_heads * n_q_layers * nq        # ring CNOT

    time_arm = single_gates * GATE_1Q_ARM_NS + two_q_gates * GATE_2Q_ARM_NS
    time_ibm = single_gates * GATE_1Q_IBM_NS + two_q_gates * GATE_2Q_IBM_NS

    return {
        "circuit_time_arm_ns": time_arm,
        "circuit_time_ibm_ns": time_ibm,
        "coherence_ratio_arm": time_arm / T2E_ARM_NS,
        "coherence_ratio_ibm": time_ibm / T2E_IBM_NS,
        "statevector_bytes": (2 ** nq) * 16,  # complex128
    }


# LaTeX table

def generate_latex_table(results, targets, qubits):
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Multi-Qubit Scaling: Test MAE for Q-PRISM, PRISM, and Classical Twin",
        r"  at varying qubit counts ($N{=}1{,}000$, 5 seeds, target: HOMO-LUMO Gap).}",
        r"\label{tab:qubit_scaling}",
        r"\small",
        r"\begin{tabular}{rcccccc}",
        r"\toprule",
        r"$n_q$ & Q-PRISM & PRISM & Twin & $\Delta_\%$(Q-P) & $p$ & Time/seed \\",
        r"\midrule",
    ]

    for target in targets:
        for nq in qubits:
            nqkey = str(nq)
            td = results.get(nqkey, {}).get(target, {})
            qp = td.get("qprism", {})
            pr = td.get("prism", {})
            tw = td.get("twin", {})
            st = td.get("stats_qprism_vs_prism", {})

            if not qp:
                continue

            qp_str = f"${qp['mean_mae']:.4f}\\pm{qp['std_mae']:.4f}$"
            pr_str = f"${pr['mean_mae']:.4f}\\pm{pr['std_mae']:.4f}$" if pr else "---"
            tw_str = f"${tw['mean_mae']:.4f}\\pm{tw['std_mae']:.4f}$" if tw else "---"

            if pr and pr["mean_mae"] > 0:
                delta_pct = (pr["mean_mae"] - qp["mean_mae"]) / pr["mean_mae"] * 100
                delta_str = f"${delta_pct:+.1f}\\%$"
            else:
                delta_str = "---"

            pval = st.get("fdr_adjusted_p", st.get("ttest_p", 1.0))
            p_str = "$<0.001$" if pval < 0.001 else f"${pval:.3f}$" if pval < 1.0 else "---"

            # Time per seed
            qp_time = qp.get("mean_time_per_seed_s", 0)
            if qp_time >= 3600:
                time_str = f"{qp_time/3600:.1f}h"
            elif qp_time >= 60:
                time_str = f"{qp_time/60:.0f}m"
            else:
                time_str = f"{qp_time:.0f}s"

            lines.append(
                rf"{nq} & {qp_str} & {pr_str} & {tw_str} & {delta_str} & {p_str} & {time_str} \\"
            )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


# Figure

def generate_figure(results, targets, qubits, fig_path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    target = targets[0]  # Primary target

    nq_vals, qp_m, qp_s, pr_m, pr_s, tw_m, tw_s = [], [], [], [], [], [], []
    times_qp, times_pr = [], []

    for nq in qubits:
        td = results.get(str(nq), {}).get(target, {})
        qp = td.get("qprism")
        pr = td.get("prism")
        tw = td.get("twin")
        if qp:
            nq_vals.append(nq)
            qp_m.append(qp["mean_mae"]); qp_s.append(qp["std_mae"])
            pr_m.append(pr["mean_mae"] if pr else np.nan)
            pr_s.append(pr["std_mae"] if pr else 0)
            tw_m.append(tw["mean_mae"] if tw else np.nan)
            tw_s.append(tw["std_mae"] if tw else 0)
            times_qp.append(qp.get("mean_time_per_seed_s", 0))
            times_pr.append(pr.get("mean_time_per_seed_s", 0) if pr else 0)

    nq_vals = np.array(nq_vals)

    # (a) MAE vs Qubit Count
    ax = axes[0]
    ax.errorbar(nq_vals, qp_m, yerr=qp_s, marker="o", capsize=3, linewidth=2,
                color="#1f77b4", label="Q-PRISM (PQC)")
    ax.errorbar(nq_vals, pr_m, yerr=pr_s, marker="s", capsize=3, linewidth=2,
                color="#2ca02c", label="PRISM (sin)")
    ax.errorbar(nq_vals, tw_m, yerr=tw_s, marker="^", capsize=3, linewidth=2,
                color="#d62728", label="Twin (SiLU)")
    ax.set_xlabel("Qubit Count $n_q$", fontsize=11)
    ax.set_ylabel(f"Test MAE ({TARGET_UNITS.get(target, '')})", fontsize=11)
    ax.set_title("(a) Dequantization at Scale", fontsize=12, fontweight="bold")
    ax.set_xticks(qubits)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (b) Training Time Scaling
    ax = axes[1]
    ax.semilogy(nq_vals, times_qp, marker="o", linewidth=2, color="#1f77b4", label="Q-PRISM")
    ax.semilogy(nq_vals, times_pr, marker="s", linewidth=2, color="#2ca02c", label="PRISM")
    ax.set_xlabel("Qubit Count $n_q$", fontsize=11)
    ax.set_ylabel("Training Time / seed (s)", fontsize=11)
    ax.set_title("(b) Computational Cost", fontsize=12, fontweight="bold")
    ax.set_xticks(qubits)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (c) Hardware Coherence Budget
    ax = axes[2]
    hw_arm, hw_ibm = [], []
    for nq in qubits:
        hw = compute_hw_budget(nq)
        hw_arm.append(hw["coherence_ratio_arm"])
        hw_ibm.append(hw["coherence_ratio_ibm"])
    ax.semilogy(qubits, hw_arm, marker="o", color="#2ca02c", linewidth=2, label="Arm Qubit")
    ax.semilogy(qubits, hw_ibm, marker="s", color="#d62728", linewidth=2, label="IBM Transmon")
    ax.axhline(0.01, color="green", linestyle="--", alpha=0.5, label="1% budget")
    ax.axhline(0.10, color="red", linestyle="--", alpha=0.5, label="10% budget")
    ax.fill_between(qubits, 0, 0.01, color="green", alpha=0.05)
    ax.fill_between(qubits, 0.10, 10, color="red", alpha=0.05)
    ax.set_xlabel("Qubit Count $n_q$", fontsize=11)
    ax.set_ylabel("Circuit / Coherence Ratio", fontsize=11)
    ax.set_title("(c) Hardware Feasibility", fontsize=12, fontweight="bold")
    ax.set_xticks(qubits)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {fig_path}")


# Main

def main():
    parser = argparse.ArgumentParser(
        description="Experiment A: Multi-Qubit Scaling (4→12 qubits)")
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--qubits", nargs="+", type=int, default=QUBIT_VALUES)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--num-train", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--skip-qprism", action="store_true",
                        help="Skip Q-PRISM (only train classical baselines)")
    parser.add_argument("--dry-run", action="store_true",
                        help="2-epoch smoke test with N=100")
    args = parser.parse_args()

    protocol = dict(DEFAULT_PROTOCOL)
    protocol["epochs"] = args.epochs
    protocol["num_train"] = args.num_train
    protocol["batch_size"] = args.batch_size

    is_dry = args.dry_run
    if is_dry:
        protocol["epochs"] = 2
        protocol["num_train"] = 100
        protocol["num_test"] = 100
        save_path = DRYRUN_PATH
        if "--seeds" not in sys.argv:
            args.seeds = [42, 43]
        if "--qubits" not in sys.argv:
            args.qubits = [4, 6, 8]
    else:
        save_path = SAVE_PATH

    qubits = sorted(args.qubits)
    targets = args.targets
    seeds = args.seeds
    model_types = ["prism", "twin"] if args.skip_qprism else ["qprism", "prism", "twin"]

    # ── Dry-run header ──
    if is_dry:
        print("=" * 80)
        print("  DRY RUN — Smoke test (2 epochs, N=100)")
        print("=" * 80)

    print(f"\n  Qubits:  {qubits}")
    print(f"  Targets: {targets}")
    print(f"  Seeds:   {seeds}")
    print(f"  Models:  {model_types}")
    print()

    # Print param counts
    print("  Parameter counts by qubit count:")
    print(f"  {'nq':>3}  {'Q-PRISM':>9}  {'PRISM':>9}  {'Twin':>9}  "
          f"{'PQC/head':>9}  {'MLP/head':>9}  {'head_h':>6}")
    for nq in qubits:
        qp = NQModel("qprism", nq=nq)
        pr = NQModel("prism", nq=nq)
        tw = NQModel("twin", nq=nq)
        pqc_per_head = 2 * nq * (3 + N_RBF)  # n_q_layers=2
        hh = _compute_head_hidden(nq)
        mlp_per_head = hh * (nq + N_RBF + 1 + 1) + hh * 2 * nq
        print(f"  {nq:>3}  {count_parameters(qp):>9,}  {count_parameters(pr):>9,}  "
              f"{count_parameters(tw):>9,}  {pqc_per_head:>9}  {mlp_per_head:>9}  {hh:>6}")
    print()

    # ── Load checkpoint ──
    results = {}
    if save_path.exists():
        with open(save_path) as f:
            results = json.load(f)
        print(f"Loaded checkpoint: {save_path}")

    total_start = time.time()

    for nq in qubits:
        nqkey = str(nq)
        if nqkey not in results:
            results[nqkey] = {"hw_budget": compute_hw_budget(nq)}

        print(f"\n{'='*80}")
        print(f"  QUBITS: {nq}")
        print(f"{'='*80}")

        for target in targets:
            if target not in results[nqkey]:
                results[nqkey][target] = {}

            display = TARGET_DISPLAY.get(target, target)
            print(f"\n  ── Target: {display} ──")

            for kind in model_types:
                existing = results[nqkey][target].get(kind, {})
                existing_seeds = existing.get("maes_by_seed", {})

                # Check if all seeds done
                missing = [s for s in seeds if str(s) not in existing_seeds]
                if not missing and len(existing_seeds) >= len(seeds):
                    print(f"    {kind:7s} nq={nq} already done "
                          f"(MAE={existing['mean_mae']:.4f}±{existing['std_mae']:.4f})")
                    continue

                seed_maes = dict(existing_seeds)
                param_count = existing.get("params", 0)
                total_time = existing.get("total_train_time_s", 0.0)

                for seed in seeds:
                    if str(seed) in seed_maes:
                        print(f"    {kind:7s} nq={nq} seed={seed} (cached)")
                        continue

                    print(f"    {kind:7s} nq={nq} seed={seed}...", end="", flush=True)
                    torch.manual_seed(seed)
                    model = NQModel(kind, nq=nq)
                    if param_count == 0:
                        param_count = count_parameters(model)
                    mae, wall = train_eval(model, target, seed, protocol)
                    seed_maes[str(seed)] = mae
                    total_time += wall
                    print(f" MAE={mae:.4f} ({wall:.0f}s)")

                maes_list = [seed_maes[str(s)] for s in seeds if str(s) in seed_maes]
                results[nqkey][target][kind] = {
                    "mean_mae": float(np.mean(maes_list)),
                    "std_mae": float(np.std(maes_list, ddof=1)) if len(maes_list) > 1 else 0.0,
                    "maes": maes_list,
                    "maes_by_seed": seed_maes,
                    "params": param_count,
                    "total_train_time_s": round(total_time, 1),
                    "mean_time_per_seed_s": round(total_time / max(1, len(maes_list)), 1),
                }

            # Paired stats
            for cmp_name, a_key, b_key in [
                ("stats_qprism_vs_prism", "qprism", "prism"),
                ("stats_qprism_vs_twin", "qprism", "twin"),
                ("stats_prism_vs_twin", "prism", "twin"),
            ]:
                a_data = results[nqkey][target].get(a_key, {})
                b_data = results[nqkey][target].get(b_key, {})
                if a_data and b_data:
                    paired = sorted(set(a_data.get("maes_by_seed", {}).keys())
                                    & set(b_data.get("maes_by_seed", {}).keys()))
                    if len(paired) >= 3:
                        results[nqkey][target][cmp_name] = compute_paired_stats(
                            [a_data["maes_by_seed"][s] for s in paired],
                            [b_data["maes_by_seed"][s] for s in paired],
                        )

            # Checkpoint
            with open(save_path, "w") as f:
                json.dump(results, f, indent=2)

    total_wall = time.time() - total_start

    # ── BH-FDR correction ──
    all_pvals = []
    for nqkey in results:
        for target in targets:
            td = results.get(nqkey, {}).get(target, {})
            for stat_key in ["stats_qprism_vs_prism", "stats_qprism_vs_twin",
                             "stats_prism_vs_twin"]:
                st = td.get(stat_key)
                if st:
                    all_pvals.append(((nqkey, target, stat_key), st["ttest_p"]))

    if all_pvals:
        raw_ps = [p for _, p in all_pvals]
        fdr_results = bh_fdr(raw_ps)
        for i, ((nqkey, target, stat_key), _) in enumerate(all_pvals):
            _, adj_p, sig = fdr_results[i]
            results[nqkey][target][stat_key]["fdr_adjusted_p"] = adj_p
            results[nqkey][target][stat_key]["fdr_significant_005"] = sig

    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary Report
    print(f"\n\n{'='*120}")
    print(f"  EXPERIMENT A: MULTI-QUBIT SCALING RESULTS")
    print(f"  Total wall time: {total_wall/60:.1f} min")
    print(f"{'='*120}")

    for target in targets:
        display = TARGET_DISPLAY.get(target, target)
        unit = TARGET_UNITS.get(target, "")
        print(f"\n  Target: {display} ({unit})")
        print(f"  {'nq':>3}  {'Q-PRISM':>14}  {'PRISM':>14}  {'Twin':>14}  "
              f"{'Δ(Q-P)%':>8}  {'p(Q-P)':>10}  {'d(Q-P)':>7}  "
              f"{'Δ(P-T)%':>8}  {'Q time':>8}  {'P time':>8}")
        print(f"  {'─'*3}  {'─'*14}  {'─'*14}  {'─'*14}  "
              f"{'─'*8}  {'─'*10}  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*8}")

        for nq in qubits:
            nqkey = str(nq)
            td = results.get(nqkey, {}).get(target, {})
            qp = td.get("qprism", {})
            pr = td.get("prism", {})
            tw = td.get("twin", {})
            st_qp = td.get("stats_qprism_vs_prism", {})
            st_pt = td.get("stats_prism_vs_twin", {})

            qp_str = f"{qp['mean_mae']:.4f}±{qp['std_mae']:.4f}" if qp else "—"
            pr_str = f"{pr['mean_mae']:.4f}±{pr['std_mae']:.4f}" if pr else "—"
            tw_str = f"{tw['mean_mae']:.4f}±{tw['std_mae']:.4f}" if tw else "—"

            # Δ(Q-P)
            if qp and pr and pr["mean_mae"] > 0:
                d_qp = (pr["mean_mae"] - qp["mean_mae"]) / pr["mean_mae"] * 100
                d_qp_str = f"{d_qp:+.1f}%"
            else:
                d_qp_str = "—"

            # Δ(P-T) = periodic advantage
            if pr and tw and tw["mean_mae"] > 0:
                d_pt = (tw["mean_mae"] - pr["mean_mae"]) / tw["mean_mae"] * 100
                d_pt_str = f"{d_pt:+.1f}%"
            else:
                d_pt_str = "—"

            pval = st_qp.get("fdr_adjusted_p", st_qp.get("ttest_p", 1.0))
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            p_str = f"{pval:.4f}{sig}" if st_qp else "—"
            d_str = f"{st_qp.get('cohens_d', 0):+.2f}" if st_qp else "—"

            qt = qp.get("mean_time_per_seed_s", 0)
            pt = pr.get("mean_time_per_seed_s", 0)
            qt_str = f"{qt/3600:.1f}h" if qt >= 3600 else f"{qt/60:.0f}m" if qt >= 60 else f"{qt:.0f}s"
            pt_str = f"{pt/60:.0f}m" if pt >= 60 else f"{pt:.0f}s"

            print(f"  {nq:>3}  {qp_str:>14}  {pr_str:>14}  {tw_str:>14}  "
                  f"{d_qp_str:>8}  {p_str:>10}  {d_str:>7}  "
                  f"{d_pt_str:>8}  {qt_str:>8}  {pt_str:>8}")

    # Key Finding
    print(f"\n  Key Finding: Does dequantization hold at higher qubit counts?")
    for target in targets:
        display = TARGET_DISPLAY.get(target, target)
        for nq in qubits:
            td = results.get(str(nq), {}).get(target, {})
            qp = td.get("qprism", {})
            pr = td.get("prism", {})
            st = td.get("stats_qprism_vs_prism", {})
            if qp and pr:
                delta = (pr["mean_mae"] - qp["mean_mae"]) / pr["mean_mae"] * 100
                pval = st.get("fdr_adjusted_p", st.get("ttest_p", 1.0))
                verdict = ("Q-PRISM better" if delta > 2 and pval < 0.05 else
                          "PRISM better" if delta < -2 and pval < 0.05 else
                          "No significant difference")
                print(f"    {display} nq={nq:>2}: Δ={delta:+.1f}%, p={pval:.3f} → {verdict}")

    # Outputs
    try:
        latex = generate_latex_table(results, targets, qubits)
        table_path = DRYRUN_PATH.with_suffix(".tex") if is_dry else TABLE_PATH
        with open(table_path, "w") as f:
            f.write(latex)
        if not is_dry:
            with open(PAPER_TABLE_PATH, "w") as f:
                f.write(latex)
        print(f"\n  LaTeX table: {table_path}")
    except Exception as e:
        print(f"\n  LaTeX table error: {e}")

    try:
        fig_out = DRYRUN_PATH.with_suffix(".png") if is_dry else FIG_PATH
        generate_figure(results, targets, qubits, fig_out)
        if not is_dry:
            generate_figure(results, targets, qubits, PAPER_FIG_PATH)
    except Exception as e:
        print(f"\n  Figure error: {e}")

    print(f"\n  Results saved: {save_path}")


if __name__ == "__main__":
    main()
