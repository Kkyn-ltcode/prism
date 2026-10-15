"""Quantum architecture ablation suite."""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
import torch
import torch.nn as nn
import pennylane as qml
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

from datasets.dataset_qm9 import get_qm9_splits
from models.eq_gnn import (
    QPRISMModelV2, QPRISMLayerV2, EQGNNModelV2, EQGNNLayerV2,
    GaussianRBF, QuantumHead, compute_edge_mean_angles, _create_v2_qnode, N_RBF, n_qubits
)

# Configurable ablation quantum edge circuit
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

def create_ablation_qnode(encoding_type="multi_axis", topology="ring"):
    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights, dist_weights):
        if inputs.ndim == 1:
            h_i0, h_i1, h_j0, h_j1, d_ij = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
        else:
            h_i0, h_i1, h_j0, h_j1, d_ij = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3], inputs[:, 4]

        n_layers = weights.shape[0]

        for l in range(n_layers):
            # Feature encoding
            if encoding_type == "multi_axis":
                qml.RY(h_i0, wires=0)
                qml.RY(h_i1, wires=1)
                qml.RY(h_j0, wires=2)
                qml.RY(h_j1, wires=3)
                scaled_d = d_ij * dist_weights[l]
                for q in range(n_qubits):
                    qml.RZ(scaled_d, wires=q)
            elif encoding_type == "angle_sum":
                sum_feat = h_i0 + h_i1 + h_j0 + h_j1 + d_ij
                for q in range(n_qubits):
                    qml.RY(sum_feat, wires=q)

            # Single qubit rotations
            for q in range(n_qubits):
                qml.RX(weights[l, q, 0], wires=q)
                qml.RY(weights[l, q, 1], wires=q)
                qml.RZ(weights[l, q, 2], wires=q)

            # Entanglement topology
            if topology == "ring":
                for q in range(n_qubits):
                    qml.CNOT(wires=[q, (q + 1) % n_qubits])
            elif topology == "linear":
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
            elif topology == "all_to_all":
                for q1 in range(n_qubits):
                    for q2 in range(q1 + 1, n_qubits):
                        qml.CNOT(wires=[q1, q2])
            elif topology == "separable":
                pass  # No entanglement

        return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

    return circuit


class AblationEQGNNLayer(nn.Module):
    def __init__(self, in_features, n_q_layers=2, encoding_type="multi_axis", topology="ring", dynamic_coords=True):
        super().__init__()
        self.in_features = in_features
        self.dynamic_coords = dynamic_coords
        self.feature_proj = nn.Linear(in_features, 2)

        qnode = create_ablation_qnode(encoding_type=encoding_type, topology=topology)
        weight_shapes = {
            "weights": (n_q_layers, n_qubits, 3),
            "dist_weights": (n_q_layers,)
        }
        self.q_edge_layer = qml.qnn.TorchLayer(qnode, weight_shapes)
        with torch.no_grad():
            self.q_edge_layer.dist_weights.uniform_(0.05, 0.2)

        self.node_mlp = nn.Sequential(
            nn.Linear(in_features + (n_qubits - 1), in_features),
            nn.SiLU(),
            nn.Linear(in_features, in_features)
        )

    def forward(self, h, x, edge_index):
        row, col = edge_index[0], edge_index[1]
        rel_x = x[row] - x[col]
        d_ij = torch.sum(rel_x**2, dim=-1, keepdim=True)

        h_proj = self.feature_proj(h)
        h_i_p = h_proj[row]
        h_j_p = h_proj[col]
        q_inputs = torch.cat([h_i_p, h_j_p, d_ij], dim=-1)

        q_out = self.q_edge_layer(q_inputs)
        m_ij = q_out[:, :3]
        phi_x = q_out[:, 3:4]

        # Coordinate Update
        if self.dynamic_coords:
            coord_msg = rel_x * phi_x
            x_update = torch.zeros_like(x)
            x_update.index_add_(0, row, coord_msg)
            degrees = torch.zeros((x.size(0), 1), device=x.device)
            degrees.index_add_(0, row, torch.ones((row.size(0), 1), device=x.device))
            x_new = x + x_update / degrees.clamp(min=1.0)
        else:
            x_new = x  # Static coordinates

        m_i = torch.zeros((h.size(0), m_ij.size(1)), device=h.device)
        m_i.index_add_(0, row, m_ij)
        h_new = h + self.node_mlp(torch.cat([h, m_i], dim=-1))

        return h_new, x_new


class AblationEQGNNModel(nn.Module):
    def __init__(self, in_features=11, hidden_features=16, num_layers=2, n_q_layers=2,
                 encoding_type="multi_axis", topology="ring", dynamic_coords=True):
        super().__init__()
        self.embedding = nn.Linear(in_features, hidden_features)
        self.layers = nn.ModuleList([
            AblationEQGNNLayer(
                hidden_features,
                n_q_layers=n_q_layers,
                encoding_type=encoding_type,
                topology=topology,
                dynamic_coords=dynamic_coords
            ) for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, 1)
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


def train_eval_model(model, train_loader, val_loader, test_loader, mean, std, epochs=12, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            y_norm = (batch.y - mean) / std
            pred = model(batch.x, batch.pos, batch.edge_index, batch.batch)
            loss = criterion(pred, y_norm)
            loss.backward()
            optimizer.step()

    # Evaluate on test set
    model.eval()
    total_mae = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in test_loader:
            pred_norm = model(batch.x, batch.pos, batch.edge_index, batch.batch)
            pred = pred_norm * std + mean
            total_mae += torch.abs(pred - batch.y).sum().item()
            total_samples += batch.num_graphs

    return total_mae / total_samples


def run_ablation_suite(save_dir="./results"):
    os.makedirs(save_dir, exist_ok=True)
    print("Executing quantum architecture ablation suite (V1)")

    train_set, val_set, test_set, (mean, std) = get_qm9_splits(
        root="./data/QM9", target="gap", num_train=250, num_val=100, num_test=250, seed=42
    )
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

    ablation_results = {}

    # Feature encoding ablations
    print("\n[1/4] Feature Encoding Ablation...")
    model_ours = AblationEQGNNModel(encoding_type="multi_axis", topology="ring", dynamic_coords=True)
    mae_ours = train_eval_model(model_ours, train_loader, val_loader, test_loader, mean, std)
    print(f"  Multi-Axis Re-Uploading (Ours): Test MAE = {mae_ours:.4f} eV")

    model_sum = AblationEQGNNModel(encoding_type="angle_sum", topology="ring", dynamic_coords=True)
    mae_sum = train_eval_model(model_sum, train_loader, val_loader, test_loader, mean, std)
    print(f"  Destructive Summation (Base Eq. 6): Test MAE = {mae_sum:.4f} eV")
    ablation_results["encoding"] = {"Multi-Axis (Ours)": mae_ours, "Angle Sum (Eq. 6)": mae_sum}

    # Entanglement topology ablations
    print("\n[2/4] Entanglement Topology Ablation...")
    model_linear = AblationEQGNNModel(encoding_type="multi_axis", topology="linear", dynamic_coords=True)
    mae_linear = train_eval_model(model_linear, train_loader, val_loader, test_loader, mean, std)
    print(f"  Linear CNOT: Test MAE = {mae_linear:.4f} eV")

    model_all = AblationEQGNNModel(encoding_type="multi_axis", topology="all_to_all", dynamic_coords=True)
    mae_all = train_eval_model(model_all, train_loader, val_loader, test_loader, mean, std)
    print(f"  All-to-All CNOT: Test MAE = {mae_all:.4f} eV")

    model_sep = AblationEQGNNModel(encoding_type="multi_axis", topology="separable", dynamic_coords=True)
    mae_sep = train_eval_model(model_sep, train_loader, val_loader, test_loader, mean, std)
    print(f"  Separable (No CNOT): Test MAE = {mae_sep:.4f} eV")
    ablation_results["topology"] = {
        "Ring CNOT (Ours)": mae_ours,
        "Linear CNOT": mae_linear,
        "All-to-All CNOT": mae_all,
        "Separable (None)": mae_sep
    }

    # Coordinate dynamics ablation
    print("\n[3/4] Coordinate Dynamics Ablation...")
    model_static = AblationEQGNNModel(encoding_type="multi_axis", topology="ring", dynamic_coords=False)
    mae_static = train_eval_model(model_static, train_loader, val_loader, test_loader, mean, std)
    print(f"  Static Coordinates (No update): Test MAE = {mae_static:.4f} eV")
    ablation_results["coordinates"] = {"Dynamic Equivariant (Ours)": mae_ours, "Static Geometry": mae_static}

    # Quantum circuit depth ablation
    print("\n[4/4] Quantum Circuit Depth (Layers L) Ablation...")
    model_l1 = AblationEQGNNModel(n_q_layers=1)
    mae_l1 = train_eval_model(model_l1, train_loader, val_loader, test_loader, mean, std)
    print(f"  L = 1 Quantum Layer: Test MAE = {mae_l1:.4f} eV")

    model_l4 = AblationEQGNNModel(n_q_layers=4)
    mae_l4 = train_eval_model(model_l4, train_loader, val_loader, test_loader, mean, std)
    print(f"  L = 4 Quantum Layers: Test MAE = {mae_l4:.4f} eV")
    ablation_results["depth"] = {"L = 1": mae_l1, "L = 2 (Ours)": mae_ours, "L = 4": mae_l4}

    # Save JSON summary
    with open(os.path.join(save_dir, "ablation_study.json"), "w") as f:
        json.dump(ablation_results, f, indent=2)

    # Generate Figure 3
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), dpi=300)

    # Plot (a): Feature Encoding
    ax = axes[0, 0]
    names = list(ablation_results["encoding"].keys())
    vals = list(ablation_results["encoding"].values())
    bars = ax.bar(names, vals, color=["#1f77b4", "#d62728"], width=0.5)
    ax.set_ylabel("Test MAE (eV)", fontweight="bold")
    ax.set_title("(a) Feature Encoding Mechanism", fontweight="bold")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{bar.get_height():.3f}", ha="center", fontweight="bold")

    # Plot (b): Entanglement Topology
    ax = axes[0, 1]
    names = list(ablation_results["topology"].keys())
    vals = list(ablation_results["topology"].values())
    bars = ax.bar(names, vals, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#7f7f7f"], width=0.55)
    ax.set_ylabel("Test MAE (eV)", fontweight="bold")
    ax.set_title("(b) Entanglement Topology", fontweight="bold")
    ax.tick_params(axis='x', rotation=15)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{bar.get_height():.3f}", ha="center", fontweight="bold")

    # Plot (c): Coordinate Dynamics
    ax = axes[1, 0]
    names = list(ablation_results["coordinates"].keys())
    vals = list(ablation_results["coordinates"].values())
    bars = ax.bar(names, vals, color=["#1f77b4", "#9467bd"], width=0.5)
    ax.set_ylabel("Test MAE (eV)", fontweight="bold")
    ax.set_title("(c) Coordinate Refinement Dynamics", fontweight="bold")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{bar.get_height():.3f}", ha="center", fontweight="bold")

    # Plot (d): Quantum Depth
    ax = axes[1, 1]
    names = list(ablation_results["depth"].keys())
    vals = list(ablation_results["depth"].values())
    bars = ax.bar(names, vals, color=["#aec7e8", "#1f77b4", "#08519c"], width=0.5)
    ax.set_ylabel("Test MAE (eV)", fontweight="bold")
    ax.set_title("(d) Variational PQC Circuit Depth", fontweight="bold")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{bar.get_height():.3f}", ha="center", fontweight="bold")

    plt.tight_layout()
    fig_path = os.path.join(save_dir, "figure3_ablations.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"\nSaved Figure 3 to: {fig_path}")
    print("Experiment 4 completed successfully.")


class AblationEQGNNLayerV2(nn.Module):
    """Ablation layer with toggleable architectural components."""
    def __init__(self, in_features, n_q_layers=2, n_rbf=N_RBF,
                 use_rbf=True, n_heads=2, use_correlators=True,
                 use_angles=True, use_tanh_gate=True):
        super().__init__()
        self.in_features = in_features
        self.use_rbf = use_rbf
        self.n_heads = n_heads
        self.use_correlators = use_correlators
        self.use_angles = use_angles
        self.use_tanh_gate = use_tanh_gate

        # RBF or scalar distance
        if use_rbf:
            self.rbf = GaussianRBF(n_rbf=n_rbf)
        else:
            self.rbf = None
            n_rbf = 1  # single scalar distance

        # Quantum heads
        self.heads = nn.ModuleList()
        for _ in range(n_heads):
            head = QuantumHead(in_features, n_q_layers=n_q_layers, n_rbf=n_rbf)
            self.heads.append(head)

        # Per-head output parsing depends on use_correlators
        if use_correlators:
            msg_per_head = 7  # 3 single-qubit + 4 correlators
        else:
            msg_per_head = 3  # 3 single-qubit only
        msg_dim = msg_per_head * n_heads

        # Coord gate
        if use_tanh_gate:
            self.coord_gate = nn.Linear(n_heads, 1)
        else:
            self.coord_gate = None

        self.node_mlp = nn.Sequential(
            nn.Linear(in_features + msg_dim, in_features),
            nn.SiLU(),
            nn.Linear(in_features, in_features)
        )

    def forward(self, h, x, edge_index):
        row, col = edge_index[0], edge_index[1]
        rel_x = x[row] - x[col]
        d_ij = torch.sum(rel_x ** 2, dim=-1, keepdim=True)

        # Distance features
        if self.use_rbf:
            rbf_features = self.rbf(d_ij)
        else:
            rbf_features = d_ij  # [E, 1] raw scalar

        # Angle features
        if self.use_angles:
            angles = compute_edge_mean_angles(x, edge_index)
        else:
            angles = torch.zeros(d_ij.size(0), 1, device=x.device)

        # Multi-head evaluation
        msgs = []
        coord_scalars = []
        for head in self.heads:
            out = head(h[row], h[col], rbf_features, angles)
            if self.use_correlators:
                msg = torch.cat([out[:, :3], out[:, 4:]], dim=-1)  # [E, 7]
            else:
                msg = out[:, :3]  # [E, 3]
            msgs.append(msg)
            coord_scalars.append(out[:, 3:4])

        m_ij = torch.cat(msgs, dim=-1)
        phi_x_raw = torch.cat(coord_scalars, dim=-1)

        # Coordinate update
        if self.use_tanh_gate and self.coord_gate is not None:
            phi_x = torch.tanh(self.coord_gate(phi_x_raw))
        else:
            phi_x = phi_x_raw.mean(dim=-1, keepdim=True)

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


class AblationEQGNNModelV2(nn.Module):
    """V2 Ablation Model with toggleable upgrades."""
    def __init__(self, in_features=11, hidden_features=16, num_layers=2, n_q_layers=2,
                 use_rbf=True, n_heads=2, use_correlators=True,
                 use_angles=True, use_tanh_gate=True):
        super().__init__()
        self.embedding = nn.Linear(in_features, hidden_features)
        self.layers = nn.ModuleList([
            AblationEQGNNLayerV2(
                hidden_features, n_q_layers=n_q_layers,
                use_rbf=use_rbf, n_heads=n_heads,
                use_correlators=use_correlators,
                use_angles=use_angles, use_tanh_gate=use_tanh_gate
            ) for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, 1)
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


def run_v2_ablation_suite(save_dir="./results"):
    """Run V2 ablation study benchmarking individual component contributions."""
    os.makedirs(save_dir, exist_ok=True)
    print("Executing V2 architecture ablation suite")

    train_set, val_set, test_set, (mean, std) = get_qm9_splits(
        root="./data/QM9", target="gap", num_train=250, num_val=100, num_test=250, seed=42
    )
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

    v2_ablation_results = {}

    # Full V2 baseline
    print("\n[0/4] Full V2 (All Components)...")
    model_full = AblationEQGNNModelV2(
        use_rbf=True, n_heads=2, use_correlators=True, use_angles=True, use_tanh_gate=True
    )
    params_full = sum(p.numel() for p in model_full.parameters() if p.requires_grad)
    mae_full = train_eval_model(model_full, train_loader, val_loader, test_loader, mean, std)
    print(f"  Full V2: Test MAE = {mae_full:.4f} eV ({params_full:,} params)")

    # RBF ablation
    print("\n[1/4] RBF Distance Encoding Ablation...")
    model_no_rbf = AblationEQGNNModelV2(
        use_rbf=False, n_heads=2, use_correlators=True, use_angles=True, use_tanh_gate=True
    )
    mae_no_rbf = train_eval_model(model_no_rbf, train_loader, val_loader, test_loader, mean, std)
    print(f"  Without RBF (scalar dist): Test MAE = {mae_no_rbf:.4f} eV")
    v2_ablation_results["rbf_encoding"] = {
        "Full V2 (RBF)": mae_full,
        "Without RBF (Scalar)": mae_no_rbf
    }

    # Multi-head ablation
    print("\n[2/4] Multi-Head Ablation...")
    model_1head = AblationEQGNNModelV2(
        use_rbf=True, n_heads=1, use_correlators=True, use_angles=True, use_tanh_gate=True
    )
    mae_1head = train_eval_model(model_1head, train_loader, val_loader, test_loader, mean, std)
    print(f"  Single Head: Test MAE = {mae_1head:.4f} eV")
    v2_ablation_results["multi_head"] = {
        "Full V2 (2 Heads)": mae_full,
        "Single Head": mae_1head
    }

    # Correlator ablation
    print("\n[3/4] Entanglement Correlator Ablation...")
    model_no_corr = AblationEQGNNModelV2(
        use_rbf=True, n_heads=2, use_correlators=False, use_angles=True, use_tanh_gate=True
    )
    mae_no_corr = train_eval_model(model_no_corr, train_loader, val_loader, test_loader, mean, std)
    print(f"  Without Correlators (⟨Z⟩ only): Test MAE = {mae_no_corr:.4f} eV")
    v2_ablation_results["correlators"] = {
        "Full V2 (⟨ZZ⟩ Correlators)": mae_full,
        "Without Correlators (⟨Z⟩ only)": mae_no_corr
    }

    # Depth ablation
    print("\n[4/4] V2 Circuit Depth Ablation...")
    model_v2_l1 = AblationEQGNNModelV2(n_q_layers=1)
    mae_v2_l1 = train_eval_model(model_v2_l1, train_loader, val_loader, test_loader, mean, std)
    print(f"  V2 L=1: Test MAE = {mae_v2_l1:.4f} eV")

    model_v2_l4 = AblationEQGNNModelV2(n_q_layers=4)
    mae_v2_l4 = train_eval_model(model_v2_l4, train_loader, val_loader, test_loader, mean, std)
    print(f"  V2 L=4: Test MAE = {mae_v2_l4:.4f} eV")
    v2_ablation_results["v2_depth"] = {
        "V2 L=1": mae_v2_l1,
        "V2 L=2 (Default)": mae_full,
        "V2 L=4": mae_v2_l4
    }

    # Save JSON
    with open(os.path.join(save_dir, "ablation_study_v2.json"), "w") as f:
        json.dump(v2_ablation_results, f, indent=2)

    plot_v2_ablations(v2_ablation_results, fig_path=os.path.join(save_dir, "figure3b_ablations_v2.png"))

def plot_v2_ablations(v2_ablation_results, fig_path="./results/figure3b_ablations_v2.png"):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), dpi=300)

    ax = axes[0, 0]
    names = list(v2_ablation_results["rbf_encoding"].keys())
    vals = list(v2_ablation_results["rbf_encoding"].values())
    bars = ax.bar(names, vals, color=["#1f77b4", "#d62728"], width=0.5)
    ax.set_ylabel("Test MAE (eV)", fontweight="bold")
    ax.set_title("(a) RBF Distance Encoding", fontweight="bold")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{bar.get_height():.3f}", ha="center", fontweight="bold")

    ax = axes[0, 1]
    names = list(v2_ablation_results["multi_head"].keys())
    vals = list(v2_ablation_results["multi_head"].values())
    bars = ax.bar(names, vals, color=["#1f77b4", "#ff7f0e"], width=0.5)
    ax.set_ylabel("Test MAE (eV)", fontweight="bold")
    ax.set_title("(b) Multi-Head Architecture", fontweight="bold")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{bar.get_height():.3f}", ha="center", fontweight="bold")

    ax = axes[1, 0]
    names = [r"Full Q-PRISM ($\langle ZZ \rangle$ Correlators)", r"Without Correlators ($\langle Z \rangle$ only)"]
    vals = list(v2_ablation_results["correlators"].values())
    bars = ax.bar(names, vals, color=["#1f77b4", "#9467bd"], width=0.5)
    ax.set_ylabel("Test MAE (eV)", fontweight="bold")
    ax.set_title("(c) Entanglement Correlators", fontweight="bold")
    ax.tick_params(axis='x', rotation=5)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{bar.get_height():.3f}", ha="center", fontweight="bold")

    ax = axes[1, 1]
    names = list(v2_ablation_results["v2_depth"].keys())
    vals = list(v2_ablation_results["v2_depth"].values())
    bars = ax.bar(names, vals, color=["#aec7e8", "#1f77b4", "#08519c"], width=0.5)
    ax.set_ylabel("Test MAE (eV)", fontweight="bold")
    ax.set_title("(d) Variational Circuit Depth", fontweight="bold")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{bar.get_height():.3f}", ha="center", fontweight="bold")

    plt.suptitle("Q-PRISM Architecture Ablation Study (HOMO-LUMO Gap, QM9)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path)
    plt.close()
    print(f"\nSaved Figure 3b to: {fig_path}")
    print("V2 ablation plot completed successfully.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quantum Architecture Ablation Suite")
    parser.add_argument("--version", type=str, default="both", choices=["v1", "v2", "both"],
                        help="Which ablation suite to run: v1 (original), v2 (new upgrades), or both")
    parser.add_argument("--plot-only", action="store_true", help="Re-render figures from saved JSON")
    args = parser.parse_args()

    if args.plot_only:
        with open("./results/ablation_study_v2.json", "r") as f:
            v2_results = json.load(f)
        plot_v2_ablations(v2_results, fig_path="./results/figure3b_ablations_v2.png")
    else:
        if args.version in ("v1", "both"):
            run_ablation_suite()
        if args.version in ("v2", "both"):
            run_v2_ablation_suite()
