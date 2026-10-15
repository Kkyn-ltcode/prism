"""Entanglement ablation benchmark comparing entangled and separable PQCs."""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from scipy import stats
from torch_geometric.loader import DataLoader

from datasets.dataset_qm9 import get_qm9_splits
from models.eq_gnn import (
    GaussianRBF, compute_edge_mean_angles, N_RBF, n_qubits,
    EQGNNModelV2
)
from models.egnn_baseline import count_parameters

# Separable V2 PQC (independent single-qubit evolutions)

def _create_separable_v2_qnode():
    """V2 QNode with CNOT gates removed — each qubit evolves independently."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def separable_circuit_v2(inputs, weights, dist_weights):
        K = dist_weights.shape[2]
        n_layers = weights.shape[0]

        if inputs.ndim == 1:
            h_i0, h_i1 = inputs[0], inputs[1]
            h_j0, h_j1 = inputs[2], inputs[3]
            rbf = inputs[4:4 + K]
            angle = inputs[4 + K]
        else:
            h_i0, h_i1 = inputs[:, 0], inputs[:, 1]
            h_j0, h_j1 = inputs[:, 2], inputs[:, 3]
            rbf = inputs[:, 4:4 + K]
            angle = inputs[:, 4 + K]

        for l in range(n_layers):
            # Multi-axis feature encoding
            qml.RY(h_i0, wires=0)
            qml.RY(h_i1, wires=1)
            qml.RY(h_j0, wires=2)
            qml.RY(h_j1, wires=3)

            # RBF distance re-uploading
            for q in range(n_qubits):
                if inputs.ndim == 1:
                    phase = sum(dist_weights[l, q, k] * rbf[k] for k in range(K))
                else:
                    phase = sum(dist_weights[l, q, k] * rbf[:, k] for k in range(K))
                qml.RZ(phase, wires=q)

            # Angular re-uploading
            for q in range(n_qubits):
                qml.RY(angle, wires=q)

            # Variational rotations
            for q in range(n_qubits):
                qml.RX(weights[l, q, 0], wires=q)
                qml.RY(weights[l, q, 1], wires=q)
                qml.RZ(weights[l, q, 2], wires=q)

            # Separable: no CNOT entanglement

        # 8-observable readout
        return [
            qml.expval(qml.PauliZ(0)),
            qml.expval(qml.PauliZ(1)),
            qml.expval(qml.PauliZ(2)),
            qml.expval(qml.PauliZ(3)),
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),
            qml.expval(qml.PauliZ(1) @ qml.PauliZ(2)),
            qml.expval(qml.PauliZ(2) @ qml.PauliZ(3)),
            qml.expval(qml.PauliZ(3) @ qml.PauliZ(0)),
        ]

    return separable_circuit_v2


class SeparableQuantumHead(nn.Module):
    """Quantum head with NO entanglement — identical to QuantumHead but uses separable PQC."""
    def __init__(self, in_features, n_q_layers=2, n_rbf=N_RBF):
        super().__init__()
        self.feature_proj = nn.Linear(in_features, 2)
        qnode = _create_separable_v2_qnode()
        weight_shapes = {
            "weights": (n_q_layers, n_qubits, 3),
            "dist_weights": (n_q_layers, n_qubits, n_rbf),
        }
        self.q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)
        with torch.no_grad():
            self.q_layer.dist_weights.uniform_(0.05, 0.2)

    def forward(self, h_row, h_col, rbf_features, angles):
        h_i_proj = self.feature_proj(h_row)
        h_j_proj = self.feature_proj(h_col)
        q_inputs = torch.cat([h_i_proj, h_j_proj, rbf_features, angles], dim=-1)
        return self.q_layer(q_inputs)


class SeparableQPRISMLayerV2(nn.Module):
    """V2 layer with separable (unentangled) quantum heads."""
    def __init__(self, in_features, n_q_layers=2, n_rbf=N_RBF):
        super().__init__()
        self.rbf = GaussianRBF(n_rbf=n_rbf)
        self.head_electronic = SeparableQuantumHead(in_features, n_q_layers=n_q_layers, n_rbf=n_rbf)
        self.head_topological = SeparableQuantumHead(in_features, n_q_layers=n_q_layers, n_rbf=n_rbf)
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


SeparableEQGNNLayerV2 = SeparableQPRISMLayerV2


class SeparableQPRISMModelV2(nn.Module):
    """Full Q-PRISM model with entanglement removed from all PQC heads."""
    def __init__(self, in_features=11, hidden_features=16, num_layers=2, out_dim=1,
                 n_q_layers=2, n_rbf=N_RBF):
        super().__init__()
        self.embedding = nn.Linear(in_features, hidden_features)
        self.layers = nn.ModuleList([
            SeparableQPRISMLayerV2(hidden_features, n_q_layers=n_q_layers, n_rbf=n_rbf)
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


SeparableEQGNNModelV2 = SeparableQPRISMModelV2  # Backward compatibility alias
SeparableQPRISMModel = SeparableQPRISMModelV2
SeparableEQGNNModel = SeparableQPRISMModelV2


# Classical tanh-MLP baseline (identical inputs, tanh output, matching parameters)

class ClassicalTanhHead(nn.Module):
    """Classical MLP replacement for PQC: same inputs, same output dim, tanh bounds."""
    def __init__(self, in_features=16, n_rbf=8, hidden_dim=4):
        super().__init__()
        self.feature_proj = nn.Linear(in_features, 2)
        # Input: [h_i_proj(2), h_j_proj(2), rbf(8), angle(1)] = 13
        # Linear(13,4) + SiLU + Linear(4,8,no bias) + Tanh = 88 params (matches PQC's 88)
        self.mlp = nn.Sequential(
            nn.Linear(13, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 8, bias=False),
            nn.Tanh()
        )

    def forward(self, h_row, h_col, rbf_features, angles):
        h_i_p = self.feature_proj(h_row)
        h_j_p = self.feature_proj(h_col)
        inputs = torch.cat([h_i_p, h_j_p, rbf_features, angles], dim=-1)
        return self.mlp(inputs)


class ClassicalTanhQPRISMLayer(nn.Module):
    """Q-PRISM layer with PQC replaced by classical tanh-MLP."""
    def __init__(self, in_features=16, n_rbf=N_RBF):
        super().__init__()
        self.rbf = GaussianRBF(n_rbf=n_rbf)
        self.head_electronic = ClassicalTanhHead(in_features, n_rbf=n_rbf)
        self.head_topological = ClassicalTanhHead(in_features, n_rbf=n_rbf)
        self.coord_gate = nn.Linear(2, 1)
        self.node_mlp = nn.Sequential(
            nn.Linear(in_features + 14, in_features),
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


ClassicalTanhEQGNNLayer = ClassicalTanhQPRISMLayer


class ClassicalTanhQPRISMModel(nn.Module):
    """Full model with classical tanh-MLP replacing all PQC heads."""
    def __init__(self, in_features=11, hidden_features=16, num_layers=2, out_dim=1):
        super().__init__()
        self.embedding = nn.Linear(in_features, hidden_features)
        self.layers = nn.ModuleList([
            ClassicalTanhQPRISMLayer(hidden_features) for _ in range(num_layers)
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


ClassicalTanhEQGNNModel = ClassicalTanhQPRISMModel
ClassicalTanhEQGNNLayer = ClassicalTanhQPRISMLayer
ClassicalTanhQPRISMLayer = ClassicalTanhQPRISMLayer
ClassicalTanhQPRISMModel = ClassicalTanhQPRISMModel


# Training and evaluation

def train_and_evaluate(model, target, seed, num_train=1000, num_val=250, num_test=500,
                       epochs=15, batch_size=16, lr=1e-3):
    """Train a model and return test MAE."""
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

    return total_mae / total_samples


# Main benchmark

def main():
    targets = ["homo", "lumo", "gap"]
    seeds = [42, 43, 44, 45, 46]
    results = {}

    # Verify parameter counts
    print("=" * 70)
    print("Entanglement ablation benchmark (Bowles test)")
    print("=" * 70)

    m_entangled = EQGNNModelV2(in_features=11, hidden_features=16, num_layers=2, n_q_layers=2, out_dim=1)
    m_separable = SeparableEQGNNModelV2(in_features=11, hidden_features=16, num_layers=2, n_q_layers=2, out_dim=1)
    m_tanh = ClassicalTanhEQGNNModel(in_features=11, hidden_features=16, num_layers=2, out_dim=1)

    p_ent = count_parameters(m_entangled)
    p_sep = count_parameters(m_separable)
    p_tanh = count_parameters(m_tanh)

    print(f"\nParameter counts:")
    print(f"  Q-PRISM (entangled):      {p_ent:,}")
    print(f"  Separable V2 (no CNOTs):  {p_sep:,}")
    print(f"  Classical tanh-MLP:       {p_tanh:,}")
    assert p_ent == p_sep, f"Parameter mismatch! entangled={p_ent} vs separable={p_sep}"
    assert p_ent == p_tanh, f"Parameter mismatch! entangled={p_ent} vs tanh={p_tanh}"
    print(f"  All models: {p_ent:,} parameters (exact match)\n")

    del m_entangled, m_separable, m_tanh

    # Load existing Q-PRISM results
    existing_results_path = PROJECT_ROOT / "results" / "multitarget_table1_v2.json"
    existing_data = {}
    if existing_results_path.exists():
        with open(existing_results_path) as f:
            existing_data = json.load(f)
        print("Loaded existing Q-PRISM and EGNN results from multitarget_table1_v2.json\n")

    # --- Run new models: Separable V2 + Classical tanh-MLP ---
    for target in targets:
        print(f"\n{'='*60}")
        print(f"TARGET: {target.upper()}")
        print(f"{'='*60}")

        results[target] = {}

        # Pull existing results
        if target in existing_data:
            results[target]["eqgnn_v2"] = existing_data[target]["eqgnn_v2"]["maes"]
            results[target]["egnn_pm"] = existing_data[target]["egnn_pm"]["maes"]
            print(f"  Q-PRISM (from file):     {existing_data[target]['eqgnn_v2']['maes']}")
            print(f"  EGNN PM (from file):     {existing_data[target]['egnn_pm']['maes']}")

        # --- Separable V2 ---
        print(f"\n  Running: Separable V2 (no entanglement)...")
        sep_maes = []
        for seed in seeds:
            t0 = time.time()
            model = SeparableEQGNNModelV2(
                in_features=11, hidden_features=16, num_layers=2, n_q_layers=2, out_dim=1
            )
            mae = train_and_evaluate(model, target, seed)
            elapsed = time.time() - t0
            sep_maes.append(mae)
            print(f"    Seed {seed}: {mae:.4f} ({elapsed:.1f}s)")
        results[target]["separable_v2"] = sep_maes
        print(f"  >> Separable V2 Mean: {np.mean(sep_maes):.4f} ± {np.std(sep_maes):.4f}")

        # --- Classical tanh-MLP ---
        print(f"\n  Running: Classical tanh-MLP...")
        tanh_maes = []
        for seed in seeds:
            t0 = time.time()
            model = ClassicalTanhEQGNNModel(
                in_features=11, hidden_features=16, num_layers=2, out_dim=1
            )
            mae = train_and_evaluate(model, target, seed)
            elapsed = time.time() - t0
            tanh_maes.append(mae)
            print(f"    Seed {seed}: {mae:.4f} ({elapsed:.1f}s)")
        results[target]["tanh_mlp"] = tanh_maes
        print(f"  >> Classical tanh Mean: {np.mean(tanh_maes):.4f} ± {np.std(tanh_maes):.4f}")

    # Statistical analysis
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)

    all_results = {}

    for target in targets:
        eq_maes = np.array(results[target].get("eqgnn_v2", []))
        sep_maes = np.array(results[target]["separable_v2"])
        tanh_maes = np.array(results[target]["tanh_mlp"])
        pm_maes = np.array(results[target].get("egnn_pm", []))

        print(f"\n--- {target.upper()} ---")
        print(f"  EQ-GNN V2 (entangled):    {np.mean(eq_maes):.4f} ± {np.std(eq_maes):.4f}")
        print(f"  Separable V2 (no CNOTs):  {np.mean(sep_maes):.4f} ± {np.std(sep_maes):.4f}")
        print(f"  Classical tanh-MLP:       {np.mean(tanh_maes):.4f} ± {np.std(tanh_maes):.4f}")
        if len(pm_maes) > 0:
            print(f"  EGNN Param-Matched:       {np.mean(pm_maes):.4f} ± {np.std(pm_maes):.4f}")

        target_stats = {
            "eqgnn_v2": {"maes": eq_maes.tolist(), "mean": float(np.mean(eq_maes)), "std": float(np.std(eq_maes))},
            "separable_v2": {"maes": sep_maes.tolist(), "mean": float(np.mean(sep_maes)), "std": float(np.std(sep_maes))},
            "tanh_mlp": {"maes": tanh_maes.tolist(), "mean": float(np.mean(tanh_maes)), "std": float(np.std(tanh_maes))},
        }

        # Test: Entangled vs Separable
        if len(eq_maes) == 5:
            t_stat, t_p = stats.ttest_rel(eq_maes, sep_maes)
            diff = eq_maes - sep_maes
            print(f"\n  Entangled vs Separable:")
            print(f"    Mean diff: {np.mean(diff):+.4f} eV")
            print(f"    Paired t-test p = {t_p:.4f} {'**' if t_p < 0.01 else '*' if t_p < 0.05 else '(ns)'}")
            print(f"    Per-seed: {['%.4f' % d for d in diff]}")
            target_stats["entangled_vs_separable"] = {"mean_diff": float(np.mean(diff)), "ttest_p": float(t_p)}

        # Test: Separable vs tanh (should be similar if entanglement is the key)
        t_stat2, t_p2 = stats.ttest_rel(sep_maes, tanh_maes)
        diff2 = sep_maes - tanh_maes
        print(f"\n  Separable vs Classical tanh:")
        print(f"    Mean diff: {np.mean(diff2):+.4f} eV")
        print(f"    Paired t-test p = {t_p2:.4f} {'**' if t_p2 < 0.01 else '*' if t_p2 < 0.05 else '(ns)'}")
        print(f"    Per-seed: {['%.4f' % d for d in diff2]}")
        target_stats["separable_vs_tanh"] = {"mean_diff": float(np.mean(diff2)), "ttest_p": float(t_p2)}

        # Test: Entangled vs tanh
        if len(eq_maes) == 5:
            t_stat3, t_p3 = stats.ttest_rel(eq_maes, tanh_maes)
            diff3 = eq_maes - tanh_maes
            print(f"\n  Entangled vs Classical tanh:")
            print(f"    Mean diff: {np.mean(diff3):+.4f} eV")
            print(f"    Paired t-test p = {t_p3:.4f} {'**' if t_p3 < 0.01 else '*' if t_p3 < 0.05 else '(ns)'}")
            target_stats["entangled_vs_tanh"] = {"mean_diff": float(np.mean(diff3)), "ttest_p": float(t_p3)}

        all_results[target] = target_stats

    # Save results
    save_path = PROJECT_ROOT / "results" / "entanglement_ablation_v2.json"
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {save_path}")

    # Bowles test verdict
    print("\n" + "=" * 70)
    print("BOWLES TEST VERDICT")
    print("=" * 70)
    for target in targets:
        eq_m = all_results[target]["eqgnn_v2"]["mean"]
        sep_m = all_results[target]["separable_v2"]["mean"]
        tanh_m = all_results[target]["tanh_mlp"]["mean"]
        ent_p = all_results[target].get("entangled_vs_separable", {}).get("ttest_p", 1.0)
        sep_tanh_p = all_results[target]["separable_vs_tanh"]["ttest_p"]

        entanglement_helps = eq_m < sep_m and ent_p < 0.05
        separable_eq_tanh = sep_tanh_p > 0.05  # not significantly different

        print(f"\n  {target.upper()}:")
        print(f"    Entangled ({eq_m:.4f}) < Separable ({sep_m:.4f})?  {'YES (p={:.4f})'.format(ent_p) if entanglement_helps else 'NO (p={:.4f})'.format(ent_p)}")
        print(f"    Separable ({sep_m:.4f}) ≈ Tanh ({tanh_m:.4f})?     {'YES (p={:.4f})'.format(sep_tanh_p) if separable_eq_tanh else 'NO (p={:.4f})'.format(sep_tanh_p)}")

        if entanglement_helps and separable_eq_tanh:
            print("    VERDICT: Entanglement is the source of quantum advantage.")
        elif entanglement_helps:
            print("    VERDICT: Entanglement helps, but separable PQC != tanh (partial quantum advantage).")
        else:
            print("    VERDICT: Entanglement effect not significant for this target.")

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
