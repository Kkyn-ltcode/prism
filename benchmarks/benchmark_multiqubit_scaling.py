"""Multi-qubit scaling benchmark measuring expressivity and parameter efficiency across qubit counts."""

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
from models.eq_gnn import GaussianRBF, compute_edge_mean_angles, N_RBF

def _create_qnode_nq(nq):
    dev = qml.device("default.qubit", wires=nq)
    
    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights, dist_weights):
        # inputs: [E, 2*feat_per_node + K + 1]
        #   first nq values = feature encoding (nq/2 per node for i and j)
        #   next K values = RBF distance features
        #   last value = angle
        # weights: [n_layers, nq, 3]
        # dist_weights: [n_layers, nq, K]
        
        K = dist_weights.shape[2]
        n_layers = weights.shape[0]
        feat_dim = nq  # nq/2 features per node × 2 nodes = nq features total
        
        # Parse inputs
        if inputs.ndim == 1:
            features = [inputs[f] for f in range(feat_dim)]
            rbf = inputs[feat_dim:feat_dim + K]
            angle = inputs[feat_dim + K]
        else:
            features = [inputs[:, f] for f in range(feat_dim)]
            rbf = inputs[:, feat_dim:feat_dim + K]
            angle = inputs[:, feat_dim + K]
        
        for l in range(n_layers):
            # Feature encoding: RY on each qubit
            for q in range(nq):
                qml.RY(features[q], wires=q)
            
            # RBF distance re-uploading
            for q in range(nq):
                if inputs.ndim == 1:
                    phase = sum(dist_weights[l, q, k] * rbf[k] for k in range(K))
                else:
                    phase = sum(dist_weights[l, q, k] * rbf[:, k] for k in range(K))
                qml.RZ(phase, wires=q)
            
            # Angular encoding
            for q in range(nq):
                qml.RY(angle, wires=q)
            
            # Variational rotations
            for q in range(nq):
                qml.RX(weights[l, q, 0], wires=q)
                qml.RY(weights[l, q, 1], wires=q)
                qml.RZ(weights[l, q, 2], wires=q)
            
            # Ring CNOT entanglement
            for q in range(nq):
                qml.CNOT(wires=[q, (q + 1) % nq])
        
        # Readout: single-qubit Z + adjacent correlators
        observables = []
        for q in range(nq):
            observables.append(qml.expval(qml.PauliZ(q)))
        for q in range(nq):
            observables.append(qml.expval(qml.PauliZ(q) @ qml.PauliZ((q + 1) % nq)))
        return observables
    
    return circuit

class QuantumHeadNQ(nn.Module):
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
        h_i_proj = self.feature_proj(h_row)  # [E, nq // 2]
        h_j_proj = self.feature_proj(h_col)  # [E, nq // 2]
        q_inputs = torch.cat([h_i_proj, h_j_proj, rbf_features, angles], dim=-1)
        return self.q_layer(q_inputs)  # [E, 2*nq]

class QPRISMLayerNQ(nn.Module):
    def __init__(self, in_features, nq=4, n_q_layers=2, n_rbf=N_RBF):
        super().__init__()
        self.in_features = in_features
        self.nq = nq
        self.n_rbf = n_rbf
        self.rbf = GaussianRBF(n_rbf=n_rbf)
        self.head_electronic = QuantumHeadNQ(in_features, nq=nq, n_q_layers=n_q_layers, n_rbf=n_rbf)
        self.head_topological = QuantumHeadNQ(in_features, nq=nq, n_q_layers=n_q_layers, n_rbf=n_rbf)
        self.coord_gate = nn.Linear(2, 1)
        
        # 2*nq observables per head: (2*nq - 1) message + 1 coord
        msg_per_head = 2 * nq - 1
        msg_dim = msg_per_head * 2
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
        
        coord_idx = self.nq - 1
        msg_e_parts = [out_e[:, :coord_idx], out_e[:, coord_idx + 1:]]
        msg_t_parts = [out_t[:, :coord_idx], out_t[:, coord_idx + 1:]]
        msg_e = torch.cat(msg_e_parts, dim=-1)
        msg_t = torch.cat(msg_t_parts, dim=-1)
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


EQGNNLayerNQ = QPRISMLayerNQ  # Backward compatibility alias


class QPRISMModelNQ(nn.Module):
    def __init__(self, in_features=11, hidden_features=16, num_layers=2, out_dim=1,
                 nq=4, n_q_layers=2, n_rbf=N_RBF):
        super().__init__()
        self.embedding = nn.Linear(in_features, hidden_features)
        self.layers = nn.ModuleList([
            QPRISMLayerNQ(hidden_features, nq=nq, n_q_layers=n_q_layers, n_rbf=n_rbf)
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


EQGNNModelNQ = QPRISMModelNQ  # Backward compatibility alias


def train_eval_model(model, train_loader, test_loader, mean, std, epochs=15, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            y_norm = (batch.y - mean) / std
            pred = model(batch.x, batch.pos, batch.edge_index, batch.batch)
            loss = criterion(pred, y_norm)
            loss.backward()
            optimizer.step()
    
    train_time = time.time() - start_time
    
    model.eval()
    total_mae = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in test_loader:
            pred_norm = model(batch.x, batch.pos, batch.edge_index, batch.batch)
            pred = pred_norm * std + mean
            total_mae += torch.abs(pred - batch.y).sum().item()
            total_samples += batch.num_graphs
            
    return total_mae / total_samples, train_time

def run_multiqubit_scaling(save_dir="./results"):
    os.makedirs(save_dir, exist_ok=True)
    print("Starting Multi-Qubit Scaling Projection...")

    train_set, val_set, test_set, (mean, std) = get_qm9_splits(
        root="./data/QM9", target="gap", num_train=250, num_val=100, num_test=250, seed=42
    )
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False)
    
    qubit_counts = [4, 6, 8]
    gap_maes = []
    param_counts = []
    observable_counts = []
    training_times_sec = []
    circuit_time_arm_ns = []
    circuit_time_ibm_ns = []
    coherence_ratio_arm = []
    coherence_ratio_ibm = []
    statevector_bytes = []
    
    T2E_arm = 380e3  # ns
    T2E_ibm = 50e3   # ns
    L = 2
    heads = 2
    
    for nq in qubit_counts:
        print(f"\nEvaluating {nq}-qubit architecture...")
        model = EQGNNModelNQ(nq=nq, n_q_layers=L, num_layers=2)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        mae, train_time = train_eval_model(model, train_loader, test_loader, mean, std, epochs=15, lr=1e-3)
        print(f"  • Test MAE = {mae:.4f} eV")
        print(f"  • Params = {params}")
        print(f"  • Train Time = {train_time:.1f} s")
        
        gap_maes.append(mae)
        param_counts.append(params)
        training_times_sec.append(train_time)
        
        obs_per_eval = 2 * nq
        observable_counts.append(obs_per_eval)
        
        single_gates = 2 * L * 6 * nq
        two_qubit_gates = 2 * L * nq
        
        arm_time = single_gates * 5 + two_qubit_gates * 17
        ibm_time = single_gates * 25 + two_qubit_gates * 300
        
        circuit_time_arm_ns.append(arm_time)
        circuit_time_ibm_ns.append(ibm_time)
        
        coherence_ratio_arm.append(arm_time / T2E_arm)
        coherence_ratio_ibm.append(ibm_time / T2E_ibm)
        
        sv_size = (2 ** nq) * 16
        statevector_bytes.append(sv_size)

    results_data = {
        "qubit_counts": qubit_counts,
        "gap_maes": gap_maes,
        "param_counts": param_counts,
        "observable_counts": observable_counts,
        "training_times_sec": training_times_sec,
        "circuit_time_arm_ns": circuit_time_arm_ns,
        "circuit_time_ibm_ns": circuit_time_ibm_ns,
        "coherence_ratio_arm": coherence_ratio_arm,
        "coherence_ratio_ibm": coherence_ratio_ibm,
        "statevector_bytes": statevector_bytes
    }
    
    json_path = os.path.join(save_dir, "multiqubit_scaling.json")
    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=2)
        
    print(f"\nSaved JSON to {json_path}")
    
    plot_multiqubit_scaling(results_data, save_dir=save_dir)

def plot_multiqubit_scaling(results_data, save_dir="./results"):
    qubit_counts = results_data["qubit_counts"]
    gap_maes = results_data["gap_maes"]
    param_counts = results_data["param_counts"]
    observable_counts = results_data["observable_counts"]
    circuit_time_arm_ns = results_data["circuit_time_arm_ns"]
    circuit_time_ibm_ns = results_data["circuit_time_ibm_ns"]
    coherence_ratio_arm = results_data["coherence_ratio_arm"]
    coherence_ratio_ibm = results_data["coherence_ratio_ibm"]

    # Plotting Figure 8
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    
    # (a) Gap MAE vs Qubit Count
    ax = axes[0]
    colors = ['#add8e6', '#4682b4', '#00008b']
    bars = ax.bar([str(q) for q in qubit_counts], gap_maes, color=colors)
    ax.set_ylabel("Test MAE (eV)", fontweight="bold")
    ax.set_xlabel("Qubit Count", fontweight="bold")
    ax.set_title("(a) HOMO-LUMO Gap MAE vs Qubit Count", fontweight="bold")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{bar.get_height():.3f}", ha="center", fontweight="bold")
        
    # (b) Model Complexity
    ax = axes[1]
    x = range(len(qubit_counts))
    width = 0.35
    ax.bar([i - width/2 for i in x], param_counts, width, label='Trainable Parameters', color='#1f77b4')
    ax.set_ylabel("Parameter Count", fontweight="bold", color='#1f77b4')
    ax.tick_params(axis='y', labelcolor='#1f77b4')
    
    ax2 = ax.twinx()
    ax2.bar([i + width/2 for i in x], observable_counts, width, label='Observable Count', color='#ff7f0e')
    ax2.set_ylabel("Observable Count per PQC", fontweight="bold", color='#ff7f0e')
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')
    
    ax.set_xticks(x)
    ax.set_xticklabels([str(q) for q in qubit_counts])
    ax.set_xlabel("Qubit Count", fontweight="bold")
    ax.set_title("(b) Model Capacity & Observable Scaling", fontweight="bold")
    
    # (c) Hardware Resource Projection
    ax = axes[2]
    width2 = 0.35
    ax.bar([i - width2/2 for i in x], circuit_time_arm_ns, width2, label='Arm Qubit Time (ns)', color='#2ca02c')
    ax.bar([i + width2/2 for i in x], circuit_time_ibm_ns, width2, label='IBM Transmon Time (ns)', color='#d62728')
    ax.set_ylabel("Circuit Time (ns)", fontweight="bold")
    ax.set_yscale('log')
    ax.legend(loc='upper left')
    
    ax3 = ax.twinx()
    ax3.plot(x, coherence_ratio_arm, marker='o', color='#2ca02c', linestyle='-', label=r'$\tau/T_2^E$ Arm')
    ax3.plot(x, coherence_ratio_ibm, marker='s', color='#d62728', linestyle='-', label=r'$\tau/T_2^E$ IBM')
    ax3.axhline(0.01, color='gray', linestyle='--', label='1% Budget')
    ax3.set_ylabel("Coherence Ratio", fontweight="bold")
    ax3.set_yscale('log')
    ax3.legend(loc='upper right')
    
    ax.set_xticks(x)
    ax.set_xticklabels([str(q) for q in qubit_counts])
    ax.set_xlabel("Qubit Count", fontweight="bold")
    ax.set_title("(c) Projected Hardware Requirements", fontweight="bold")
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig_path = os.path.join(save_dir, "figure8_multiqubit_scaling.png")
    plt.savefig(fig_path)
    plt.close()
    
    print(f"Saved Figure to {fig_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true", help="Re-generate plots from existing JSON results")
    args = parser.parse_args()
    
    if args.plot_only:
        with open("./results/multiqubit_scaling.json", "r") as f:
            data = json.load(f)
        plot_multiqubit_scaling(data)
        print("Plot re-generated successfully.")
    else:
        run_multiqubit_scaling()
