import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import sys
from pathlib import Path
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from torch_geometric.loader import DataLoader

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dataset_qm9 import get_qm9_splits, QM9_TARGET_INDICES
from models.eq_gnn import GaussianRBF, compute_edge_mean_angles, N_RBF
from models.classical_baselines import PaiNNModel
from models.egnn_baseline import count_parameters
from training.run_extended_baselines_sweep import TARGET_UNITS, TARGET_DISPLAY, resolve_data_dir

# ---------------------------------------------------------
# 1. Scalable Heads, Layers, and Models
# ---------------------------------------------------------

class ScalablePRISMHead(nn.Module):
    def __init__(self, in_features, head_hidden, n_rbf=N_RBF, omega_0=30.0):
        super().__init__()
        self.feature_proj = nn.Linear(in_features, 2)
        self.omega_0 = omega_0

        input_dim = 4 + n_rbf + 1
        n_outputs = 8

        self.linear1 = nn.Linear(input_dim, head_hidden)
        # SIREN init
        with torch.no_grad():
            self.linear1.weight.uniform_(-1 / input_dim, 1 / input_dim)
        
        self.linear2 = nn.Linear(head_hidden, n_outputs, bias=False)

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
        self.in_features = in_features
        self.n_rbf = n_rbf
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
    def __init__(self, head_type="prism", in_features=11, hidden_features=16, head_hidden=4, num_layers=2, out_dim=1, n_rbf=N_RBF):
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

# ---------------------------------------------------------
# 2. Training Loop
# ---------------------------------------------------------

def create_model_for_scale(model_type, scale_dict):
    if model_type == "prism":
        model = ScalableModel(head_type="prism", 
                              hidden_features=scale_dict["hidden_features"],
                              head_hidden=scale_dict["head_hidden"])
    elif model_type == "twin":
        model = ScalableModel(head_type="classical", 
                              hidden_features=scale_dict["hidden_features"],
                              head_hidden=scale_dict["head_hidden"])
    elif model_type == "painn":
        model = PaiNNModel(in_features=11, hidden_channels=scale_dict["painn_h"])
    return model

def compute_statistical_tests(maes_a, maes_b):
    a = np.array(maes_a)
    b = np.array(maes_b)
    diff = a - b
    mean_diff = float(np.mean(diff))

    try:
        w_stat, w_p = stats.wilcoxon(a, b)
    except ValueError:
        w_p = 1.0
    t_stat, t_p = stats.ttest_rel(a, b)

    return {
        "mean_diff": mean_diff,
        "wilcoxon_p": float(w_p),
        "ttest_p": float(t_p),
        "significant_005": bool(w_p < 0.05 and t_p < 0.05),
        "significant_001": bool(w_p < 0.01 and t_p < 0.01),
    }

def train_eval_single_run(model_type, scale_dict, target, seed,
                          num_train=1000, num_val=250, num_test=500,
                          epochs=15, batch_size=16, lr=1e-3,
                          device="cpu", data_dir=None):
    resolved_data_dir = resolve_data_dir(data_dir)
    train_set, val_set, test_set, (mean, std) = get_qm9_splits(
        root=resolved_data_dir, target=target,
        num_train=num_train, num_val=num_val, num_test=num_test, seed=seed,
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    torch.manual_seed(seed)
    model = create_model_for_scale(model_type, scale_dict).to(device)
    total_params = count_parameters(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.L1Loss()

    model.train()
    for epoch in range(epochs):
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            # The features in QM9 data usually come as data.x which is size [N, 11]
            out = model(data.x.float(), data.pos.float(), data.edge_index, data.batch)
            
            loss = criterion(out.view(-1), data.y.view(-1))
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    maes = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data.x.float(), data.pos.float(), data.edge_index, data.batch)
            maes.append(torch.abs(out.view(-1) - data.y.view(-1)).cpu())
    
    test_mae = torch.cat(maes).mean().item()
    return test_mae, total_params

def main():
    parser = argparse.ArgumentParser(description="Revision C2: Parameter Scaling")
    parser.add_argument("--targets", nargs="+", default=["homo", "lumo", "gap"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--num-train", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    SCALES = {
        "Small": {"d_h": 4, "head_hidden": 4, "hidden_features": 16, "painn_h": 8},
        "Medium": {"d_h": 16, "head_hidden": 16, "hidden_features": 32, "painn_h": 32},
        "Large": {"d_h": 64, "head_hidden": 64, "hidden_features": 64, "painn_h": 128},
    }
    
    MODELS = ["prism", "twin", "painn"]
    
    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    json_path = out_dir / "revision_c2_param_scaling.json"
    
    results = {}

    for scale_name, scale_cfg in SCALES.items():
        results[scale_name] = {"config": scale_cfg}
        print(f"\n{'='*80}\nSCALE: {scale_name} (d_h={scale_cfg['d_h']})\n{'='*80}")
        
        for target in args.targets:
            if target not in results[scale_name]:
                results[scale_name][target] = {}
                
            print(f"\n--- Target: {TARGET_DISPLAY.get(target, target)} ---")
            for model_type in MODELS:
                print(f"Training {model_type}...")
                maes = []
                param_count = 0
                for seed in args.seeds:
                    mae, p_count = train_eval_single_run(
                        model_type, scale_cfg, target, seed,
                        num_train=args.num_train, epochs=args.epochs,
                        device=args.device, data_dir=args.data_dir
                    )
                    maes.append(mae)
                    param_count = p_count
                
                mean_mae = np.mean(maes)
                std_mae = np.std(maes)
                results[scale_name][target][model_type] = {
                    "mean_mae": float(mean_mae),
                    "std_mae": float(std_mae),
                    "maes": [float(m) for m in maes],
                    "params": param_count
                }
                print(f"  {model_type} -> MAE: {mean_mae:.4f} ± {std_mae:.4f} ({param_count:,} params)")

            # Stats PRISM vs Twin
            prism_maes = results[scale_name][target]["prism"]["maes"]
            twin_maes = results[scale_name][target]["twin"]["maes"]
            stats_res = compute_statistical_tests(prism_maes, twin_maes)
            results[scale_name][target]["stats"] = stats_res
            
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2)

    # Print Table
    print("\n\n" + "="*120)
    print("📈 PARAMETER SCALING RESULTS (Revision C2)")
    print("="*120)
    for scale_name, scale_cfg in SCALES.items():
        print(f"\nScale: {scale_name} (d_h = {scale_cfg['d_h']})")
        header = f"| Target | PRISM MAE | Twin MAE | Δ(%) | p-value | PaiNN MAE | PaiNN params |"
        print(header)
        print("|" + "-"*7 + "|" + "-"*11 + "|" + "-"*10 + "|" + "-"*6 + "|" + "-"*9 + "|" + "-"*11 + "|" + "-"*14 + "|")
        
        for target in args.targets:
            t_res = results[scale_name][target]
            prism = t_res["prism"]
            twin = t_res["twin"]
            painn = t_res["painn"]
            st = t_res["stats"]
            
            prism_str = f"{prism['mean_mae']:.4f}"
            twin_str = f"{twin['mean_mae']:.4f}"
            painn_str = f"{painn['mean_mae']:.4f}"
            
            # Δ(%) = (Twin - PRISM) / Twin * 100
            diff_pct = (twin['mean_mae'] - prism['mean_mae']) / twin['mean_mae'] * 100
            pval = st['ttest_p']
            
            print(f"| {target:<6} | {prism_str:<9} | {twin_str:<8} | {diff_pct:>3.1f}% | {pval:.4f} | {painn_str:<9} | {painn['params']:<12,} |")

    # Trend summary
    print("\n🔍 Scaling Trend Summary:")
    for target in args.targets:
        print(f"  Target: {target}")
        gaps = []
        for scale_name in SCALES:
            p = results[scale_name][target]["prism"]["mean_mae"]
            t = results[scale_name][target]["twin"]["mean_mae"]
            gaps.append(t - p) # positive means PRISM is better
        
        diff = gaps[-1] - gaps[0]
        if diff > 0.005:
            trend = "Widens"
        elif diff < -0.005:
            trend = "Narrows"
        else:
            trend = "Persists"
            
        print(f"    Small Gap: {gaps[0]:.4f} -> Large Gap: {gaps[-1]:.4f} ===> Advantage {trend}")

if __name__ == "__main__":
    main()
