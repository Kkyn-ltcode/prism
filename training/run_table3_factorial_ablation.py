"""Factorial ablation study of activation functions and initialization schemes."""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import sys
from pathlib import Path
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import math
from typing import Dict, Any

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.eq_gnn import GaussianRBF, compute_edge_mean_angles, N_RBF
from models.classical_trig_twin import PRISMModelV2
from models.classical_twin import ClassicalTwinModelV2
from datasets.dataset_qm9 import get_qm9_splits, QM9_TARGET_INDICES
from training.run_table1_qm9_benchmark import (
    TARGET_UNITS, TARGET_DISPLAY, TARGET_TYPE,
    compute_statistical_tests, train_eval_single_run,
    resolve_data_dir, get_model_device
)


# Custom heads for factorial ablation

class SinXavierHead(nn.Module):
    """Sin activation head with standard Xavier uniform initialization."""
    def __init__(self, in_features, n_q_layers=2, n_rbf=N_RBF, omega_0=30.0):
        super().__init__()
        self.feature_proj = nn.Linear(in_features, 2)
        self.omega_0 = omega_0

        input_dim = 4 + n_rbf + 1
        n_outputs = 8
        hidden = 4

        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, n_outputs, bias=False)

        # Standard Xavier init instead of SIREN init
        nn.init.xavier_uniform_(self.linear1.weight)
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, h_row, h_col, rbf_features, angles):
        h_i_proj = self.feature_proj(h_row)
        h_j_proj = self.feature_proj(h_col)
        inputs = torch.cat([h_i_proj, h_j_proj, rbf_features, angles], dim=-1)

        x = torch.sin(self.omega_0 * self.linear1(inputs))
        return torch.tanh(self.linear2(x))


class TanhSirenInitHead(nn.Module):
    """SiLU activation head with SIREN-analogous initialization."""
    def __init__(self, in_features, n_q_layers=2, n_rbf=N_RBF, omega_0=30.0):
        super().__init__()
        self.feature_proj = nn.Linear(in_features, 2)
        self.omega_0 = omega_0

        input_dim = 4 + n_rbf + 1
        n_outputs = 8
        hidden = 4

        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, n_outputs, bias=False)

        # SIREN-analogous init
        n_in1 = self.linear1.in_features
        with torch.no_grad():
            self.linear1.weight.uniform_(-1.0 / n_in1, 1.0 / n_in1)
            if self.linear1.bias is not None:
                self.linear1.bias.uniform_(-1.0 / n_in1, 1.0 / n_in1)

        n_in2 = self.linear2.in_features
        bound2 = math.sqrt(6.0 / n_in2) / self.omega_0
        with torch.no_grad():
            self.linear2.weight.uniform_(-bound2, bound2)

    def forward(self, h_row, h_col, rbf_features, angles):
        h_i_proj = self.feature_proj(h_row)
        h_j_proj = self.feature_proj(h_col)
        inputs = torch.cat([h_i_proj, h_j_proj, rbf_features, angles], dim=-1)

        # SiLU matches ClassicalHead
        x = torch.nn.functional.silu(self.linear1(inputs))
        return torch.tanh(self.linear2(x))


# Custom layer and model for swappable heads

class AblationLayer(nn.Module):
    """Equivariant layer with customizable head module."""
    def __init__(self, in_features, head_class, n_q_layers=2, n_rbf=N_RBF, omega_0=30.0):
        super().__init__()
        self.in_features = in_features
        self.n_rbf = n_rbf
        self.rbf = GaussianRBF(n_rbf=n_rbf)

        # Instantiate provided head class
        if head_class in (SinXavierHead, TanhSirenInitHead):
            self.head_electronic = head_class(in_features, n_q_layers=n_q_layers, n_rbf=n_rbf, omega_0=omega_0)
            self.head_topological = head_class(in_features, n_q_layers=n_q_layers, n_rbf=n_rbf, omega_0=omega_0)
        else:
            self.head_electronic = head_class(in_features, n_q_layers=n_q_layers, n_rbf=n_rbf)
            self.head_topological = head_class(in_features, n_q_layers=n_q_layers, n_rbf=n_rbf)

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


class AblationModel(nn.Module):
    """Equivariant model for head ablation studies."""
    def __init__(self, head_class, in_features=11, hidden_features=16, num_layers=3, out_dim=1,
                 n_q_layers=2, n_rbf=N_RBF, omega_0=30.0):
        super().__init__()
        self.embedding = nn.Linear(in_features, hidden_features)
        self.layers = nn.ModuleList([
            AblationLayer(hidden_features, head_class, n_q_layers=n_q_layers, n_rbf=n_rbf, omega_0=omega_0)
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


MODEL_CONFIGS = {
    "prism_siren_init": {"label": "PRISM (Sin + SIREN Init)", "short": "PRISM"},
    "sin_xavier_init": {"label": "Ablation (Sin + Xavier)", "short": "Sin+Xavier"},
    "tanh_siren_init": {"label": "Ablation (Tanh + SIREN Init)", "short": "Tanh+SIREN"},
    "classical_twin": {"label": "Classical Twin (Tanh + Standard)", "short": "ClassTwin"}
}

def get_ablation_model(model_key, target_dim=1, hidden_dim=16):
    """Instantiate ablation model by key."""
    if model_key == "prism_siren_init":
        return PRISMModelV2(in_features=11, hidden_features=hidden_dim, num_layers=2, out_dim=target_dim, head_type="siren", omega_0=30.0)
    elif model_key == "classical_twin":
        return ClassicalTwinModelV2(in_features=11, hidden_features=hidden_dim, num_layers=2, out_dim=target_dim)
    elif model_key == "sin_xavier_init":
        return AblationModel(SinXavierHead, in_features=11, hidden_features=hidden_dim, num_layers=2, out_dim=target_dim, omega_0=30.0)
    elif model_key == "tanh_siren_init":
        return AblationModel(TanhSirenInitHead, in_features=11, hidden_features=hidden_dim, num_layers=2, out_dim=target_dim, omega_0=30.0)
    else:
        raise ValueError(f"Unknown model_key: {model_key}")

def custom_create_model(model_key, in_features=11, hidden_dim=16, num_layers=3, out_dim=1):
    """Create model and return parameter count."""
    model = get_ablation_model(model_key, target_dim=out_dim, hidden_dim=hidden_dim)
    return model, count_parameters(model)

import training.run_table1_qm9_benchmark
training.run_table1_qm9_benchmark.create_model = custom_create_model


def cohens_d(x, y):
    """Compute Cohen's d effect size between two samples."""
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pool_sd = np.sqrt(((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof)
    return (np.mean(x) - np.mean(y)) / pool_sd


def run_benchmark(targets, seeds, num_train, num_val, num_test, epochs, batch_size, lr, hidden_dim, device, data_dir, save_dir):
    """Run the factorial ablation benchmark."""
    data_dir = resolve_data_dir(data_dir)
    if save_dir is None:
        save_dir = "/kaggle/working/results" if os.path.exists("/kaggle/working") else "./results"
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, "revision_c1_init_ablation.json")

    # Verification Block
    print("=" * 70)
    print("Model Parameter Count Verification")
    print("=" * 70)
    for mk in MODEL_CONFIGS.keys():
        m = get_ablation_model(mk, target_dim=1, hidden_dim=hidden_dim)
        pc = count_parameters(m)
        print(f"{mk:<20}: {pc:,} parameters")
    print("=" * 70)
    
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            results = json.load(f)
    else:
        results = {}

    for target in targets:
        t_type = TARGET_TYPE.get(target, "other")
        print(f"\nTarget: {TARGET_DISPLAY[target]} [{TARGET_UNITS[target]}] ({t_type})")

        if target not in results:
            results[target] = {
                "unit": TARGET_UNITS[target],
                "display": TARGET_DISPLAY[target],
                "type": t_type,
            }

        for model_key in MODEL_CONFIGS.keys():
            if model_key in results[target] and "maes" in results[target][model_key]:
                existing = results[target][model_key]
                print(f"\nModel: {MODEL_CONFIGS[model_key]['label']} - Cached: {existing['mean_mae']:.4f} ± {existing['std_mae']:.4f}")
                continue

            config = MODEL_CONFIGS[model_key]
            model_dev = get_model_device(model_key, device)
            print(f"\nModel: {config['label']} on {model_dev}")
            maes = []
            param_count = 0
            for seed in seeds:
                mae, param_count = train_eval_single_run(
                    model_key=model_key,
                    target=target,
                    seed=seed,
                    num_train=num_train,
                    num_val=num_val,
                    num_test=num_test,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    hidden_dim=hidden_dim,
                    device=model_dev,
                    data_dir=data_dir,
                )
                maes.append(mae)

            mean_mae = float(np.mean(maes))
            std_mae = float(np.std(maes))
            results[target][model_key] = {
                "label": config["label"],
                "params": param_count,
                "seeds": seeds,
                "maes": maes,
                "mean_mae": mean_mae,
                "std_mae": std_mae,
            }
            print(f"  {config['label']} ({len(seeds)} Seeds): {mean_mae:.4f} ± {std_mae:.4f} {TARGET_UNITS[target]} (Params: {param_count:,})")

            with open(json_path, "w") as f:
                json.dump(results, f, indent=2)

        # Statistical testing
        stat_tests = {}
        for m1 in MODEL_CONFIGS.keys():
            for m2 in MODEL_CONFIGS.keys():
                if m1 != m2 and m1 in results[target] and m2 in results[target]:
                    maes1 = results[target][m1]["maes"]
                    maes2 = results[target][m2]["maes"]
                    st = compute_statistical_tests(maes1, maes2)
                    st["cohens_d"] = cohens_d(maes1, maes2)
                    stat_tests[f"{m1}_vs_{m2}"] = st
        results[target]["statistical_tests"] = stat_tests
        
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Print Publication Table
    print("\n" + "=" * 140)
    print(f"Table: 2x2 Factorial Init/Activation Ablation (N={num_train:,}, Mean ± Std over {len(seeds)} Seeds)")
    print("=" * 140)
    
    header = f"| {'Target':<15} | {'Unit':<8} |"
    for m in MODEL_CONFIGS.keys():
        header += f" {MODEL_CONFIGS[m]['short']:<20} |"
    print(header)
    divider = f"|{'-'*17}|{'-'*10}|" + "|".join([f"{'-'*22}" for _ in MODEL_CONFIGS]) + "|"
    print(divider)
    
    for t in targets:
        d = results[t]
        row = f"| {d['display']:<15} | {d['unit']:<8} |"
        valid_maes = {m: d[m]['mean_mae'] for m in MODEL_CONFIGS.keys() if m in d and 'mean_mae' in d[m]}
        best_mae = min(valid_maes.values()) if valid_maes else float('inf')
        for m in MODEL_CONFIGS.keys():
            if m in d and 'mean_mae' in d[m]:
                mae_str = f"{d[m]['mean_mae']:.4f} ± {d[m]['std_mae']:.4f}"
                if d[m]['mean_mae'] == best_mae:
                    mae_str = f"**{mae_str}**"
                row += f" {mae_str:<20} |"
            else:
                row += f" {'---':<20} |"
        print(row)
    print(divider)

    print("\n" + "=" * 140)
    print("2x2 Factorial ANOVA Effects and Pairwise Comparisons")
    print("=" * 140)
    for t in targets:
        d = results[t]
        if not all(m in d and 'mean_mae' in d[m] for m in MODEL_CONFIGS):
            continue
        print(f"\n--- Target: {d['display']} ---")
        
        # Means
        mean_sin_siren = d["prism_siren_init"]["mean_mae"]
        mean_sin_xav = d["sin_xavier_init"]["mean_mae"]
        mean_tanh_siren = d["tanh_siren_init"]["mean_mae"]
        mean_tanh_xav = d["classical_twin"]["mean_mae"]
        
        # Main effects
        eff_act = ((mean_sin_siren + mean_sin_xav) / 2) - ((mean_tanh_siren + mean_tanh_xav) / 2)
        eff_init = ((mean_sin_siren + mean_tanh_siren) / 2) - ((mean_sin_xav + mean_tanh_xav) / 2)
        
        # Interaction effect (difference in differences)
        eff_inter = (mean_sin_siren - mean_sin_xav) - (mean_tanh_siren - mean_tanh_xav)
        
        print(f"Main Effect: Activation (Sin vs Tanh): {eff_act:+.4f}")
        print(f"Main Effect: Initialization (SIREN vs Xavier): {eff_init:+.4f}")
        print(f"Interaction Effect: {eff_inter:+.4f}")
        
        print("\nPairwise Comparisons:")
        st = d.get("statistical_tests", {})
        pairs_to_print = [
            ("prism_siren_init", "sin_xavier_init", "Init Effect for Sin"),
            ("tanh_siren_init", "classical_twin", "Init Effect for Tanh"),
            ("prism_siren_init", "tanh_siren_init", "Activation Effect for SIREN"),
            ("sin_xavier_init", "classical_twin", "Activation Effect for Xavier"),
            ("prism_siren_init", "classical_twin", "Combined Effect (PRISM vs Classical Twin)"),
        ]
        for m1, m2, desc in pairs_to_print:
            k = f"{m1}_vs_{m2}"
            if k in st:
                stats = st[k]
                p_val = stats['ttest_p']
                sig = "**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns")
                print(f"  {desc}:")
                print(f"    {MODEL_CONFIGS[m1]['short']} vs {MODEL_CONFIGS[m2]['short']}")
                print(f"    Δ MAE: {stats['mean_diff']:+.4f}, t-test p={p_val:.4e} {sig}, Cohen's d={stats['cohens_d']:+.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Activation vs Init Ablation")
    parser.add_argument("--targets", nargs="+", default=["homo", "lumo", "gap"], choices=list(QM9_TARGET_INDICES.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51])
    parser.add_argument("--num-train", type=int, default=1000)
    parser.add_argument("--num-val", type=int, default=250)
    parser.add_argument("--num-test", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()

    run_benchmark(
        targets=args.targets,
        seeds=args.seeds,
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        device=args.device,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
    )
