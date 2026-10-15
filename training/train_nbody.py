"""3D charged N-body trajectory forecasting benchmark."""

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
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

from datasets.dataset_nbody import get_nbody_splits
from models.eq_gnn import EQGNNLayer
from models.egnn_baseline import EGNNLayer
from models.egnn_baseline import EGNNModel, create_param_matched_egnn, count_parameters
from models.classical_twin import ClassicalTwinTrajectoryModel
from scipy import stats

def compute_pairwise_tests(values_a, values_b):
    """Compute Wilcoxon signed-rank and paired t-test between two sets of metric values."""
    result = {}
    values_a = np.array(values_a)
    values_b = np.array(values_b)
    try:
        if len(values_a) >= 5:
            stat_w, p_wilcoxon = stats.wilcoxon(values_a, values_b)
        else:
            p_wilcoxon = float('nan')
        stat_t, p_ttest = stats.ttest_rel(values_a, values_b)
    except Exception:
        p_wilcoxon = float('nan')
        p_ttest = float('nan')
    result['wilcoxon_p'] = float(p_wilcoxon)
    result['ttest_p'] = float(p_ttest)
    diff = values_a - values_b
    mean_diff = np.mean(diff)
    se_diff = np.std(diff, ddof=1) / np.sqrt(len(diff))
    ci_95 = (float(mean_diff - 1.96 * se_diff), float(mean_diff + 1.96 * se_diff))
    result['mean_diff'] = float(mean_diff)
    result['ci_95'] = ci_95
    return result


# Trajectory models
class QPRISMTrajectoryModel(nn.Module):
    """Q-PRISM model for 3D coordinate trajectory forecasting."""
    def __init__(self, in_features=2, hidden_features=16, num_layers=3, n_q_layers=2):
        super().__init__()
        self.embedding = nn.Linear(in_features, hidden_features)
        self.layers = nn.ModuleList([
            EQGNNLayer(hidden_features, n_q_layers=n_q_layers) for _ in range(num_layers)
        ])

    def forward(self, h, x, edge_index):
        h = self.embedding(h)
        for layer in self.layers:
            h, x = layer(h, x, edge_index)
        return x  # [N, 3] Equivariant predicted coordinates


EQGNNTrajectoryModel = QPRISMTrajectoryModel  # Backward compatibility alias


class EGNNTrajectoryModel(nn.Module):
    """Classical EGNN model for 3D coordinate trajectory forecasting."""
    def __init__(self, in_features=2, hidden_features=16, num_layers=3):
        super().__init__()
        self.embedding = nn.Linear(in_features, hidden_features)
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_features, hidden_features=hidden_features) for _ in range(num_layers)
        ])

    def forward(self, h, x, edge_index):
        h = self.embedding(h)
        for layer in self.layers:
            h, x = layer(h, x, edge_index)
        return x  # [N, 3] Predicted coordinates


class NonEquivTrajectoryModel(nn.Module):
    """Non-equivariant GNN predicting coordinate offsets via Cartesian concatenation."""
    def __init__(self, in_features=2, hidden_features=16, num_layers=3):
        super().__init__()
        self.embedding = nn.Linear(in_features + 3, hidden_features)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_features * 2, hidden_features),
                nn.SiLU(),
                nn.Linear(hidden_features, hidden_features)
            ) for _ in range(num_layers)
        ])
        self.coord_head = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, 3)
        )

    def forward(self, h, x, edge_index):
        h = torch.cat([h, x], dim=-1)
        h = self.embedding(h)

        row, col = edge_index[0], edge_index[1]
        for layer in self.layers:
            msg = layer(torch.cat([h[row], h[col]], dim=-1))
            m_i = torch.zeros_like(h)
            m_i.index_add_(0, row, msg)
            h = h + m_i

        delta_x = self.coord_head(h)
        return x + delta_x


class ParamMatchedEGNNTrajectoryModel(nn.Module):
    """Param-Matched EGNN for trajectory forecasting (Tier 1 Iso-Parameter)."""
    def __init__(self, in_features=2, hidden_features=8, num_layers=3):
        super().__init__()
        self.embedding = nn.Linear(in_features, hidden_features)
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_features, hidden_features=hidden_features) for _ in range(num_layers)
        ])

    def forward(self, h, x, edge_index):
        h = self.embedding(h)
        for layer in self.layers:
            h, x = layer(h, x, edge_index)
        return x


# Training and evaluation
def train_epoch(model, loader, optimizer, criterion, device):
    """Train model for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred_pos = model(batch.x, batch.pos, batch.edge_index)
        loss = criterion(pred_pos, batch.target_pos)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        total_samples += batch.num_graphs

    return total_loss / total_samples


@torch.no_grad()
def evaluate_mse(model, loader, device):
    """Evaluate coordinate trajectory MSE."""
    model.eval()
    total_mse = 0.0
    total_particles = 0

    for batch in loader:
        batch = batch.to(device)
        pred_pos = model(batch.x, batch.pos, batch.edge_index)
        # Coordinate MSE: sum over particles
        mse = torch.sum((pred_pos - batch.target_pos)**2, dim=-1).sum().item()
        total_mse += mse
        total_particles += batch.pos.size(0)

    return total_mse / total_particles


def run_single_nbody_benchmark(model_type, train_loader, val_loader, test_loader, epochs=15, lr=1e-3, hidden_dim=16, num_layers=2):
    """Run a single N-body trajectory benchmark."""
    device = torch.device("cpu")
    if model_type == "eqgnn":
        model = EQGNNTrajectoryModel(in_features=2, hidden_features=hidden_dim, num_layers=num_layers, n_q_layers=2).to(device)
    elif model_type == "egnn":
        model = EGNNTrajectoryModel(in_features=2, hidden_features=hidden_dim, num_layers=num_layers).to(device)
    elif model_type == "noneq":
        model = NonEquivTrajectoryModel(in_features=2, hidden_features=hidden_dim, num_layers=num_layers).to(device)
    elif model_type == "egnn_pm":
        model = ParamMatchedEGNNTrajectoryModel(in_features=2, hidden_features=8, num_layers=num_layers).to(device)
    elif model_type == "classical_twin":
        model = ClassicalTwinTrajectoryModel(in_features=2, hidden_features=hidden_dim, num_layers=num_layers, n_q_layers=2).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.MSELoss()

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

    test_mse = evaluate_mse(model, test_loader, device)
    elapsed = time.time() - start_time
    return test_mse, total_params, elapsed


def run_full_nbody_suite(num_train=1000, num_val=250, num_test=500, seeds=[42, 43, 44, 45, 46], epochs=15, save_dir="./results"):
    """Run full N-body benchmark suite across models and seeds."""
    os.makedirs(save_dir, exist_ok=True)
    print("=" * 75)
    print("3D Charged N-Body Particle Trajectory Benchmark Suite (Table 2)")
    print(f"   Particles: 5 | N_train: {num_train} | Seeds: {seeds} | Epochs: {epochs}")
    print("=" * 75)

    results = {}
    models_to_test = ["noneq", "egnn_pm", "egnn", "eqgnn", "classical_twin"]

    for model_name in models_to_test:
        print(f"\nEvaluating Model: {model_name.upper()}")
        mses = []
        param_count = 0
        total_time = 0.0

        for seed in seeds:
            train_data, val_data, test_data = get_nbody_splits(
                num_train=num_train,
                num_val=num_val,
                num_test=num_test,
                seed=seed
            )
            train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
            test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

            torch.manual_seed(seed)
            mse, params, el_time = run_single_nbody_benchmark(
                model_type=model_name,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                epochs=epochs,
                hidden_dim=16,
                num_layers=2
            )
            mses.append(mse)
            param_count = params
            total_time += el_time
            print(f"    Seed {seed} | Test Coordinate MSE = {mse:.4f} ({el_time:.1f}s)")

        mean_mse = float(np.mean(mses))
        std_mse = float(np.std(mses))
        results[model_name] = {
            "params": param_count,
            "mean_mse": mean_mse,
            "std_mse": std_mse,
            "mses": mses,
            "avg_time": total_time / len(seeds)
        }
        print(f"  {model_name.upper()} Overall: MSE = {mean_mse:.4f} ± {std_mse:.4f} (Params: {param_count:,})")

    # Linear baseline (Initial static position)
    _, _, test_data = get_nbody_splits(num_train=num_train, num_val=num_val, num_test=num_test, seed=42)
    static_mse = np.mean([torch.sum((d.pos - d.target_pos)**2, dim=-1).mean().item() for d in test_data])
    results["static_baseline"] = {"mean_mse": float(static_mse), "std_mse": 0.0, "params": 0}
    print(f"\n  Static Baseline (x^(T) = x^(0)): MSE = {static_mse:.4f}")

    # Statistical significance tests
    stat_tests = {}
    eqgnn_mses = results['eqgnn']['mses']
    for baseline in ['egnn_pm', 'egnn', 'noneq', 'classical_twin']:
        if baseline in results:
            stat_tests[f'eqgnn_vs_{baseline}'] = compute_pairwise_tests(
                eqgnn_mses, results[baseline]['mses']
            )
    results['statistical_tests'] = stat_tests
    
    # Parameter compression ratio
    if 'egnn' in results and 'eqgnn' in results:
        compression = 1.0 - results['eqgnn']['params'] / results['egnn']['params']
        results['parameter_compression_ratio'] = float(compression)

    # Save JSON summary
    json_path = os.path.join(save_dir, "nbody_benchmark_table2_v2.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved Table 2 JSON results to: {json_path}")

    # Generate Figure 5: Trajectory MSE Comparison
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=300)

    model_keys = ["static_baseline", "noneq", "egnn_pm", "egnn", "eqgnn"]
    labels = ["Static Baseline\n($x^T = x^0$)", "Non-Equiv GNN\n(Cartesian)", "Param-Matched\nEGNN", "EGNN\n(Classical SOTA)", "Q-PRISM (Ours)\n(4-Qubit PQC)"]
    means = [results[k]["mean_mse"] for k in model_keys]
    stds = [results[k]["std_mse"] for k in model_keys]
    colors = ["#7f7f7f", "#d62728", "#9467bd", "#ff7f0e", "#1f77b4"]

    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors, width=0.55, edgecolor="black", alpha=0.9)
    ax.set_ylabel("Coordinate Trajectory Prediction MSE", fontsize=11, fontweight="bold")
    ax.set_title("3D Charged N-Body 5-Particle Trajectory Forecasting (Table 2)", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, which="both", ls="--", alpha=0.5)

    for bar, mean_val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015, f"{mean_val:.4f}", ha="center", fontweight="bold")

    plt.tight_layout()
    fig_path = os.path.join(save_dir, "figure5_nbody_trajectory.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved Figure 5 to: {fig_path}")

    # Copy to paper figures
    os.makedirs("paper/figures", exist_ok=True)
    os.system(f"cp {fig_path} paper/figures/")

    # Print Table 2 Summary
    print("\n" + "=" * 75)
    print("Table 2: 3D Charged N-Body Trajectory Forecasting (Mean ± Std over 5 Seeds)")
    print("=" * 75)
    print(f"| {'Model Architecture':<24} | {'Parameters':<12} | {'Coordinate MSE (10⁻²)':<25} |")
    print(f"|{'-'*26}|{'-'*14}|{'-'*27}|")
    for k, name in zip(model_keys, ["Static Baseline", "Non-Equivariant GNN", "Param-Matched EGNN", "EGNN (Classical SOTA)", "Q-PRISM (Ours)"]):
        d = results[k]
        p_str = f"{d['params']:,}" if d['params'] > 0 else "---"
        mse_str = f"{d['mean_mse']:.4f} ± {d['std_mse']:.4f}"
        print(f"| {name:<24} | {p_str:<12} | {mse_str:<25} |")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-train", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    args = parser.parse_args()

    run_full_nbody_suite(num_train=args.num_train, seeds=args.seeds, epochs=args.epochs)
