"""MD17 molecular dynamics energy prediction benchmark suite."""

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

from datasets.dataset_md17 import get_md17_splits
from models.eq_gnn import EQGNNModel
from models.egnn_baseline import EGNNModel
from models.egnn_baseline import create_param_matched_egnn, count_parameters
from models.classical_twin import ClassicalTwinModelV2
from scipy import stats

def compute_pairwise_tests(values_a, values_b):
    """Compute Wilcoxon signed-rank and paired t-test between two metric distributions."""
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

# Non-equivariant GNN baseline for MD17
class NonEquivMD17Model(nn.Module):
    """Standard non-equivariant GNN that concatenates raw Cartesian coordinates."""
    def __init__(self, in_features=10, hidden_features=32, num_layers=3, out_dim=1):
        super().__init__()
        self.embedding = nn.Linear(in_features + 3, hidden_features)  # concat pos
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_features * 2 + 1, hidden_features),
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
        h = torch.cat([h, x], dim=-1)
        h = self.embedding(h)

        row, col = edge_index[0], edge_index[1]
        for layer in self.layers:
            rel_x = x[row] - x[col]
            d_ij = torch.sum(rel_x**2, dim=-1, keepdim=True)
            msg = layer(torch.cat([h[row], h[col], d_ij], dim=-1))
            m_i = torch.zeros_like(h)
            m_i.index_add_(0, row, msg)
            h = h + m_i

        # Global mean pooling
        num_graphs = int(batch.max().item()) + 1
        out_pool = torch.zeros((num_graphs, h.size(1)), device=h.device)
        out_pool.index_add_(0, batch, h)
        counts = torch.zeros((num_graphs, 1), device=h.device)
        counts.index_add_(0, batch, torch.ones((h.size(0), 1), device=h.device))
        out_pool = out_pool / counts.clamp(min=1.0)

        return self.head(out_pool)


# Training and evaluation
def train_epoch(model, loader, optimizer, criterion, device, energy_mean, energy_std, model_type):
    """Train model for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(batch.x, batch.pos, batch.edge_index, batch.batch)
        # Target: normalized energy
        target = batch.energy_norm.view(-1, 1)
        loss = criterion(pred, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        total_samples += batch.num_graphs

    return total_loss / total_samples


@torch.no_grad()
def evaluate_energy_mae(model, loader, device, energy_mean, energy_std):
    """Evaluate energy MAE in original kcal/mol units."""
    model.eval()
    total_ae = 0.0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)
        pred_norm = model(batch.x, batch.pos, batch.edge_index, batch.batch)

        # Denormalize predictions
        pred_energy = pred_norm.view(-1) * energy_std + energy_mean
        true_energy = batch.energy.view(-1)

        total_ae += torch.sum(torch.abs(pred_energy - true_energy)).item()
        total_samples += batch.num_graphs

    return total_ae / total_samples


def run_single_md17_benchmark(model_type, molecule_name, train_data, val_data, test_data,
                               energy_mean, energy_std,
                               epochs=30, lr=5e-4, hidden_dim=16, num_layers=2,
                               batch_size=16, in_features=10):
    """Run a single MD17 energy prediction benchmark."""
    device = torch.device("cpu")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    if model_type == "eqgnn":
        model = EQGNNModel(
            in_features=in_features, hidden_features=hidden_dim,
            num_layers=num_layers, out_dim=1, n_q_layers=2
        ).to(device)
    elif model_type == "egnn":
        model = EGNNModel(
            in_features=in_features, hidden_features=hidden_dim,
            num_layers=num_layers, out_dim=1
        ).to(device)
    elif model_type == "noneq":
        model = NonEquivMD17Model(
            in_features=in_features, hidden_features=hidden_dim * 2,
            num_layers=num_layers, out_dim=1
        ).to(device)
    elif model_type == "egnn_pm":
        model = EGNNModel(
            in_features=in_features, hidden_features=8,
            num_layers=num_layers, out_dim=1
        ).to(device)
    elif model_type == "classical_twin":
        model = ClassicalTwinModelV2(
            in_features=in_features, hidden_features=hidden_dim,
            num_layers=num_layers, out_dim=1, n_q_layers=2
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.MSELoss()

    start_time = time.time()
    best_val_mae = float('inf')
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device,
                                  energy_mean, energy_std, model_type)
        scheduler.step()

        # Validate every 5 epochs
        if epoch % 5 == 0 or epoch == epochs:
            val_mae = evaluate_energy_mae(model, val_loader, device, energy_mean, energy_std)
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Load best model and evaluate on test set
    if best_state is not None:
        model.load_state_dict(best_state)

    test_mae = evaluate_energy_mae(model, test_loader, device, energy_mean, energy_std)
    elapsed = time.time() - start_time

    return test_mae, total_params, elapsed


# Full MD17 Benchmark Suite
def run_full_md17_suite(molecules=('ethanol', 'aspirin'),
                        num_train=1000, num_val=250, num_test=500,
                        seeds=(42, 43, 44, 45, 46), epochs=30,
                        save_dir="./results"):
    os.makedirs(save_dir, exist_ok=True)
    print("=" * 75)
    print("MD17 Molecular Dynamics Energy Prediction Benchmark Suite (Table 3)")
    print(f"   Molecules: {list(molecules)} | N_train: {num_train} | Seeds: {list(seeds)} | Epochs: {epochs}")
    print("=" * 75)

    all_results = {}
    models_to_test = ["noneq", "egnn_pm", "egnn", "eqgnn", "classical_twin"]
    model_labels = {
        "noneq": "Non-Equiv GNN",
        "egnn_pm": "EGNN Param-Matched",
        "egnn": "EGNN (Classical)",
        "eqgnn": "Q-PRISM (Ours)",
        "classical_twin": "Classical Twin",
    }

    for mol in molecules:
        print(f"\nMolecule: {mol.upper()}")

        mol_results = {}

        for model_name in models_to_test:
            print(f"\nModel: {model_labels[model_name]}")
            maes = []
            param_count = 0
            total_time = 0.0

            for seed in seeds:
                train_data, val_data, test_data, e_mean, e_std = get_md17_splits(
                    molecule_name=mol,
                    num_train=num_train,
                    num_val=num_val,
                    num_test=num_test,
                    seed=seed,
                    k_nn=8
                )

                torch.manual_seed(seed)
                np.random.seed(seed)

                mae, params, el_time = run_single_md17_benchmark(
                    model_type=model_name,
                    molecule_name=mol,
                    train_data=train_data,
                    val_data=val_data,
                    test_data=test_data,
                    energy_mean=e_mean,
                    energy_std=e_std,
                    epochs=epochs,
                    lr=5e-4,
                    hidden_dim=16,
                    num_layers=2,
                    batch_size=16,
                    in_features=10
                )
                maes.append(mae)
                param_count = params
                total_time += el_time
                print(f"    Seed {seed} | Test Energy MAE = {mae:.4f} kcal/mol ({el_time:.1f}s)")

            mean_mae = float(np.mean(maes))
            std_mae = float(np.std(maes))
            mol_results[model_name] = {
                "params": param_count,
                "mean_mae": mean_mae,
                "std_mae": std_mae,
                "maes": maes,
                "avg_time": total_time / len(seeds)
            }
            print(f"  {model_labels[model_name]}: {mean_mae:.4f} ± {std_mae:.4f} kcal/mol (Params: {param_count:,})")

        all_results[mol] = mol_results

    # Statistical significance tests per molecule
    for mol in molecules:
        stat_tests = {}
        if 'eqgnn' in all_results[mol]:
            eqgnn_maes = all_results[mol]['eqgnn']['maes']
            for baseline in ['egnn_pm', 'egnn', 'noneq']:
                if baseline in all_results[mol]:
                    stat_tests[f'eqgnn_vs_{baseline}'] = compute_pairwise_tests(
                        eqgnn_maes, all_results[mol][baseline]['maes']
                    )
            all_results[mol]['statistical_tests'] = stat_tests

    # Save JSON results
    json_path = os.path.join(save_dir, "md17_benchmark_table3_v2.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved Table 3 JSON results to: {json_path}")

    # Generate Figure 6: MD17 Energy Prediction Comparison
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    n_mols = len(molecules)
    fig, axes = plt.subplots(1, n_mols, figsize=(6 * n_mols, 5), dpi=300)
    if n_mols == 1:
        axes = [axes]

    colors = {"noneq": "#d62728", "egnn_pm": "#9467bd", "egnn": "#ff7f0e", "eqgnn": "#1f77b4"}
    labels = {"noneq": "Non-Equiv\nGNN", "egnn_pm": "EGNN\n(PM)", "egnn": "EGNN\n(Classical)", "eqgnn": "Q-PRISM\n(Ours)"}

    for ax, mol in zip(axes, molecules):
        mol_res = all_results[mol]
        model_keys = ["noneq", "egnn_pm", "egnn", "eqgnn"]
        means = [mol_res[k]["mean_mae"] for k in model_keys]
        stds = [mol_res[k]["std_mae"] for k in model_keys]
        bar_labels = [labels[k] for k in model_keys]
        bar_colors = [colors[k] for k in model_keys]

        bars = ax.bar(bar_labels, means, yerr=stds, capsize=5,
                      color=bar_colors, width=0.5, edgecolor="black", alpha=0.9)
        ax.set_ylabel("Energy MAE [kcal/mol]", fontsize=11, fontweight="bold")
        ax.set_title(f"{mol.capitalize()}", fontsize=13, fontweight="bold", pad=10)
        ax.grid(True, which="both", ls="--", alpha=0.5)

        for bar, mean_val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{mean_val:.2f}", ha="center", fontweight="bold", fontsize=9)

    fig.suptitle("MD17 Conformational Energy Prediction (Table 3)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    fig_path = os.path.join(save_dir, "figure6_md17_energy.png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 6 to: {fig_path}")

    os.makedirs("paper/figures", exist_ok=True)
    os.system(f"cp {fig_path} paper/figures/")

    # Print Table 3 Summary
    print("\n" + "=" * 85)
    print("Table 3: MD17 Conformational Energy Prediction (Mean ± Std over 5 Seeds) [kcal/mol]")
    print("=" * 85)

    header = f"| {'Model':<24} | {'Parameters':<12} |"
    for mol in molecules:
        header += f" {mol.capitalize():<24} |"
    print(header)
    print(f"|{'-'*26}|{'-'*14}|" + "|".join([f"{'-'*26}" for _ in molecules]) + "|")

    for model_name in models_to_test:
        row = f"| {model_labels[model_name]:<24} | "
        params = all_results[molecules[0]][model_name]['params']
        row += f"{params:>10,} |"
        for mol in molecules:
            d = all_results[mol][model_name]
            row += f" {d['mean_mae']:.4f} ± {d['std_mae']:.4f}        |"
        print(row)
    print("=" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MD17 Energy Prediction Benchmark")
    parser.add_argument("--molecules", nargs="+", default=["ethanol", "aspirin"])
    parser.add_argument("--num-train", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    args = parser.parse_args()

    run_full_md17_suite(
        molecules=tuple(args.molecules),
        num_train=args.num_train,
        seeds=tuple(args.seeds),
        epochs=args.epochs
    )
