"""Multi-target multi-seed QM9 benchmark suite across chemical targets."""

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
from scipy import stats
from torch_geometric.loader import DataLoader

from datasets.dataset_qm9 import get_qm9_splits, QM9_TARGET_INDICES, QM9_ELECTRONIC_TARGETS
from models.eq_gnn import EQGNNModelV2
from models.egnn_baseline import EGNNModel, create_param_matched_egnn, count_parameters


# Target metadata
TARGET_UNITS = {
    "homo": "eV",
    "lumo": "eV",
    "gap": "eV",
    "mu": "Debye",
    "alpha": "Bohr³",
    "u0": "eV",
}

TARGET_DISPLAY = {
    "homo": "HOMO Energy (ε_HOMO)",
    "lumo": "LUMO Energy (ε_LUMO)",
    "gap": "HOMO-LUMO Gap (Δε)",
    "mu": "Dipole Moment (μ)",
    "alpha": "Polarizability (α)",
    "u0": "Internal Energy (U₀)",
}

TARGET_TYPE = {
    "homo": "electronic",
    "lumo": "electronic",
    "gap": "electronic",
    "mu": "extensive",
    "alpha": "extensive",
    "u0": "extensive",
}

# Model configurations
MODEL_CONFIGS = {
    "eqgnn_v2": {
        "label": "Q-PRISM (Ours)",
        "tier": 1,
    },
    "egnn_pm": {
        "label": "EGNN Param-Matched",
        "tier": 1,
    },
    "egnn_std": {
        "label": "EGNN Standard",
        "tier": 2,
    },
}


# Statistical significance tests
def compute_statistical_tests(maes_a, maes_b):
    """Compute two-sided Wilcoxon signed-rank and paired Student's t-test."""
    a = np.array(maes_a, dtype=np.float64)
    b = np.array(maes_b, dtype=np.float64)
    result = {}

    # Wilcoxon signed-rank test (requires >= 5 samples for meaningful result)
    try:
        if len(a) >= 5:
            _, p_wilcoxon = stats.wilcoxon(a, b)
        else:
            p_wilcoxon = float("nan")
    except Exception:
        p_wilcoxon = float("nan")
    result["wilcoxon_p"] = float(p_wilcoxon)

    # Paired Student's t-test
    try:
        _, p_ttest = stats.ttest_rel(a, b)
    except Exception:
        p_ttest = float("nan")
    result["ttest_p"] = float(p_ttest)

    # Mean difference and 95% Confidence Interval
    diff = a - b
    mean_diff = float(np.mean(diff))
    se_diff = float(np.std(diff, ddof=1) / np.sqrt(len(diff)))
    ci_low = mean_diff - 1.96 * se_diff
    ci_high = mean_diff + 1.96 * se_diff
    result["mean_diff"] = mean_diff
    result["ci_95"] = [float(ci_low), float(ci_high)]

    # Significance flags
    p_min = min(p_wilcoxon, p_ttest) if not (np.isnan(p_wilcoxon) and np.isnan(p_ttest)) else float("nan")
    result["significant_005"] = bool(p_min < 0.05) if not np.isnan(p_min) else False
    result["significant_001"] = bool(p_min < 0.01) if not np.isnan(p_min) else False

    return result


# Single run: train and evaluate
def train_eval_single_run(model_key, target, seed,
                          num_train=1000, num_val=250, num_test=500,
                          epochs=15, batch_size=16, lr=1e-3, hidden_dim=16):
    """Train and evaluate a single model on a single target with a single seed."""
    device = torch.device("cpu")

    train_set, val_set, test_set, (mean, std) = get_qm9_splits(
        root="./data/QM9",
        target=target,
        num_train=num_train,
        num_val=num_val,
        num_test=num_test,
        seed=seed
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    torch.manual_seed(seed)

    if model_key == "eqgnn_v2":
        model = EQGNNModelV2(
            in_features=11,
            hidden_features=hidden_dim,
            num_layers=2,
            n_q_layers=2,
            out_dim=1
        ).to(device)
    elif model_key == "egnn_pm":
        # Auto-match parameter count to EQ-GNN V2
        ref_model = EQGNNModelV2(in_features=11, hidden_features=hidden_dim, num_layers=2, n_q_layers=2, out_dim=1)
        ref_params = count_parameters(ref_model)
        model, _ = create_param_matched_egnn(target_params=ref_params, in_features=11, num_layers=2, out_dim=1)
        model = model.to(device)
        del ref_model
    elif model_key == "egnn_std":
        model = EGNNModel(
            in_features=11,
            hidden_features=hidden_dim,
            num_layers=2,
            out_dim=1
        ).to(device)
    else:
        raise ValueError(f"Unknown model key: {model_key}")

    total_params = count_parameters(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.MSELoss()

    start_time = time.time()

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

    # Evaluate on test set
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

    test_mae = total_mae / total_samples
    elapsed = time.time() - start_time
    print(f"    Seed {seed} | Test MAE = {test_mae:.4f} {TARGET_UNITS[target]} ({elapsed:.1f}s)")
    return test_mae, total_params


# Main benchmark loop
def run_benchmark(targets=None, seeds=None, models=None,
                  num_train=1000, epochs=15, save_dir="./results"):
    if targets is None:
        targets = ["homo", "lumo", "gap", "mu", "alpha", "u0"]
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]
    if models is None:
        models = ["eqgnn_v2", "egnn_pm", "egnn_std"]

    os.makedirs(save_dir, exist_ok=True)
    print("=" * 85)
    print("Multi-Target Multi-Seed QM9 Benchmark Suite (Table 1)")
    print(f"   Targets: {targets}")
    print(f"   Models: {[MODEL_CONFIGS[m]['label'] for m in models]}")
    print(f"   Seeds: {seeds} | N_train: {num_train} | Epochs: {epochs}")
    print("=" * 85)

    json_path = os.path.join(save_dir, "multitarget_table1_v2.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                results = json.load(f)
        except Exception:
            results = {}
    else:
        results = {}

    for target in targets:
        t_type = TARGET_TYPE.get(target, "other")
        print(f"\nTarget: {TARGET_DISPLAY[target]} [{TARGET_UNITS[target]}] ({t_type})")

        results[target] = {
            "unit": TARGET_UNITS[target],
            "display": TARGET_DISPLAY[target],
            "type": t_type,
        }

        for model_key in models:
            config = MODEL_CONFIGS[model_key]
            print(f"\nModel: {config['label']} (Tier {config['tier']})")
            maes = []
            param_count = 0
            for seed in seeds:
                mae, param_count = train_eval_single_run(
                    model_key=model_key,
                    target=target,
                    seed=seed,
                    num_train=num_train,
                    epochs=epochs
                )
                maes.append(mae)

            mean_mae = float(np.mean(maes))
            std_mae = float(np.std(maes))
            results[target][model_key] = {
                "label": config["label"],
                "tier": config["tier"],
                "params": param_count,
                "seeds": seeds,
                "maes": maes,
                "mean_mae": mean_mae,
                "std_mae": std_mae,
            }
            print(f"  {config['label']} ({len(seeds)} Seeds): "
                  f"{mean_mae:.4f} ± {std_mae:.4f} {TARGET_UNITS[target]} "
                  f"(Params: {param_count:,})")

        # Statistical significance tests: EQ-GNN V2 vs each baseline
        if "eqgnn_v2" in results[target]:
            stat_tests = {}
            eqgnn_maes = results[target]["eqgnn_v2"]["maes"]
            for baseline in models:
                if baseline != "eqgnn_v2" and baseline in results[target]:
                    stat_tests[f"eqgnn_v2_vs_{baseline}"] = compute_statistical_tests(
                        eqgnn_maes, results[target][baseline]["maes"]
                    )
            results[target]["statistical_tests"] = stat_tests

            # Print significance results
            print(f"\n  Statistical Significance ({target}):")
            for key, st in stat_tests.items():
                sig_marker = "**" if st["significant_001"] else ("*" if st["significant_005"] else "ns")
                print(f"     {key}: Wilcoxon p={st['wilcoxon_p']:.4f}, "
                      f"t-test p={st['ttest_p']:.4f}, "
                      f"Δ={st['mean_diff']:.4f} {sig_marker}")

    # Save JSON
    json_path = os.path.join(save_dir, "multitarget_table1_v2.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark results to: {json_path}")

    # Print Publication Table
    print("\n" + "=" * 115)
    print(f"Table 1: 3-Tier Multi-Target QM9 Results (N={num_train:,}, Mean ± Std over {len(seeds)} Seeds)")
    print("=" * 115)

    # Header
    header = f"| {'Target':<28} | {'Unit':<6} | {'Type':<11} |"
    for m in models:
        header += f" {MODEL_CONFIGS[m]['label']:<22} |"
    print(header)
    divider = f"|{'-'*30}|{'-'*8}|{'-'*13}|"
    for _ in models:
        divider += f"{'-'*24}|"
    print(divider)

    # Rows
    for t in targets:
        d = results[t]
        row = f"| {d['display']:<28} | {d['unit']:<6} | {d['type']:<11} |"
        for m in models:
            if m in d:
                entry = d[m]
                mae_str = f"{entry['mean_mae']:.4f} ± {entry['std_mae']:.4f}"
                row += f" {mae_str:<22} |"
            else:
                row += f" {'---':<22} |"
        print(row)

    # Param counts
    param_row = f"| {'Parameters':<28} | {'':<6} | {'':<11} |"
    for m in models:
        first_target = targets[0]
        if m in results[first_target]:
            p = results[first_target][m]["params"]
            param_row += f" {p:>20,} |"
        else:
            param_row += f" {'---':<22} |"
    print(divider)
    print(param_row)
    print("=" * 115)

    # Significance summary
    print("\nStatistical Significance Summary (EQ-GNN V2 vs Baselines):")
    print(f"| {'Target':<28} | {'vs EGNN-PM Wilcoxon':<22} | {'vs EGNN-PM t-test':<22} | {'vs EGNN-Std Wilcoxon':<22} | {'vs EGNN-Std t-test':<22} |")
    for t in targets:
        st = results[t].get("statistical_tests", {})
        pm_w = st.get("eqgnn_v2_vs_egnn_pm", {}).get("wilcoxon_p", float("nan"))
        pm_t = st.get("eqgnn_v2_vs_egnn_pm", {}).get("ttest_p", float("nan"))
        std_w = st.get("eqgnn_v2_vs_egnn_std", {}).get("wilcoxon_p", float("nan"))
        std_t = st.get("eqgnn_v2_vs_egnn_std", {}).get("ttest_p", float("nan"))
        print(f"| {TARGET_DISPLAY[t]:<28} | {pm_w:<22.4f} | {pm_t:<22.4f} | {std_w:<22.4f} | {std_t:<22.4f} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Target Multi-Seed QM9 Benchmark v2.0")
    parser.add_argument("--targets", nargs="+",
                        default=["homo", "lumo", "gap", "mu", "alpha", "u0"],
                        choices=list(QM9_TARGET_INDICES.keys()))
    parser.add_argument("--models", nargs="+",
                        default=["eqgnn_v2", "egnn_pm", "egnn_std"],
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--num-train", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--save-dir", type=str, default="./results")
    args = parser.parse_args()

    run_benchmark(
        targets=args.targets,
        seeds=args.seeds,
        models=args.models,
        num_train=args.num_train,
        epochs=args.epochs,
        save_dir=args.save_dir,
    )
