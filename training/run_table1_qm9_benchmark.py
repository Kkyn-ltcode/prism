"""Extended multi-target multi-seed QM9 benchmark."""

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
from models.eq_gnn import QPRISMModelV2, EQGNNModelV2
from models.egnn_baseline import EGNNModel, create_param_matched_egnn, count_parameters
from models.classical_twin import ClassicalTwinModelV2
from models.classical_trig_twin import PRISMModelV2, TrigTwinModelV2
from models.classical_baselines import SchNetModel, PaiNNModel


# Target Metadata

TARGET_UNITS = {
    "homo": "eV", "lumo": "eV", "gap": "eV",
    "mu": "Debye", "alpha": "Bohr³", "u0": "eV",
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
    "homo": "electronic", "lumo": "electronic", "gap": "electronic",
    "mu": "extensive", "alpha": "extensive", "u0": "extensive",
}

# Extended Model Configurations

MODEL_CONFIGS = {
    "eqgnn_v2": {
        "label": "Q-PRISM (Ours)",
        "tier": 1,
        "short": "Q-PRISM",
    },
    "classical_twin": {
        "label": "Classical Twin",
        "tier": 1,
        "short": "Twin",
    },
    "trig_twin_siren": {
        "label": "PRISM (SIREN)",
        "tier": 1,
        "short": "PRISM",
    },
    "trig_twin_fourier": {
        "label": "Fourier Twin (RFF)",
        "tier": 1,
        "short": "Fourier",
    },
    "schnet": {
        "label": "SchNet",
        "tier": 1,
        "short": "SchNet",
    },
    "painn": {
        "label": "PaiNN",
        "tier": 1,
        "short": "PaiNN",
    },
    "egnn_pm": {
        "label": "EGNN Param-Matched",
        "tier": 1,
        "short": "EGNN-PM",
    },
    "egnn_std": {
        "label": "EGNN Standard",
        "tier": 2,
        "short": "EGNN-Std",
    },
}


# Statistical Significance Tests
def compute_statistical_tests(maes_a, maes_b):
    """Two-sided Wilcoxon signed-rank + paired t-test between seed MAEs."""
    a = np.array(maes_a, dtype=np.float64)
    b = np.array(maes_b, dtype=np.float64)
    result = {}

    try:
        _, p_wilcoxon = stats.wilcoxon(a, b) if len(a) >= 5 else (None, float("nan"))
    except Exception:
        p_wilcoxon = float("nan")
    result["wilcoxon_p"] = float(p_wilcoxon)

    try:
        _, p_ttest = stats.ttest_rel(a, b)
    except Exception:
        p_ttest = float("nan")
    result["ttest_p"] = float(p_ttest)

    diff = a - b
    mean_diff = float(np.mean(diff))
    se_diff = float(np.std(diff, ddof=1) / np.sqrt(len(diff)))
    ci_low = mean_diff - 1.96 * se_diff
    ci_high = mean_diff + 1.96 * se_diff
    result["mean_diff"] = mean_diff
    result["ci_95"] = [float(ci_low), float(ci_high)]

    p_min = min(p_wilcoxon, p_ttest) if not (np.isnan(p_wilcoxon) and np.isnan(p_ttest)) else float("nan")
    result["significant_005"] = bool(p_min < 0.05) if not np.isnan(p_min) else False
    result["significant_001"] = bool(p_min < 0.01) if not np.isnan(p_min) else False

    return result


# Model Factory

def create_model(model_key, in_features=11, hidden_dim=16, num_layers=2, out_dim=1):
    """Create a model by key. Returns (model, param_count)."""

    if model_key == "eqgnn_v2":
        model = QPRISMModelV2(
            in_features=in_features,
            hidden_features=hidden_dim,
            num_layers=num_layers,
            n_q_layers=2,
            out_dim=out_dim,
        )
    elif model_key == "egnn":
        model = create_param_matched_egnn(
            target_params=2527,
            in_features=in_features,
            num_layers=num_layers,
            out_dim=out_dim,
        )
    elif model_key == "classical_twin":
        model = ClassicalTwinModelV2(
            in_features=in_features,
            hidden_features=hidden_dim,
            num_layers=num_layers,
            n_q_layers=2,
            out_dim=out_dim,
        )
    elif model_key == "trig_twin_siren":
        model = PRISMModelV2(
            in_features=in_features,
            hidden_features=hidden_dim,
            num_layers=num_layers,
            n_q_layers=2,
            out_dim=out_dim,
            head_type="siren",
            omega_0=30.0,
        )
    elif model_key == "trig_twin_fourier":
        model = PRISMModelV2(
            in_features=in_features,
            hidden_features=hidden_dim,
            num_layers=num_layers,
            n_q_layers=2,
            out_dim=out_dim,
            head_type="fourier",
        )
    elif model_key == "schnet":
        # Best param match: hidden_channels=12 → 2,473 params (Δ=-54)
        model = SchNetModel(
            in_features=in_features,
            hidden_channels=12,
            num_interactions=num_layers,
            num_gaussians=50,
            cutoff=10.0,
            out_dim=out_dim,
        )
    elif model_key == "painn":
        # Best param match: hidden_channels=8 → 2,721 params (Δ=+194)
        model = PaiNNModel(
            in_features=in_features,
            hidden_channels=8,
            num_interactions=num_layers,
            n_rbf=20,
            cutoff=5.0,
            out_dim=out_dim,
        )
    elif model_key == "egnn_pm":
        ref_model = EQGNNModelV2(in_features=in_features, hidden_features=hidden_dim,
                                  num_layers=num_layers, n_q_layers=2, out_dim=out_dim)
        ref_params = count_parameters(ref_model)
        model, _ = create_param_matched_egnn(target_params=ref_params,
                                              in_features=in_features,
                                              num_layers=num_layers, out_dim=out_dim)
        del ref_model
    elif model_key == "egnn_std":
        model = EGNNModel(
            in_features=in_features,
            hidden_features=hidden_dim,
            num_layers=num_layers,
            out_dim=out_dim,
        )
    else:
        raise ValueError(f"Unknown model key: {model_key}")

    return model, count_parameters(model)


# Kaggle & Hardware Helpers

def resolve_data_dir(data_dir=None):
    """Auto-resolve QM9 dataset directory with Kaggle environment support."""
    if data_dir is not None:
        return data_dir

    # Check common Kaggle dataset mounts
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for candidate in [
            kaggle_input / "qm9-dataset",
            kaggle_input / "qm9",
            kaggle_input / "qm9-pyg",
            kaggle_input / "qm9csv",
        ]:
            if candidate.exists():
                return str(candidate)

    if os.path.exists("/kaggle/working"):
        return "/kaggle/working/data/QM9"

    return "./data/QM9"


def get_model_device(model_key, requested_device="auto"):
    """
    Determine target device.
    In 'auto' mode:
      - Classical models (classical_twin, schnet, painn, egnn_pm, egnn_std) use CUDA if available.
      - EQ-GNN V2 uses CPU (where PennyLane default.qubit state-vector simulation runs stably).
    """
    if requested_device == "cpu":
        return torch.device("cpu")
    elif requested_device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif requested_device == "auto":
        if model_key == "eqgnn_v2":
            return torch.device("cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(requested_device)


# Single Run: Train + Evaluate

def train_eval_single_run(model_key, target, seed,
                          num_train=1000, num_val=250, num_test=500,
                          epochs=15, batch_size=16, lr=1e-3, hidden_dim=16,
                          device=None, data_dir=None):
    """Train and evaluate a single model on a single target with a single seed."""
    if device is None:
        device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    resolved_data_dir = resolve_data_dir(data_dir)

    train_set, val_set, test_set, (mean, std) = get_qm9_splits(
        root=resolved_data_dir,
        target=target,
        num_train=num_train,
        num_val=num_val,
        num_test=num_test,
        seed=seed,
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    torch.manual_seed(seed)

    model, total_params = create_model(model_key, hidden_dim=hidden_dim)
    model = model.to(device)

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
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
    print(f"    Seed {seed} ({device}) | Test MAE = {test_mae:.4f} {TARGET_UNITS[target]} "
          f"({elapsed:.1f}s, {total_params:,} params)")
    return test_mae, total_params


# Main Benchmark Loop

def run_benchmark(targets=None, seeds=None, models=None,
                  num_train=1000, num_val=250, num_test=500,
                  epochs=15, batch_size=16, lr=1e-3, hidden_dim=16,
                  device="auto", data_dir=None, save_dir=None):
    if targets is None:
        targets = ["homo", "lumo", "gap", "mu", "alpha", "u0"]
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]
    if models is None:
        models = ["eqgnn_v2", "classical_twin", "trig_twin_siren", "trig_twin_fourier",
                  "schnet", "painn", "egnn_pm", "egnn_std"]

    if save_dir is None:
        save_dir = "/kaggle/working/results" if os.path.exists("/kaggle/working") else "./results"

    os.makedirs(save_dir, exist_ok=True)

    print("=" * 100)
    print("Extended Multi-Target Multi-Seed QM9 Benchmark (Table 1 Revision)")
    print(f"   Targets: {targets}")
    print(f"   Models: {[MODEL_CONFIGS[m]['label'] for m in models]}")
    print(f"   Seeds: {seeds} | N_train: {num_train} | Epochs: {epochs} | Batch size: {batch_size}")
    print(f"   Device mode: {device} | Save dir: {save_dir}")
    print("=" * 100)

    json_path = os.path.join(save_dir, "extended_table1_v2.json")
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
        print(f"\n{'='*100}")
        print(f"Target: {TARGET_DISPLAY[target]} [{TARGET_UNITS[target]}] ({t_type})")
        print(f"{'='*100}")

        if target not in results:
            results[target] = {
                "unit": TARGET_UNITS[target],
                "display": TARGET_DISPLAY[target],
                "type": t_type,
            }

        for model_key in models:
            # Skip if already completed
            if model_key in results[target] and "maes" in results[target][model_key]:
                existing = results[target][model_key]
                print(f"\nModel: {MODEL_CONFIGS[model_key]['label']} "
                      f"— CACHED: {existing['mean_mae']:.4f} ± {existing['std_mae']:.4f}")
                continue

            config = MODEL_CONFIGS[model_key]
            model_dev = get_model_device(model_key, device)
            print(f"\nModel: {config['label']} (Tier {config['tier']}) on {model_dev}")
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

            # Save checkpoint after each model×target
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2)

        # Statistical tests: EQ-GNN V2 vs each baseline
        if "eqgnn_v2" in results[target] and "maes" in results[target]["eqgnn_v2"]:
            stat_tests = {}
            eqgnn_maes = results[target]["eqgnn_v2"]["maes"]
            for baseline in models:
                if baseline != "eqgnn_v2" and baseline in results[target] and "maes" in results[target][baseline]:
                    stat_tests[f"eqgnn_v2_vs_{baseline}"] = compute_statistical_tests(
                        eqgnn_maes, results[target][baseline]["maes"]
                    )
            results[target]["statistical_tests"] = stat_tests

            print(f"\n  Statistical Significance ({target}):")
            for key, st in stat_tests.items():
                sig_marker = "**" if st["significant_001"] else ("*" if st["significant_005"] else "ns")
                print(f"     {key}: Wilcoxon p={st['wilcoxon_p']:.4f}, "
                      f"t-test p={st['ttest_p']:.4f}, "
                      f"Δ={st['mean_diff']:.4f} {sig_marker}")

    # Save final results
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Print Publication Table
    print("\n" + "=" * 140)
    print(f"EXTENDED TABLE 1: Multi-Target QM9 Results (N={num_train:,}, Mean ± Std over {len(seeds)} Seeds)")
    print("=" * 140)

    # Header
    header = f"| {'Target':<28} | {'Unit':<6} |"
    for m in models:
        header += f" {MODEL_CONFIGS[m]['short'] + ' (' + str(results[targets[0]].get(m, {}).get('params', '?')) + ')':<20} |"
    print(header)
    divider = f"|{'-'*30}|{'-'*8}|"
    for _ in models:
        divider += f"{'-'*22}|"
    print(divider)

    # Rows
    for t in targets:
        d = results[t]
        row = f"| {d['display']:<28} | {d['unit']:<6} |"
        # Find best MAE for this target
        valid_maes = {m: d[m]['mean_mae'] for m in models if m in d and 'mean_mae' in d.get(m, {})}
        best_mae = min(valid_maes.values()) if valid_maes else float('inf')

        for m in models:
            if m in d and 'mean_mae' in d.get(m, {}):
                entry = d[m]
                mae_str = f"{entry['mean_mae']:.4f} ± {entry['std_mae']:.4f}"
                if entry['mean_mae'] == best_mae:
                    mae_str = f"**{mae_str}**"
                row += f" {mae_str:<20} |"
            else:
                row += f" {'---':<20} |"
        print(row)

    print(divider)
    print("=" * 140)

    # Significance summary table
    print("\nStatistical Significance: EQ-GNN V2 vs Each Baseline (negative Δ = EQ-GNN is better)")
    print(f"| {'Target':<20} |", end="")
    baselines = [m for m in models if m != "eqgnn_v2"]
    for b in baselines:
        print(f" {MODEL_CONFIGS[b]['short']:<14} |", end="")
    print()

    for t in targets:
        st = results[t].get("statistical_tests", {})
        row = f"| {t:<20} |"
        for b in baselines:
            key = f"eqgnn_v2_vs_{b}"
            if key in st:
                d = st[key]
                sig = "**" if d["significant_001"] else ("*" if d["significant_005"] else "ns")
                row += f" {d['mean_diff']:+.4f} {sig:<5} |"
            else:
                row += f" {'---':<14} |"
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extended QM9 Baseline Benchmark")
    parser.add_argument("--targets", nargs="+",
                        default=["homo", "lumo", "gap", "mu", "alpha", "u0"],
                        choices=list(QM9_TARGET_INDICES.keys()))
    parser.add_argument("--models", nargs="+",
                        default=["eqgnn_v2", "classical_twin", "trig_twin_siren", "trig_twin_fourier",
                                 "schnet", "painn", "egnn_pm", "egnn_std"],
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--num-train", type=int, default=1000)
    parser.add_argument("--num-val", type=int, default=250)
    parser.add_argument("--num-test", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Device to use ('auto' routes classical models to CUDA and EQ-GNN to CPU)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="QM9 data directory (auto-resolves on Kaggle if omitted)")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Results output directory (defaults to /kaggle/working/results on Kaggle, else ./results)")
    args = parser.parse_args()

    run_benchmark(
        targets=args.targets,
        seeds=args.seeds,
        models=args.models,
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
