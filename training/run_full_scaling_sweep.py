"""
Full-Dataset Scaling Sweep: QM9 + MD17 + N-Body
=================================================
Addresses Reviewer Issue M-5: "Insufficient Data Scale for Robust Claims"

Runs scaling experiments across all three benchmarks using the FULL datasets,
evaluates multiple targets, and records exact split indices for reproducibility.

Usage:
    # Run everything (all benchmarks, all targets, all N values, all seeds)
    pixi run python training/run_full_scaling_sweep.py --benchmark all

    # Run only QM9, all targets
    pixi run python training/run_full_scaling_sweep.py --benchmark qm9

    # Run only QM9 HOMO target
    pixi run python training/run_full_scaling_sweep.py --benchmark qm9 --targets homo

    # Run only QM9 HOMO at specific N values
    pixi run python training/run_full_scaling_sweep.py --benchmark qm9 --targets homo --n-values 100 500 1000

    # Run only MD17
    pixi run python training/run_full_scaling_sweep.py --benchmark md17

    # Run only N-body
    pixi run python training/run_full_scaling_sweep.py --benchmark nbody

    # Run with specific seeds
    pixi run python training/run_full_scaling_sweep.py --benchmark qm9 --seeds 42 43

    # Dry run (2 epochs, 1 seed, smallest N only) to verify pipeline
    pixi run python training/run_full_scaling_sweep.py --benchmark qm9 --dry-run
"""

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


# =============================================================================
# Imports from project modules
# =============================================================================
from datasets.dataset_qm9 import get_qm9_splits, QM9_TARGET_INDICES
from datasets.dataset_md17 import get_md17_splits
from datasets.dataset_nbody import get_nbody_splits
from models.eq_gnn import QPRISMModelV2, EQGNNModelV2
from models.egnn_baseline import (
    EGNNModel,
    create_param_matched_egnn,
    count_parameters,
)
from models.classical_twin import ClassicalTwinModelV2
from models.classical_baselines import SchNetModel, PaiNNModel

# N-body trajectory models (imported from train_nbody)
from training.train_nbody import (
    QPRISMTrajectoryModel,
    EQGNNTrajectoryModel,
    EGNNTrajectoryModel,
    NonEquivTrajectoryModel,
    ParamMatchedEGNNTrajectoryModel,
)
from models.classical_twin import ClassicalTwinTrajectoryModel

# MD17 non-equivariant model
from training.train_md17 import NonEquivMD17Model


# =============================================================================
# Configuration Constants
# =============================================================================

# QM9 scaling configuration
QM9_N_VALUES = [100, 250, 500, 1000, 2500, 5000, 10000]
QM9_TARGETS = ["homo", "lumo", "gap", "mu", "u0"]
QM9_MODELS = ["eqgnn_v2", "classical_twin", "schnet", "painn", "egnn_pm", "egnn_std"]
QM9_TEST_SIZE = 10000   # Large fixed test set from the full 130K dataset
QM9_VAL_SIZE = 2000     # Large fixed validation set
QM9_EPOCHS = 20
QM9_BATCH_SIZE = 64
QM9_LR = 1e-3
QM9_HIDDEN_DIM = 16

# MD17 scaling configuration
MD17_N_VALUES = [1000, 5000, 10000, 50000]
MD17_MOLECULES = ["ethanol", "aspirin"]
MD17_MODELS = ["eqgnn", "egnn_pm", "egnn", "noneq"]
MD17_TEST_SIZE = 10000
MD17_VAL_SIZE = 2000
MD17_EPOCHS = 30
MD17_BATCH_SIZE = 16
MD17_LR = 5e-4
MD17_HIDDEN_DIM = 16

# N-Body scaling configuration
NBODY_N_VALUES = [1000, 3000, 5000, 10000]
NBODY_MODELS = ["eqgnn", "egnn_pm", "egnn", "noneq"]
NBODY_TEST_SIZE = 2000
NBODY_VAL_SIZE = 500
NBODY_EPOCHS = 15
NBODY_BATCH_SIZE = 16
NBODY_LR = 1e-3
NBODY_HIDDEN_DIM = 16

# Common
SEEDS = [42, 43, 44, 45, 46]

# Target metadata
TARGET_UNITS = {
    "homo": "eV", "lumo": "eV", "gap": "eV",
    "mu": "Debye", "alpha": "Bohr\u00b3", "u0": "eV",
}
TARGET_DISPLAY = {
    "homo": "HOMO Energy (\u03b5_HOMO)", "lumo": "LUMO Energy (\u03b5_LUMO)",
    "gap": "HOMO-LUMO Gap (\u0394\u03b5)", "mu": "Dipole Moment (\u03bc)",
    "alpha": "Polarizability (\u03b1)", "u0": "Internal Energy (U\u2080)",
}


# =============================================================================
# Statistical Tests (reused from existing codebase)
# =============================================================================
def compute_statistical_tests(maes_a, maes_b):
    """
    Compute Wilcoxon signed-rank and paired t-test between two sets of MAE values.
    """
    a = np.array(maes_a, dtype=np.float64)
    b = np.array(maes_b, dtype=np.float64)
    result = {}

    try:
        if len(a) >= 5:
            _, p_wilcoxon = stats.wilcoxon(a, b)
        else:
            p_wilcoxon = float("nan")
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


# =============================================================================
# Checkpointing Utilities
# =============================================================================
def load_checkpoint(json_path):
    """Load existing results JSON, or return empty dict if not found."""
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_checkpoint(results, json_path):
    """Save results to JSON with checkpointing."""
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)


def make_run_key(model, target_or_molecule, n_train, seed):
    """Create a unique key for a single (model, target, N, seed) run."""
    return f"{model}|{target_or_molecule}|N{n_train}|seed{seed}"


def is_run_completed(results, run_key):
    """Check if a specific run has already been completed."""
    return run_key in results.get("_completed_runs", {})


def mark_run_completed(results, run_key, result_value):
    """Mark a run as completed with its result value."""
    if "_completed_runs" not in results:
        results["_completed_runs"] = {}
    results["_completed_runs"][run_key] = result_value


# =============================================================================
# Split Index Recording
# =============================================================================
def record_split_indices(split_record, benchmark, key, seed, train_indices, val_indices, test_indices):
    """Record the exact split indices for reproducibility."""
    if benchmark not in split_record:
        split_record[benchmark] = {}
    rec_key = f"{key}|seed{seed}"
    split_record[benchmark][rec_key] = {
        "seed": seed,
        "num_train": len(train_indices),
        "num_val": len(val_indices),
        "num_test": len(test_indices),
        "train_indices_first10": [int(i) for i in train_indices[:10]],
        "train_indices_last5": [int(i) for i in train_indices[-5:]],
        "test_indices_first10": [int(i) for i in test_indices[:10]],
    }


def save_split_indices(split_record, save_dir):
    """Save split indices to a separate JSON file."""
    path = os.path.join(save_dir, "scaling_split_indices.json")
    save_checkpoint(split_record, path)
    print(f"\n\U0001f4cb Saved split index summary to: {path}")


# =============================================================================
# QM9 Scaling Sweep
# =============================================================================
def get_qm9_split_indices(total_len, num_train, num_val, num_test, seed):
    """
    Generate QM9 split indices deterministically.
    Returns (train_indices, val_indices, test_indices) as lists of ints.
    """
    torch.manual_seed(seed)
    perm = torch.randperm(total_len)
    train_idx = perm[:num_train].tolist()
    val_idx = perm[num_train:num_train + num_val].tolist()
    test_idx = perm[num_train + num_val:num_train + num_val + num_test].tolist()
    return train_idx, val_idx, test_idx


def create_qm9_model(model_key, hidden_dim=16):
    """Instantiate a QM9 model by key. Returns (model, param_count)."""
    device = torch.device("cpu")
    if model_key == "eqgnn_v2":
        model = EQGNNModelV2(
            in_features=11, hidden_features=hidden_dim,
            num_layers=2, n_q_layers=2, out_dim=1
        ).to(device)
    elif model_key == "egnn_pm":
        ref_model = EQGNNModelV2(
            in_features=11, hidden_features=hidden_dim,
            num_layers=2, n_q_layers=2, out_dim=1
        )
        ref_params = count_parameters(ref_model)
        model, _ = create_param_matched_egnn(
            target_params=ref_params, in_features=11, num_layers=2, out_dim=1
        )
        model = model.to(device)
        del ref_model
    elif model_key == "egnn_std":
        model = EGNNModel(
            in_features=11, hidden_features=hidden_dim,
            num_layers=2, out_dim=1
        ).to(device)
    elif model_key == "classical_twin":
        model = ClassicalTwinModelV2(
            in_features=11, hidden_features=hidden_dim,
            num_layers=2, n_q_layers=2, out_dim=1
        ).to(device)
    elif model_key == "schnet":
        model = SchNetModel(
            in_features=11, hidden_channels=12,
            num_interactions=2, num_gaussians=50,
            cutoff=10.0, out_dim=1
        ).to(device)
    elif model_key == "painn":
        model = PaiNNModel(
            in_features=11, hidden_channels=8,
            num_interactions=2, n_rbf=20,
            cutoff=5.0, out_dim=1
        ).to(device)
    else:
        raise ValueError(f"Unknown QM9 model key: {model_key}")

    return model, count_parameters(model)


def train_eval_qm9_single(model_key, target, num_train, seed,
                           num_val=QM9_VAL_SIZE, num_test=QM9_TEST_SIZE,
                           epochs=QM9_EPOCHS, batch_size=QM9_BATCH_SIZE,
                           lr=QM9_LR, hidden_dim=QM9_HIDDEN_DIM):
    """Train and evaluate a single QM9 run. Returns (test_mae, param_count, elapsed_time)."""
    device = torch.device("cpu")

    # Load data with the specified split
    train_set, val_set, test_set, (mean, std) = get_qm9_splits(
        root="./data/QM9", target=target,
        num_train=num_train, num_val=num_val, num_test=num_test, seed=seed
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Seed model initialization
    torch.manual_seed(seed)

    model, param_count = create_qm9_model(model_key, hidden_dim)
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

    return test_mae, param_count, elapsed


def run_qm9_scaling_sweep(targets=None, n_values=None, models=None, seeds=None,
                           epochs=QM9_EPOCHS, save_dir="./results"):
    """Run the full QM9 scaling sweep with checkpointing."""
    if targets is None:
        targets = QM9_TARGETS
    if n_values is None:
        n_values = QM9_N_VALUES
    if models is None:
        models = QM9_MODELS
    if seeds is None:
        seeds = SEEDS

    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, "qm9_scaling_sweep.json")
    results = load_checkpoint(json_path)
    split_record = load_checkpoint(os.path.join(save_dir, "scaling_split_indices.json"))

    total_runs = len(targets) * len(n_values) * len(models) * len(seeds)
    completed = 0
    skipped = 0

    print("=" * 85)
    print("\U0001f31f QM9 Full-Dataset Scaling Sweep")
    print(f"   Targets: {targets}")
    print(f"   N values: {n_values}")
    print(f"   Models: {models}")
    print(f"   Seeds: {seeds} | Epochs: {epochs}")
    print(f"   Test set: {QM9_TEST_SIZE} | Val set: {QM9_VAL_SIZE}")
    print(f"   Total runs: {total_runs}")
    print("=" * 85)

    for target in targets:
        if target not in results:
            results[target] = {"unit": TARGET_UNITS[target], "display": TARGET_DISPLAY[target]}

        for n_train in n_values:
            n_key = f"N{n_train}"
            if n_key not in results[target]:
                results[target][n_key] = {}

            for model_key in models:
                if model_key not in results[target][n_key]:
                    results[target][n_key][model_key] = {"maes": [], "seeds_completed": [], "params": 0}

                model_data = results[target][n_key][model_key]

                for seed in seeds:
                    run_key = make_run_key(model_key, f"{target}_N{n_train}", n_train, seed)

                    if is_run_completed(results, run_key):
                        skipped += 1
                        continue

                    print(f"\n\u25b6 [{target}] {model_key} | N={n_train:,} | Seed={seed}")

                    try:
                        mae, params, elapsed = train_eval_qm9_single(
                            model_key=model_key, target=target,
                            num_train=n_train, seed=seed, epochs=epochs
                        )

                        model_data["maes"].append(mae)
                        model_data["seeds_completed"].append(seed)
                        model_data["params"] = params

                        mark_run_completed(results, run_key, {
                            "mae": mae, "params": params, "time": elapsed
                        })

                        # Record split indices
                        train_idx, val_idx, test_idx = get_qm9_split_indices(
                            130831, n_train, QM9_VAL_SIZE, QM9_TEST_SIZE, seed
                        )
                        record_split_indices(split_record, "qm9", f"{target}_N{n_train}", seed,
                                             train_idx, val_idx, test_idx)

                        completed += 1
                        print(f"    \u2713 MAE = {mae:.4f} {TARGET_UNITS[target]} | "
                              f"Params = {params:,} | {elapsed:.1f}s | "
                              f"[{completed}/{total_runs - skipped} done, {skipped} skipped]")

                        # Checkpoint after every run
                        save_checkpoint(results, json_path)

                    except Exception as e:
                        print(f"    \u2717 ERROR: {e}")
                        import traceback
                        traceback.print_exc()

                # Compute aggregate statistics if all seeds are done
                if len(model_data["maes"]) == len(seeds):
                    model_data["mean_mae"] = float(np.mean(model_data["maes"]))
                    model_data["std_mae"] = float(np.std(model_data["maes"]))

            # Statistical tests for this (target, N) combination
            if all(m in results[target][n_key] for m in models):
                eqgnn_data = results[target][n_key].get("eqgnn_v2", {})
                if len(eqgnn_data.get("maes", [])) == len(seeds):
                    stat_tests = {}
                    for baseline in models:
                        if baseline != "eqgnn_v2":
                            bl_data = results[target][n_key].get(baseline, {})
                            if len(bl_data.get("maes", [])) == len(seeds):
                                stat_tests[f"eqgnn_v2_vs_{baseline}"] = compute_statistical_tests(
                                    eqgnn_data["maes"], bl_data["maes"]
                                )
                    results[target][n_key]["statistical_tests"] = stat_tests

    # Final save
    save_checkpoint(results, json_path)

    if split_record:
        save_split_indices(split_record, save_dir)

    # Print summary table
    print_qm9_summary(results, targets, n_values, models)

    return results


def print_qm9_summary(results, targets, n_values, models):
    """Print a nicely formatted summary table."""
    print("\n" + "=" * 100)
    print("\U0001f4cb QM9 SCALING SWEEP SUMMARY")
    print("=" * 100)

    for target in targets:
        if target not in results:
            continue
        unit = TARGET_UNITS.get(target, "")
        print(f"\n--- {TARGET_DISPLAY.get(target, target)} [{unit}] ---")
        header = f"  {'N':>7} |"
        for m in models:
            header += f" {m:>22} |"
        print(header)
        print("  " + "-" * (10 + 25 * len(models)))

        for n_train in n_values:
            n_key = f"N{n_train}"
            if n_key not in results[target]:
                continue
            row = f"  {n_train:>7} |"
            for m in models:
                md = results[target][n_key].get(m, {})
                if "mean_mae" in md:
                    row += f" {md['mean_mae']:.4f} \u00b1 {md['std_mae']:.4f}   |"
                elif len(md.get("maes", [])) > 0:
                    row += f" {np.mean(md['maes']):.4f} (partial)   |"
                else:
                    row += f" {'---':>22} |"
            print(row)

    print("=" * 100)


# =============================================================================
# MD17 Scaling Sweep
# =============================================================================
def create_md17_model(model_type, in_features=10, hidden_dim=16, num_layers=2):
    """Instantiate an MD17 model by type. Returns (model, param_count)."""
    device = torch.device("cpu")
    if model_type == "eqgnn":
        model = EQGNNModelV2(
            in_features=in_features, hidden_features=hidden_dim,
            num_layers=num_layers, out_dim=1, n_q_layers=2
        ).to(device)
    elif model_type == "egnn_pm":
        model = EGNNModel(
            in_features=in_features, hidden_features=8,
            num_layers=num_layers, out_dim=1
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
    else:
        raise ValueError(f"Unknown MD17 model type: {model_type}")

    return model, count_parameters(model)


def train_eval_md17_single(model_type, molecule, num_train, seed,
                            num_val=MD17_VAL_SIZE, num_test=MD17_TEST_SIZE,
                            epochs=MD17_EPOCHS, batch_size=MD17_BATCH_SIZE,
                            lr=MD17_LR, hidden_dim=MD17_HIDDEN_DIM):
    """Train and evaluate a single MD17 run. Returns (test_mae, param_count, elapsed_time)."""
    device = torch.device("cpu")

    train_data, val_data, test_data, e_mean, e_std = get_md17_splits(
        molecule_name=molecule, num_train=num_train,
        num_val=num_val, num_test=num_test, seed=seed, k_nn=8
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    torch.manual_seed(seed)
    np.random.seed(seed)

    model, param_count = create_md17_model(model_type, in_features=10, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.MSELoss()

    start_time = time.time()
    best_val_mae = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            y_norm = batch.energy_norm
            pred = model(batch.x, batch.pos, batch.edge_index, batch.batch)
            loss = criterion(pred, y_norm.unsqueeze(-1) if y_norm.dim() == 1 else y_norm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
        scheduler.step()

        # Validation checkpoint every 5 epochs
        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            val_mae = 0.0
            val_count = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    pred_norm = model(batch.x, batch.pos, batch.edge_index, batch.batch)
                    pred_energy = pred_norm.squeeze() * max(e_std, 1e-8) + e_mean
                    true_energy = batch.energy.squeeze()
                    val_mae += torch.abs(pred_energy - true_energy).sum().item()
                    val_count += batch.num_graphs
            val_mae /= max(val_count, 1)
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Reload best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Test evaluation
    model.eval()
    total_mae = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred_norm = model(batch.x, batch.pos, batch.edge_index, batch.batch)
            pred_energy = pred_norm.squeeze() * max(e_std, 1e-8) + e_mean
            true_energy = batch.energy.squeeze()
            total_mae += torch.abs(pred_energy - true_energy).sum().item()
            total_samples += batch.num_graphs

    test_mae = total_mae / max(total_samples, 1)
    elapsed = time.time() - start_time

    return test_mae, param_count, elapsed


def run_md17_scaling_sweep(molecules=None, n_values=None, models=None, seeds=None,
                            epochs=MD17_EPOCHS, save_dir="./results"):
    """Run the full MD17 scaling sweep with checkpointing."""
    if molecules is None:
        molecules = MD17_MOLECULES
    if n_values is None:
        n_values = MD17_N_VALUES
    if models is None:
        models = MD17_MODELS
    if seeds is None:
        seeds = SEEDS

    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, "md17_scaling_sweep.json")
    results = load_checkpoint(json_path)

    total_runs = len(molecules) * len(n_values) * len(models) * len(seeds)
    completed = 0
    skipped = 0

    print("=" * 85)
    print("\U0001f31f MD17 Full-Dataset Scaling Sweep")
    print(f"   Molecules: {molecules}")
    print(f"   N values: {n_values}")
    print(f"   Models: {models}")
    print(f"   Seeds: {seeds} | Epochs: {epochs}")
    print(f"   Test set: {MD17_TEST_SIZE} | Val set: {MD17_VAL_SIZE}")
    print(f"   Total runs: {total_runs}")
    print("=" * 85)

    for mol in molecules:
        if mol not in results:
            results[mol] = {}

        for n_train in n_values:
            n_key = f"N{n_train}"
            if n_key not in results[mol]:
                results[mol][n_key] = {}

            for model_type in models:
                if model_type not in results[mol][n_key]:
                    results[mol][n_key][model_type] = {"maes": [], "seeds_completed": [], "params": 0}

                model_data = results[mol][n_key][model_type]

                for seed in seeds:
                    run_key = make_run_key(model_type, f"{mol}_N{n_train}", n_train, seed)

                    if is_run_completed(results, run_key):
                        skipped += 1
                        continue

                    print(f"\n\u25b6 [{mol}] {model_type} | N={n_train:,} | Seed={seed}")

                    try:
                        mae, params, elapsed = train_eval_md17_single(
                            model_type=model_type, molecule=mol,
                            num_train=n_train, seed=seed, epochs=epochs
                        )

                        model_data["maes"].append(mae)
                        model_data["seeds_completed"].append(seed)
                        model_data["params"] = params

                        mark_run_completed(results, run_key, {
                            "mae": mae, "params": params, "time": elapsed
                        })

                        completed += 1
                        print(f"    \u2713 MAE = {mae:.4f} kcal/mol | "
                              f"Params = {params:,} | {elapsed:.1f}s | "
                              f"[{completed}/{total_runs - skipped} done, {skipped} skipped]")

                        save_checkpoint(results, json_path)

                    except Exception as e:
                        print(f"    \u2717 ERROR: {e}")
                        import traceback
                        traceback.print_exc()

                # Compute aggregates
                if len(model_data["maes"]) == len(seeds):
                    model_data["mean_mae"] = float(np.mean(model_data["maes"]))
                    model_data["std_mae"] = float(np.std(model_data["maes"]))

    save_checkpoint(results, json_path)
    print_md17_summary(results, molecules, n_values, models)
    return results


def print_md17_summary(results, molecules, n_values, models):
    """Print a formatted MD17 summary table."""
    print("\n" + "=" * 100)
    print("\U0001f4cb MD17 SCALING SWEEP SUMMARY [kcal/mol]")
    print("=" * 100)

    for mol in molecules:
        if mol not in results:
            continue
        print(f"\n--- {mol.capitalize()} ---")
        header = f"  {'N':>7} |"
        for m in models:
            header += f" {m:>22} |"
        print(header)
        print("  " + "-" * (10 + 25 * len(models)))

        for n_train in n_values:
            n_key = f"N{n_train}"
            if n_key not in results[mol]:
                continue
            row = f"  {n_train:>7} |"
            for m in models:
                md = results[mol][n_key].get(m, {})
                if "mean_mae" in md:
                    row += f" {md['mean_mae']:.4f} \u00b1 {md['std_mae']:.4f}   |"
                elif len(md.get("maes", [])) > 0:
                    row += f" {np.mean(md['maes']):.4f} (partial)   |"
                else:
                    row += f" {'---':>22} |"
            print(row)

    print("=" * 100)


# =============================================================================
# N-Body Scaling Sweep
# =============================================================================
def create_nbody_model(model_type, hidden_dim=16, num_layers=2):
    """Instantiate an N-body trajectory model. Returns (model, param_count)."""
    device = torch.device("cpu")
    if model_type == "eqgnn":
        model = EQGNNTrajectoryModel(
            in_features=2, hidden_features=hidden_dim,
            num_layers=num_layers, n_q_layers=2
        ).to(device)
    elif model_type == "egnn_pm":
        model = ParamMatchedEGNNTrajectoryModel(
            in_features=2, hidden_features=8, num_layers=num_layers
        ).to(device)
    elif model_type == "egnn":
        model = EGNNTrajectoryModel(
            in_features=2, hidden_features=hidden_dim, num_layers=num_layers
        ).to(device)
    elif model_type == "noneq":
        model = NonEquivTrajectoryModel(
            in_features=2, hidden_features=hidden_dim, num_layers=num_layers
        ).to(device)
    else:
        raise ValueError(f"Unknown N-body model type: {model_type}")

    return model, count_parameters(model)


def train_eval_nbody_single(model_type, num_train, seed,
                             num_val=NBODY_VAL_SIZE, num_test=NBODY_TEST_SIZE,
                             epochs=NBODY_EPOCHS, batch_size=NBODY_BATCH_SIZE,
                             lr=NBODY_LR, hidden_dim=NBODY_HIDDEN_DIM):
    """Train and evaluate a single N-body run. Returns (test_mse, param_count, elapsed_time)."""
    device = torch.device("cpu")

    train_data, val_data, test_data = get_nbody_splits(
        num_train=num_train, num_val=num_val, num_test=num_test, seed=seed
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    torch.manual_seed(seed)

    model, param_count = create_nbody_model(model_type, hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.MSELoss()

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred_pos = model(batch.x, batch.pos, batch.edge_index)
            loss = criterion(pred_pos, batch.target_pos)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        scheduler.step()

    # Test evaluation
    model.eval()
    total_mse = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred_pos = model(batch.x, batch.pos, batch.edge_index)
            mse = torch.sum((pred_pos - batch.target_pos) ** 2, dim=-1).mean().item()
            total_mse += mse * batch.num_graphs
            total_samples += batch.num_graphs

    test_mse = total_mse / max(total_samples, 1)
    elapsed = time.time() - start_time

    return test_mse, param_count, elapsed


def run_nbody_scaling_sweep(n_values=None, models=None, seeds=None,
                             epochs=NBODY_EPOCHS, save_dir="./results"):
    """Run the full N-body scaling sweep with checkpointing."""
    if n_values is None:
        n_values = NBODY_N_VALUES
    if models is None:
        models = NBODY_MODELS
    if seeds is None:
        seeds = SEEDS

    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, "nbody_scaling_sweep.json")
    results = load_checkpoint(json_path)

    total_runs = len(n_values) * len(models) * len(seeds)
    completed = 0
    skipped = 0

    print("=" * 85)
    print("\U0001f31f N-Body Full-Dataset Scaling Sweep")
    print(f"   N values: {n_values}")
    print(f"   Models: {models}")
    print(f"   Seeds: {seeds} | Epochs: {epochs}")
    print(f"   Test set: {NBODY_TEST_SIZE} | Val set: {NBODY_VAL_SIZE}")
    print(f"   Total runs: {total_runs}")
    print("=" * 85)

    for n_train in n_values:
        n_key = f"N{n_train}"
        if n_key not in results:
            results[n_key] = {}

        for model_type in models:
            if model_type not in results[n_key]:
                results[n_key][model_type] = {"mses": [], "seeds_completed": [], "params": 0}

            model_data = results[n_key][model_type]

            for seed in seeds:
                run_key = make_run_key(model_type, f"nbody_N{n_train}", n_train, seed)

                if is_run_completed(results, run_key):
                    skipped += 1
                    continue

                print(f"\n\u25b6 [nbody] {model_type} | N={n_train:,} | Seed={seed}")

                try:
                    mse, params, elapsed = train_eval_nbody_single(
                        model_type=model_type, num_train=n_train,
                        seed=seed, epochs=epochs
                    )

                    model_data["mses"].append(mse)
                    model_data["seeds_completed"].append(seed)
                    model_data["params"] = params

                    mark_run_completed(results, run_key, {
                        "mse": mse, "params": params, "time": elapsed
                    })

                    completed += 1
                    print(f"    \u2713 MSE = {mse:.6f} | "
                          f"Params = {params:,} | {elapsed:.1f}s | "
                          f"[{completed}/{total_runs - skipped} done, {skipped} skipped]")

                    save_checkpoint(results, json_path)

                except Exception as e:
                    print(f"    \u2717 ERROR: {e}")
                    import traceback
                    traceback.print_exc()

            # Compute aggregates
            if len(model_data["mses"]) == len(seeds):
                model_data["mean_mse"] = float(np.mean(model_data["mses"]))
                model_data["std_mse"] = float(np.std(model_data["mses"]))

    save_checkpoint(results, json_path)
    print_nbody_summary(results, n_values, models)
    return results


def print_nbody_summary(results, n_values, models):
    """Print a formatted N-body summary table."""
    print("\n" + "=" * 100)
    print("\U0001f4cb N-BODY SCALING SWEEP SUMMARY [Coordinate MSE]")
    print("=" * 100)

    header = f"  {'N':>7} |"
    for m in models:
        header += f" {m:>22} |"
    print(header)
    print("  " + "-" * (10 + 25 * len(models)))

    for n_train in n_values:
        n_key = f"N{n_train}"
        if n_key not in results:
            continue
        row = f"  {n_train:>7} |"
        for m in models:
            md = results[n_key].get(m, {})
            if "mean_mse" in md:
                row += f" {md['mean_mse']:.6f} \u00b1 {md['std_mse']:.6f} |"
            elif len(md.get("mses", [])) > 0:
                row += f" {np.mean(md['mses']):.6f} (partial) |"
            else:
                row += f" {'---':>22} |"
        print(row)

    print("=" * 100)


# =============================================================================
# Main CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Full-Dataset Scaling Sweep: QM9 + MD17 + N-Body",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run everything
  pixi run python training/run_full_scaling_sweep.py --benchmark all

  # Run only QM9 HOMO and LUMO
  pixi run python training/run_full_scaling_sweep.py --benchmark qm9 --targets homo lumo

  # Run only QM9 at specific N values
  pixi run python training/run_full_scaling_sweep.py --benchmark qm9 --n-values 100 500 1000

  # Run only MD17 ethanol
  pixi run python training/run_full_scaling_sweep.py --benchmark md17 --molecules ethanol

  # Dry run to verify pipeline
  pixi run python training/run_full_scaling_sweep.py --benchmark qm9 --dry-run
        """
    )

    parser.add_argument("--benchmark", type=str, default="all",
                        choices=["all", "qm9", "md17", "nbody"],
                        help="Which benchmark(s) to run")
    parser.add_argument("--targets", nargs="+", default=None,
                        help="QM9 targets (default: homo lumo gap u0)")
    parser.add_argument("--molecules", nargs="+", default=None,
                        help="MD17 molecules (default: ethanol aspirin)")
    parser.add_argument("--n-values", nargs="+", type=int, default=None,
                        help="Training sizes to sweep (overrides defaults for the chosen benchmark)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to evaluate (e.g. eqgnn_v2 egnn_pm egnn_std)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Random seeds (default: 42 43 44 45 46)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override default epoch count")
    parser.add_argument("--save-dir", type=str, default="./results",
                        help="Directory to save results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick smoke test: 2 epochs, 1 seed, smallest N only")

    args = parser.parse_args()

    # Dry run overrides
    if args.dry_run:
        args.seeds = [42]
        if args.epochs is None:
            args.epochs = 2
        print("\U0001f9ea DRY RUN MODE: 1 seed, 2 epochs, minimal N values")

    benchmarks = ["qm9", "md17", "nbody"] if args.benchmark == "all" else [args.benchmark]

    for bench in benchmarks:
        if bench == "qm9":
            n_vals = args.n_values or ([QM9_N_VALUES[0]] if args.dry_run else QM9_N_VALUES)
            run_qm9_scaling_sweep(
                targets=args.targets,
                n_values=n_vals,
                models=args.models,
                seeds=args.seeds,
                epochs=args.epochs or QM9_EPOCHS,
                save_dir=args.save_dir,
            )

        elif bench == "md17":
            n_vals = args.n_values or ([MD17_N_VALUES[0]] if args.dry_run else MD17_N_VALUES)
            run_md17_scaling_sweep(
                molecules=args.molecules,
                n_values=n_vals,
                models=args.models,
                seeds=args.seeds,
                epochs=args.epochs or MD17_EPOCHS,
                save_dir=args.save_dir,
            )

        elif bench == "nbody":
            n_vals = args.n_values or ([NBODY_N_VALUES[0]] if args.dry_run else NBODY_N_VALUES)
            run_nbody_scaling_sweep(
                n_values=n_vals,
                models=args.models,
                seeds=args.seeds,
                epochs=args.epochs or NBODY_EPOCHS,
                save_dir=args.save_dir,
            )

    print("\n" + "=" * 85)
    print("\u2705 All requested scaling sweeps completed!")
    print("   Run 'python visualization/plot_scaling_crossover.py' to generate figures.")
    print("=" * 85)


if __name__ == "__main__":
    main()
