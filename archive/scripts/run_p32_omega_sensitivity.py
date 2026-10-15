"""SIREN omega_0 sensitivity analysis across frequency scales."""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from datasets.dataset_qm9 import get_qm9_splits, QM9_TARGET_INDICES
from models.egnn_baseline import count_parameters
from models.classical_twin import ClassicalTwinModelV2
from models.classical_trig_twin import PRISMModelV2, TrigTwinModelV2


TARGET_UNITS = {"homo": "eV", "lumo": "eV", "gap": "eV",
                "mu": "Debye", "alpha": "Bohr³", "u0": "eV"}


def create_siren_model(omega_0, in_features=11, hidden_dim=16, num_layers=2, out_dim=1):
    """Create PRISM (SIREN) model with a specific omega_0 value."""
    model = PRISMModelV2(
        in_features=in_features,
        hidden_features=hidden_dim,
        num_layers=num_layers,
        n_q_layers=2,
        out_dim=out_dim,
        head_type="siren",
        omega_0=omega_0,
    )
    return model, count_parameters(model)


def create_classical_twin(in_features=11, hidden_dim=16, num_layers=2, out_dim=1):
    """Create Classical Twin (tanh-MLP) baseline model."""
    model = ClassicalTwinModelV2(
        in_features=in_features,
        hidden_features=hidden_dim,
        num_layers=num_layers,
        n_q_layers=2,
        out_dim=out_dim,
    )
    return model, count_parameters(model)


def train_eval(model, target, seed, num_train=1000, epochs=15,
               batch_size=16, lr=1e-3, data_dir=None):
    """Train and evaluate a model, returning test MAE."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = data_dir or "./data/QM9"

    train_set, val_set, test_set, (mean, std) = get_qm9_splits(
        root=data_dir, target=target,
        num_train=num_train, num_val=250, num_test=500, seed=seed,
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    torch.manual_seed(seed)
    model = model.to(device)
    params = count_parameters(model)

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

    # Evaluate
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

    return float(test_mae), params, round(elapsed, 1)


def run_omega_sensitivity(omega_values, targets, seeds, epochs, batch_size,
                          lr, data_dir, save_dir):
    """Run the ω₀ sensitivity ablation."""
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, "p32_omega_sensitivity.json")

    # Load existing results for resumption
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            results = json.load(f)
    else:
        results = {"metadata": {
            "description": "P3.2 Revision: SIREN ω₀ sensitivity analysis",
            "omega_values": omega_values,
            "targets": targets,
            "seeds": seeds,
        }}

    print("=" * 80)
    print("SIREN omega_0 Sensitivity Analysis")
    print(f"   omega_0 values: {omega_values}")
    print(f"   Targets:        {targets}")
    print(f"   Seeds:          {seeds}")
    print(f"   Epochs:         {epochs}")
    print("=" * 80)

    for target in targets:
        if target not in results:
            results[target] = {}

        print(f"\nTarget: {target.upper()} ({TARGET_UNITS[target]})")

        # Run Classical Twin baseline for reference
        for seed in seeds:
            baseline_key = f"classical_twin_seed{seed}"
            if baseline_key not in results[target]:
                print(f"\n  Classical Twin (tanh) baseline | seed={seed}")
                model, params = create_classical_twin()
                mae, _, elapsed = train_eval(model, target, seed,
                                              epochs=epochs, batch_size=batch_size,
                                              lr=lr, data_dir=data_dir)
                results[target][baseline_key] = {
                    "test_mae": mae, "params": params, "time_s": elapsed,
                }
                print(f"    Classical Twin | MAE = {mae:.4f} {TARGET_UNITS[target]} | {elapsed}s")

                # Checkpoint
                with open(json_path, "w") as f:
                    json.dump(results, f, indent=2)
            else:
                cached = results[target][baseline_key]
                print(f"  Cached: Classical Twin seed={seed} — {cached['test_mae']:.4f}")

        # Run SIREN at each omega_0
        for omega in omega_values:
            for seed in seeds:
                run_key = f"siren_omega{omega}_seed{seed}"

                if run_key in results[target]:
                    cached = results[target][run_key]
                    print(f"  Cached: SIREN omega_0={omega} seed={seed} — {cached['test_mae']:.4f}")
                    continue

                print(f"\n  Periodic-EGNN (SIREN) omega_0={omega} | seed={seed}")
                model, params = create_siren_model(omega_0=omega)
                mae, _, elapsed = train_eval(model, target, seed,
                                              epochs=epochs, batch_size=batch_size,
                                              lr=lr, data_dir=data_dir)
                results[target][run_key] = {
                    "test_mae": mae, "params": params, "omega_0": omega,
                    "time_s": elapsed,
                }
                print(f"    SIREN omega_0={omega} | MAE = {mae:.4f} {TARGET_UNITS[target]} | {elapsed}s")

                # Checkpoint
                with open(json_path, "w") as f:
                    json.dump(results, f, indent=2)

    # Summary table
    print("\n" + "=" * 80)
    print("Omega_0 Sensitivity Results")
    print("=" * 80)

    for target in targets:
        print(f"\n  Target: {target.upper()} ({TARGET_UNITS[target]})")

        # Classical Twin baseline
        baseline_maes = []
        for seed in seeds:
            key = f"classical_twin_seed{seed}"
            if key in results[target]:
                baseline_maes.append(results[target][key]["test_mae"])
        if baseline_maes:
            baseline_mean = np.mean(baseline_maes)
            baseline_str = f"{baseline_mean:.4f}" + (f" ± {np.std(baseline_maes):.4f}" if len(baseline_maes) > 1 else "")
            print(f"  Classical Twin (tanh): {baseline_str}")
        else:
            baseline_mean = None

        print(f"  {'omega_0':>8} | {'MAE':>20} | {'vs tanh':>10}")
        print(f"  {'-'*8}-+{'-'*22}+{'-'*12}")

        for omega in omega_values:
            maes = []
            for seed in seeds:
                key = f"siren_omega{omega}_seed{seed}"
                if key in results[target]:
                    maes.append(results[target][key]["test_mae"])
            if maes:
                mean_mae = np.mean(maes)
                mae_str = f"{mean_mae:.4f}" + (f" ± {np.std(maes):.4f}" if len(maes) > 1 else "")
                delta = f"{((mean_mae / baseline_mean - 1) * 100):+.1f}%" if baseline_mean else "-"
                marker = " <- default" if omega == 30.0 else ""
                print(f"  {omega:>8.0f} | {mae_str:>20} | {delta:>10}{marker}")

    print(f"\nResults saved to: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIREN omega_0 Sensitivity Analysis")
    parser.add_argument("--omega-values", nargs="+", type=float, default=[10.0, 20.0, 30.0, 50.0],
                        help="omega_0 values to test")
    parser.add_argument("--targets", nargs="+", default=["homo"],
                        choices=list(QM9_TARGET_INDICES.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42],
                        help="Random seeds")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick validation (2 epochs, omega_0=[10,30] only)")
    args = parser.parse_args()

    if args.dry_run:
        args.omega_values = [10.0, 30.0]
        args.seeds = [42]
        args.epochs = 2
        args.targets = ["homo"]
        print("DRY RUN mode: 2 epochs, omega_0=[10,30], seed=42, HOMO only")

    run_omega_sensitivity(
        omega_values=args.omega_values,
        targets=args.targets,
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
    )
