"""
P2.1 Revision Experiment: Large-Scale PRISM Evaluation
=====================================================
Addresses Reviewer Concern P2.1 (R2, R3, R5):
  "Since PRISM is fully classical and computationally cheap,
   run it on a larger QM9 subset (e.g., N=10,000) to demonstrate
   whether the SIREN advantage persists beyond the low-data regime."

Models evaluated:
  - PRISM (SIREN, 2,527 params)
  - Classical Twin (tanh-MLP, 2,527 params)
  - Q-PRISM (4-qubit PQC, 2,527 params)

Targets: HOMO, LUMO, Gap
Train sizes: N ∈ {1000, 5000, 10000}
Seeds: 1 seed (42) for efficiency; add more if time permits

Usage:
    pixi run python training/run_p21_large_scale_eval.py
    pixi run python training/run_p21_large_scale_eval.py --n-values 1000 5000 10000 --seeds 42
    pixi run python training/run_p21_large_scale_eval.py --n-values 1000 10000 --seeds 42 43 44
    pixi run python training/run_p21_large_scale_eval.py --skip-eqgnn  # Classical models only (much faster)
    pixi run python training/run_p21_large_scale_eval.py --dry-run     # Quick validation (2 epochs)

Output:
    results/p21_large_scale_eval.json
"""

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
from models.eq_gnn import QPRISMModelV2, EQGNNModelV2
from models.egnn_baseline import count_parameters
from models.classical_twin import ClassicalTwinModelV2
from models.classical_trig_twin import PRISMModelV2, TrigTwinModelV2


TARGET_UNITS = {"homo": "eV", "lumo": "eV", "gap": "eV"}


def create_model(model_key, in_features=11, hidden_dim=16, num_layers=2, out_dim=1):
    """Create model by key. Returns (model, param_count)."""
    if model_key == "trig_twin_siren":
        model = PRISMModelV2(
            in_features=in_features,
            hidden_features=hidden_dim,
            num_layers=num_layers,
            n_q_layers=2,
            out_dim=out_dim,
            head_type="siren",
            omega_0=30.0,
        )
    elif model_key == "classical_twin":
        model = ClassicalTwinModelV2(
            in_features=in_features,
            hidden_features=hidden_dim,
            num_layers=num_layers,
            n_q_layers=2,
            out_dim=out_dim,
        )
    elif model_key == "eqgnn_v2":
        model = QPRISMModelV2(
            in_features=in_features,
            hidden_features=hidden_dim,
            num_layers=num_layers,
            n_q_layers=2,
            out_dim=out_dim,
        )
    else:
        raise ValueError(f"Unknown model key: {model_key}")
    return model, count_parameters(model)


def get_device(model_key):
    """Q-PRISM uses CPU (PennyLane); classical models use CUDA if available."""
    if model_key == "eqgnn_v2":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_eval(model_key, target, seed, num_train, epochs, batch_size=32,
               lr=1e-3, hidden_dim=16, data_dir=None):
    """Train and evaluate a single model configuration."""
    device = get_device(model_key)
    data_dir = data_dir or "./data/QM9"

    # Scale val/test proportionally but cap at reasonable sizes
    num_val = min(num_train // 4, 2000)
    num_test = min(num_train // 2, 5000)

    train_set, val_set, test_set, (mean, std) = get_qm9_splits(
        root=data_dir, target=target,
        num_train=num_train, num_val=num_val, num_test=num_test, seed=seed,
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
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            y_norm = (batch.y - mean) / std
            pred = model(batch.x, batch.pos, batch.edge_index, batch.batch)
            loss = criterion(pred, y_norm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()

        if epoch % 5 == 0 or epoch == epochs:
            avg_loss = epoch_loss / n_batches
            elapsed = time.time() - start_time
            print(f"      Epoch {epoch:3d}/{epochs} | Train Loss: {avg_loss:.6f} | {elapsed:.1f}s")

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

    print(f"    ✅ {model_key} | N={num_train} | seed={seed} | "
          f"Test MAE = {test_mae:.4f} {TARGET_UNITS[target]} | "
          f"{elapsed:.1f}s | {total_params:,} params")

    return {
        "test_mae": float(test_mae),
        "params": total_params,
        "train_time_s": round(elapsed, 1),
        "num_train": num_train,
        "num_test": total_samples,
        "epochs": epochs,
    }


def run_p21_evaluation(n_values, targets, seeds, models, epochs, batch_size,
                       lr, hidden_dim, data_dir, save_dir):
    """Run the full P2.1 large-scale evaluation."""
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, "p21_large_scale_eval.json")

    # Load existing results for resumption
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            results = json.load(f)
    else:
        results = {"metadata": {
            "description": "P2.1 Revision: Large-scale Periodic-EGNN evaluation",
            "n_values": n_values,
            "targets": targets,
            "seeds": seeds,
            "models": models,
        }}

    model_labels = {
        "trig_twin_siren": "PRISM (SIREN)",
        "classical_twin": "Classical Twin (tanh)",
        "eqgnn_v2": "Q-PRISM (PQC)",
    }

    print("=" * 90)
    print("🔬 P2.1 Revision: Large-Scale Periodic-EGNN Evaluation")
    print(f"   N values: {n_values}")
    print(f"   Targets:  {targets}")
    print(f"   Seeds:    {seeds}")
    print(f"   Models:   {[model_labels[m] for m in models]}")
    print(f"   Epochs:   {epochs} | Batch size: {batch_size}")
    print("=" * 90)

    for target in targets:
        if target not in results:
            results[target] = {}

        for n_train in n_values:
            n_key = f"N={n_train}"
            if n_key not in results[target]:
                results[target][n_key] = {}

            print(f"\n{'─'*70}")
            print(f"📦 Target: {target.upper()} | N_train = {n_train:,}")
            print(f"{'─'*70}")

            for model_key in models:
                for seed in seeds:
                    run_key = f"{model_key}_seed{seed}"

                    # Skip if already computed
                    if run_key in results[target][n_key]:
                        cached = results[target][n_key][run_key]
                        print(f"  ⏭  {model_labels[model_key]} seed={seed} — "
                              f"CACHED: {cached['test_mae']:.4f} {TARGET_UNITS[target]}")
                        continue

                    print(f"\n  ▶ {model_labels[model_key]} | seed={seed}")
                    result = train_eval(
                        model_key=model_key,
                        target=target,
                        seed=seed,
                        num_train=n_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        lr=lr,
                        hidden_dim=hidden_dim,
                        data_dir=data_dir,
                    )
                    results[target][n_key][run_key] = result

                    # Checkpoint after each run
                    with open(json_path, "w") as f:
                        json.dump(results, f, indent=2)

    # =========================================================================
    # Summary Table
    # =========================================================================
    print("\n" + "=" * 90)
    print("📋 P2.1 RESULTS SUMMARY")
    print("=" * 90)

    for target in targets:
        print(f"\n{'─'*70}")
        print(f"  Target: {target.upper()} ({TARGET_UNITS[target]})")
        print(f"{'─'*70}")

        header = f"  {'N':>8} |"
        for m in models:
            header += f" {model_labels[m]:>28} |"
        print(header)
        print(f"  {'─'*8}─┼" + "─" * 30 + "─┼" * (len(models) - 1) + "─" * 30 + "─┤")

        for n_train in n_values:
            n_key = f"N={n_train}"
            if n_key not in results[target]:
                continue

            row = f"  {n_train:>8,} |"
            for m in models:
                # Aggregate across seeds
                maes = []
                for seed in seeds:
                    run_key = f"{m}_seed{seed}"
                    if run_key in results[target][n_key]:
                        maes.append(results[target][n_key][run_key]["test_mae"])
                if maes:
                    mean_mae = np.mean(maes)
                    if len(maes) > 1:
                        std_mae = np.std(maes)
                        row += f" {mean_mae:>10.4f} ± {std_mae:.4f}       |"
                    else:
                        row += f" {mean_mae:>10.4f}                  |"
                else:
                    row += f" {'—':>28} |"
            print(row)

    print(f"\n📁 Results saved to: {json_path}")
    print("\n💡 Key question: Does PRISM maintain its advantage at N=10,000?")
    print("   If yes → dequantization result is robust, not a low-data artifact.")
    print("   If no  → the advantage is regime-dependent (also interesting to report).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P2.1 Large-Scale PRISM Evaluation")
    parser.add_argument("--n-values", nargs="+", type=int, default=[1000, 5000, 10000],
                        help="Training set sizes to evaluate")
    parser.add_argument("--targets", nargs="+", default=["homo", "lumo", "gap"],
                        choices=list(QM9_TARGET_INDICES.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42],
                        help="Random seeds (default: [42] for speed; add more for statistics)")
    parser.add_argument("--models", nargs="+",
                        default=["trig_twin_siren", "classical_twin", "eqgnn_v2"],
                        choices=["trig_twin_siren", "classical_twin", "eqgnn_v2"])
    parser.add_argument("--skip-eqgnn", action="store_true",
                        help="Skip Q-PRISM (slow due to PennyLane simulation)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Training epochs (more data → may need more epochs)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick validation run (2 epochs, N=1000 only)")
    args = parser.parse_args()

    if args.skip_eqgnn:
        args.models = [m for m in args.models if m != "eqgnn_v2"]

    if args.dry_run:
        args.n_values = [1000]
        args.seeds = [42]
        args.epochs = 2
        args.targets = ["homo"]
        print("🏃 DRY RUN mode: 2 epochs, N=1000, seed=42, HOMO only")

    run_p21_evaluation(
        n_values=args.n_values,
        targets=args.targets,
        seeds=args.seeds,
        models=args.models,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
    )
