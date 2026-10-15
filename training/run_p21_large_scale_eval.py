"""Multi-seed large-scale evaluation of Periodic-EGNN scaling."""

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
from scipy import stats
from torch_geometric.loader import DataLoader

from datasets.dataset_qm9 import get_qm9_splits, QM9_TARGET_INDICES
from models.eq_gnn import QPRISMModelV2, EQGNNModelV2
from models.egnn_baseline import count_parameters
from models.classical_twin import ClassicalTwinModelV2
from models.classical_trig_twin import PRISMModelV2, TrigTwinModelV2


TARGET_UNITS = {"homo": "eV", "lumo": "eV", "gap": "eV"}
SAVE_PATH = PROJECT_ROOT / "results" / "p21_large_scale_eval_multiseed.json"
EXISTING_PATH = PROJECT_ROOT / "results" / "p21_large_scale_eval.json"


def create_model(model_key, in_features=11, hidden_dim=16, num_layers=2, out_dim=1):
    """Create model by key."""
    if model_key == "trig_twin_siren":
        return PRISMModelV2(
            in_features=in_features, hidden_features=hidden_dim,
            num_layers=num_layers, n_q_layers=2, out_dim=out_dim,
            head_type="siren", omega_0=30.0,
        )
    elif model_key == "classical_twin":
        return ClassicalTwinModelV2(
            in_features=in_features, hidden_features=hidden_dim,
            num_layers=num_layers, n_q_layers=2, out_dim=out_dim,
        )
    elif model_key == "eqgnn_v2":
        return QPRISMModelV2(
            in_features=in_features, hidden_features=hidden_dim,
            num_layers=num_layers, n_q_layers=2, out_dim=out_dim,
        )
    raise ValueError(f"Unknown model: {model_key}")


MODEL_LABELS = {
    "trig_twin_siren": "PRISM (SIREN)",
    "classical_twin": "Classical Twin (tanh)",
    "eqgnn_v2": "Q-PRISM (PQC)",
}


def train_eval(model_key, target, seed, num_train, epochs=20, batch_size=32, lr=1e-3):
    """Train and evaluate a model, returning evaluation metrics."""
    device = torch.device("cpu") if model_key == "eqgnn_v2" else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_val = min(num_train // 4, 2000)
    num_test = min(num_train // 2, 5000)

    train_set, val_set, test_set, (mean, std) = get_qm9_splits(
        root="./data/QM9", target=target,
        num_train=num_train, num_val=num_val, num_test=num_test, seed=seed,
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    torch.manual_seed(seed)
    model = create_model(model_key).to(device)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        scheduler.step()

        if epoch % 5 == 0 or epoch == epochs:
            elapsed = time.time() - start_time
            print(f"      Epoch {epoch:3d}/{epochs} | {elapsed:.1f}s")

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

    return {
        "test_mae": float(test_mae),
        "params": total_params,
        "train_time_s": round(elapsed, 1),
        "num_train": num_train,
        "num_test": total_samples,
        "epochs": epochs,
    }


def load_existing_seed42():
    """Load existing seed-42 results from p21_large_scale_eval.json (20-epoch runs only)."""
    if not EXISTING_PATH.exists():
        return {}
    with open(EXISTING_PATH) as f:
        data = json.load(f)

    imported = {}
    for target in ["homo", "lumo", "gap"]:
        if target not in data:
            continue
        imported[target] = {}
        for n_key in data[target]:
            if n_key == "N=1000":
                continue  # Skip N=1000 (some may be dry-run with 2 epochs)
            imported[target][n_key] = {}
            for run_key, result in data[target][n_key].items():
                if result.get("epochs", 0) >= 20:  # Only import full runs
                    imported[target][n_key][run_key] = result
    return imported


def main():
    parser = argparse.ArgumentParser(description="Multi-Seed Large-Scale Evaluation")
    parser.add_argument("--n-values", nargs="+", type=int, default=[5000, 10000])
    parser.add_argument("--targets", nargs="+", default=["homo", "lumo", "gap"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--include-eqgnn", action="store_true",
                        help="Include EQ-GNN V2 (very slow)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        args.n_values = [5000]
        args.seeds = [42, 43]
        args.targets = ["homo"]
        args.epochs = 2
        print("DRY RUN: N=5000, 2 seeds, 2 epochs, HOMO only\n")

    models = ["trig_twin_siren", "classical_twin"]
    if args.include_eqgnn:
        models.append("eqgnn_v2")

    print("=" * 80)
    print("Multi-Seed Large-Scale Evaluation (Table 5)")
    print(f"   N values: {args.n_values}")
    print(f"   Targets:  {args.targets}")
    print(f"   Seeds:    {args.seeds}")
    print(f"   Models:   {[MODEL_LABELS[m] for m in models]}")
    print(f"   Epochs:   {args.epochs}")
    print("=" * 80)

    # Load checkpoint or start fresh
    if SAVE_PATH.exists():
        with open(SAVE_PATH) as f:
            results = json.load(f)
    else:
        results = {"metadata": {
            "description": "Multi-seed large-scale evaluation",
            "n_values": args.n_values, "targets": args.targets,
            "seeds": args.seeds, "models": models,
        }}

    # Import existing seed-42 results
    existing = load_existing_seed42()
    for target in args.targets:
        if target not in results:
            results[target] = {}
        for n_train in args.n_values:
            n_key = f"N={n_train}"
            if n_key not in results[target]:
                results[target][n_key] = {}
            # Import seed-42 from existing file
            if target in existing and n_key in existing[target]:
                for run_key, run_data in existing[target][n_key].items():
                    if run_key not in results[target][n_key]:
                        results[target][n_key][run_key] = run_data
                        print(f"  Imported {run_key} from existing: {run_data['test_mae']:.4f}")

    # Main training loop
    for target in args.targets:
        for n_train in args.n_values:
            n_key = f"N={n_train}"
            print(f"\nTarget: {target.upper()} | N = {n_train:,}")

            for model_key in models:
                for seed in args.seeds:
                    run_key = f"{model_key}_seed{seed}"

                    if run_key in results[target][n_key]:
                        cached = results[target][n_key][run_key]
                        if cached.get("epochs", 0) >= args.epochs:
                            print(f"  Cached: {MODEL_LABELS[model_key]} seed={seed}: "
                                  f"{cached['test_mae']:.4f}")
                            continue

                    print(f"\n  {MODEL_LABELS[model_key]} | seed={seed} | N={n_train}")
                    result = train_eval(model_key, target, seed, n_train,
                                        epochs=args.epochs)
                    results[target][n_key][run_key] = result
                    print(f"    MAE = {result['test_mae']:.4f} ({result['train_time_s']:.1f}s)")

                    with open(SAVE_PATH, "w") as f:
                        json.dump(results, f, indent=2)

    # Summary with statistics
    print("\n" + "=" * 80)
    print("Multi-Seed Summary Table")
    print("=" * 80)

    for target in args.targets:
        print(f"\n  Target: {target.upper()} ({TARGET_UNITS[target]})")
        print(f"  {'N':>8} | {'Model':<28} | {'MAE (mean ± std)':>28} | seeds")
        print(f"  {'-'*8}-+{'-'*30}+{'-'*30}+{'-'*6}")

        for n_train in args.n_values:
            n_key = f"N={n_train}"
            if n_key not in results[target]:
                continue

            for model_key in models:
                maes = []
                for seed in args.seeds:
                    run_key = f"{model_key}_seed{seed}"
                    if run_key in results[target][n_key]:
                        maes.append(results[target][n_key][run_key]["test_mae"])

                if maes:
                    mean_mae = np.mean(maes)
                    if len(maes) > 1:
                        std_mae = np.std(maes)
                        mae_str = f"{mean_mae:.4f} ± {std_mae:.4f}"
                    else:
                        mae_str = f"{mean_mae:.4f}"
                    print(f"  {n_train:>8,} | {MODEL_LABELS[model_key]:<28} | "
                          f"{mae_str:>28} | {len(maes)}")

        # Paired t-test: Periodic-EGNN vs Classical Twin
        print(f"\n  Paired t-tests (Periodic-EGNN vs Classical Twin):")
        for n_train in args.n_values:
            n_key = f"N={n_train}"
            siren_maes = []
            twin_maes = []
            for seed in args.seeds:
                sk = f"trig_twin_siren_seed{seed}"
                ck = f"classical_twin_seed{seed}"
                if sk in results[target].get(n_key, {}) and ck in results[target].get(n_key, {}):
                    siren_maes.append(results[target][n_key][sk]["test_mae"])
                    twin_maes.append(results[target][n_key][ck]["test_mae"])

            if len(siren_maes) >= 2:
                siren_arr = np.array(siren_maes)
                twin_arr = np.array(twin_maes)
                t_stat, p_val = stats.ttest_rel(siren_arr, twin_arr)
                diff = siren_arr - twin_arr
                pct = np.mean(diff) / np.mean(twin_arr) * 100
                print(f"    N={n_train:>6,}: Δ={np.mean(diff):+.4f} ({pct:+.1f}%), "
                      f"p={p_val:.4f} {'**' if p_val < 0.01 else '*' if p_val < 0.05 else '(ns)'}")

                # Store stats
                results[target][n_key]["stats"] = {
                    "siren_vs_twin_mean_diff": float(np.mean(diff)),
                    "siren_vs_twin_pct_diff": float(pct),
                    "siren_vs_twin_p": float(p_val),
                    "n_seeds": len(siren_maes),
                }
            elif len(siren_maes) == 1:
                diff = siren_maes[0] - twin_maes[0]
                pct = diff / twin_maes[0] * 100
                print(f"    N={n_train:>6,}: Δ={diff:+.4f} ({pct:+.1f}%), single seed")

    with open(SAVE_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {SAVE_PATH}")
    print("Multi-seed large-scale evaluation complete.")


if __name__ == "__main__":
    main()
