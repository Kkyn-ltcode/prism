"""Entanglement ablation benchmark across 20 random seeds."""

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

from datasets.dataset_qm9 import get_qm9_splits
from models.eq_gnn import QPRISMModelV2, EQGNNModelV2
from models.egnn_baseline import count_parameters

# Import from the existing ablation benchmark
from benchmarks.benchmark_entanglement_ablation_v2 import (
    SeparableEQGNNModelV2,
    ClassicalTanhEQGNNModel,
)

SEEDS_20 = list(range(42, 62))  # [42, 43, ..., 61] = 20 seeds
SEEDS_EXISTING = [42, 43, 44, 45, 46]  # Already have EQ-GNN V2 results
SEEDS_NEW_EQGNN = list(range(47, 62))  # 15 new seeds for EQ-GNN V2

TARGET_UNITS = {"homo": "eV", "lumo": "eV", "gap": "eV"}
SAVE_PATH = PROJECT_ROOT / "results" / "entanglement_ablation_20seeds.json"


def train_and_evaluate(model, target, seed, num_train=1000, num_val=250, num_test=500,
                       epochs=15, batch_size=16, lr=1e-3):
    """Train a model and return test MAE."""
    device = torch.device("cpu")

    train_set, val_set, test_set, (mean, std) = get_qm9_splits(
        root="./data/QM9", target=target,
        num_train=num_train, num_val=num_val, num_test=num_test, seed=seed
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    torch.manual_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.MSELoss()

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

    return total_mae / total_samples


def load_existing_results():
    """Load existing 5-seed entanglement ablation results."""
    path = PROJECT_ROOT / "results" / "entanglement_ablation_v2.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_checkpoint():
    """Load checkpoint of 20-seed run for resumption."""
    if SAVE_PATH.exists():
        with open(SAVE_PATH) as f:
            return json.load(f)
    return {}


def save_checkpoint(results):
    """Save checkpoint after each seed."""
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SAVE_PATH, "w") as f:
        json.dump(results, f, indent=2)


def run_model_sweep(model_name, model_factory, targets, seeds, results, epochs=15):
    """Run training for a model across all targets and seeds."""
    for target in targets:
        if target not in results:
            results[target] = {}
        if model_name not in results[target]:
            results[target][model_name] = {"maes": {}}

        print(f"\n{model_name} x {target.upper()}")

        for seed in seeds:
            seed_key = str(seed)
            if seed_key in results[target][model_name]["maes"]:
                cached = results[target][model_name]["maes"][seed_key]
                print(f"    Seed {seed}: {cached:.4f} (cached)")
                continue

            t0 = time.time()
            model = model_factory()
            mae = train_and_evaluate(model, target, seed, epochs=epochs)
            elapsed = time.time() - t0

            results[target][model_name]["maes"][seed_key] = mae
            print(f"    Seed {seed}: {mae:.4f} ({elapsed:.1f}s)")

            # Checkpoint after each seed
            save_checkpoint(results)

        # Update summary stats
        all_maes = list(results[target][model_name]["maes"].values())
        results[target][model_name]["mean"] = float(np.mean(all_maes))
        results[target][model_name]["std"] = float(np.std(all_maes))
        results[target][model_name]["n_seeds"] = len(all_maes)
        print(f"    Mean: {np.mean(all_maes):.4f} ± {np.std(all_maes):.4f} ({len(all_maes)} seeds)")
        save_checkpoint(results)


def run_statistical_analysis(results, targets):
    """Run paired t-tests with 20 seeds and compute power analysis."""
    print("\n" + "=" * 70)
    print("Statistical Analysis (20 seeds)")
    print("=" * 70)

    for target in targets:
        print(f"\n--- {target.upper()} ---")

        # Gather MAE arrays aligned by seed
        models_data = {}
        for model_name in ["eqgnn_v2", "separable_v2", "tanh_mlp"]:
            if model_name in results[target]:
                maes_dict = results[target][model_name]["maes"]
                models_data[model_name] = maes_dict

        # Find common seeds
        common_seeds = sorted(
            set(models_data.get("eqgnn_v2", {}).keys()) &
            set(models_data.get("separable_v2", {}).keys()) &
            set(models_data.get("tanh_mlp", {}).keys())
        )
        n_common = len(common_seeds)
        print(f"  Common seeds: {n_common}")

        if n_common < 5:
            print("  Too few common seeds for analysis")
            continue

        eq_maes = np.array([models_data["eqgnn_v2"][s] for s in common_seeds])
        sep_maes = np.array([models_data["separable_v2"][s] for s in common_seeds])
        tanh_maes = np.array([models_data["tanh_mlp"][s] for s in common_seeds])

        print(f"  EQ-GNN V2:    {np.mean(eq_maes):.4f} ± {np.std(eq_maes):.4f}")
        print(f"  Separable V2: {np.mean(sep_maes):.4f} ± {np.std(sep_maes):.4f}")
        print(f"  Tanh-MLP:     {np.mean(tanh_maes):.4f} ± {np.std(tanh_maes):.4f}")

        # Test 1: Entangled vs Separable
        diff_ent = eq_maes - sep_maes
        t_stat, p_val = stats.ttest_rel(eq_maes, sep_maes)
        cohen_d = np.mean(diff_ent) / np.std(diff_ent) if np.std(diff_ent) > 0 else 0
        print(f"\n  Entangled vs Separable:")
        print(f"    Mean diff: {np.mean(diff_ent):+.4f} eV")
        print(f"    Paired t-test: t={t_stat:.3f}, p={p_val:.4f}")
        print(f"    Cohen's d: {cohen_d:.3f}")
        print(f"    {'**' if p_val < 0.01 else '*' if p_val < 0.05 else '(ns)'}")

        results[target]["stats_20seed"] = results[target].get("stats_20seed", {})
        results[target]["stats_20seed"]["entangled_vs_separable"] = {
            "mean_diff": float(np.mean(diff_ent)),
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "cohen_d": float(cohen_d),
            "n_seeds": n_common,
        }

        # Test 2: Separable vs Tanh
        diff_sep = sep_maes - tanh_maes
        t_stat2, p_val2 = stats.ttest_rel(sep_maes, tanh_maes)
        cohen_d2 = np.mean(diff_sep) / np.std(diff_sep) if np.std(diff_sep) > 0 else 0
        print(f"\n  Separable vs Tanh:")
        print(f"    Mean diff: {np.mean(diff_sep):+.4f} eV")
        print(f"    Paired t-test: t={t_stat2:.3f}, p={p_val2:.4f}")
        print(f"    Cohen's d: {cohen_d2:.3f}")
        print(f"    {'**' if p_val2 < 0.01 else '*' if p_val2 < 0.05 else '(ns)'}")

        results[target]["stats_20seed"]["separable_vs_tanh"] = {
            "mean_diff": float(np.mean(diff_sep)),
            "t_stat": float(t_stat2),
            "p_value": float(p_val2),
            "cohen_d": float(cohen_d2),
            "n_seeds": n_common,
        }

        # Test 3: Entangled vs Tanh
        diff_et = eq_maes - tanh_maes
        t_stat3, p_val3 = stats.ttest_rel(eq_maes, tanh_maes)
        cohen_d3 = np.mean(diff_et) / np.std(diff_et) if np.std(diff_et) > 0 else 0
        print(f"\n  Entangled vs Tanh:")
        print(f"    Mean diff: {np.mean(diff_et):+.4f} eV")
        print(f"    Paired t-test: t={t_stat3:.3f}, p={p_val3:.4f}")
        print(f"    Cohen's d: {cohen_d3:.3f}")
        print(f"    {'**' if p_val3 < 0.01 else '*' if p_val3 < 0.05 else '(ns)'}")

        results[target]["stats_20seed"]["entangled_vs_tanh"] = {
            "mean_diff": float(np.mean(diff_et)),
            "t_stat": float(t_stat3),
            "p_value": float(p_val3),
            "cohen_d": float(cohen_d3),
            "n_seeds": n_common,
        }

        # Power analysis (post-hoc)
        # For 20 paired samples, detectable effect size at 80% power, alpha=0.05
        # Using the approximation: d_min ≈ t_crit * sqrt(1/n)
        from scipy.stats import t as t_dist
        t_crit = t_dist.ppf(0.975, df=n_common - 1)
        d_min = t_crit / np.sqrt(n_common)
        # Convert to MAE units using the observed std of differences
        mae_detectable = d_min * np.std(diff_ent) if np.std(diff_ent) > 0 else float("nan")
        print(f"\n  Post-hoc power analysis (α=0.05, 80% power):")
        print(f"    Min detectable Cohen's d: {d_min:.3f}")
        print(f"    Min detectable MAE diff:  {mae_detectable:.4f} eV")
        print(f"    (vs. 5-seed: d_min ≈ {t_dist.ppf(0.975, df=4) / np.sqrt(5):.3f})")

        results[target]["stats_20seed"]["power_analysis"] = {
            "min_detectable_cohens_d": float(d_min),
            "min_detectable_mae_diff": float(mae_detectable),
        }

    save_checkpoint(results)


def main():
    parser = argparse.ArgumentParser(description="20-Seed Entanglement Ablation")
    parser.add_argument("--targets", nargs="+", default=["homo", "lumo", "gap"],
                        choices=["homo", "lumo", "gap"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick test: 2 epochs, 2 seeds, 1 target")
    args = parser.parse_args()

    if args.dry_run:
        args.targets = ["homo"]
        args.epochs = 2
        seeds_all = [42, 43]
        seeds_eqgnn_new = []
        print("DRY RUN: 2 seeds, 2 epochs, HOMO only\n")
    else:
        seeds_all = SEEDS_20
        seeds_eqgnn_new = SEEDS_NEW_EQGNN

    # Verify parameter counts
    print("=" * 70)
    print("20-Seed Entanglement Ablation")
    print("=" * 70)

    m_ent = EQGNNModelV2(in_features=11, hidden_features=16, num_layers=2, n_q_layers=2, out_dim=1)
    m_sep = SeparableEQGNNModelV2(in_features=11, hidden_features=16, num_layers=2, n_q_layers=2, out_dim=1)
    m_tanh = ClassicalTanhEQGNNModel(in_features=11, hidden_features=16, num_layers=2, out_dim=1)

    p_ent = count_parameters(m_ent)
    p_sep = count_parameters(m_sep)
    p_tanh = count_parameters(m_tanh)
    print(f"\nParameter counts:")
    print(f"  EQ-GNN V2 (entangled):   {p_ent:,}")
    print(f"  Separable V2 (no CNOTs): {p_sep:,}")
    print(f"  Classical tanh-MLP:      {p_tanh:,}")
    assert p_ent == p_sep == p_tanh, f"Parameter mismatch! {p_ent} vs {p_sep} vs {p_tanh}"
    print(f"  All models: {p_ent:,} parameters (exact match)")
    del m_ent, m_sep, m_tanh

    # Load existing 5-seed results for EQ-GNN V2
    results = load_checkpoint()
    existing = load_existing_results()

    # Seed existing EQ-GNN V2 results (5 seeds) into the checkpoint
    for target in args.targets:
        if target not in results:
            results[target] = {}
        if "eqgnn_v2" not in results[target]:
            results[target]["eqgnn_v2"] = {"maes": {}}
        if target in existing and "eqgnn_v2" in existing[target]:
            for i, seed in enumerate(SEEDS_EXISTING):
                seed_key = str(seed)
                if seed_key not in results[target]["eqgnn_v2"]["maes"]:
                    results[target]["eqgnn_v2"]["maes"][seed_key] = existing[target]["eqgnn_v2"]["maes"][i]
                    print(f"  Imported EQ-GNN V2 {target.upper()} seed {seed}: "
                          f"{existing[target]['eqgnn_v2']['maes'][i]:.4f} (from 5-seed run)")

    save_checkpoint(results)

    # Run EQ-GNN V2 for new seeds
    if seeds_eqgnn_new:
        print(f"\nPhase 1: EQ-GNN V2 (new seeds {seeds_eqgnn_new[0]}-{seeds_eqgnn_new[-1]})")
        run_model_sweep(
            "eqgnn_v2",
            lambda: EQGNNModelV2(in_features=11, hidden_features=16, num_layers=2, n_q_layers=2, out_dim=1),
            args.targets, seeds_eqgnn_new, results, epochs=args.epochs,
        )

    # Run Separable V2 for all 20 seeds
    print(f"\nPhase 2: Separable V2 (all {len(seeds_all)} seeds)")
    run_model_sweep(
        "separable_v2",
        lambda: SeparableEQGNNModelV2(in_features=11, hidden_features=16, num_layers=2, n_q_layers=2, out_dim=1),
        args.targets, seeds_all, results, epochs=args.epochs,
    )

    # Run Classical tanh-MLP for all 20 seeds
    print(f"\nPhase 3: Classical tanh-MLP (all {len(seeds_all)} seeds)")
    run_model_sweep(
        "tanh_mlp",
        lambda: ClassicalTanhEQGNNModel(in_features=11, hidden_features=16, num_layers=2, out_dim=1),
        args.targets, seeds_all, results, epochs=args.epochs,
    )

    # Statistical Analysis
    run_statistical_analysis(results, args.targets)

    # Final Verdict
    print("\n" + "=" * 70)
    print("Bowles Test Verdict")
    print("=" * 70)
    for target in args.targets:
        s = results[target].get("stats_20seed", {})
        ent_vs_sep = s.get("entangled_vs_separable", {})
        sep_vs_tanh = s.get("separable_vs_tanh", {})
        p_ent = ent_vs_sep.get("p_value", 1.0)
        p_sep = sep_vs_tanh.get("p_value", 1.0)
        d_ent = ent_vs_sep.get("cohen_d", 0)

        print(f"\n  {target.upper()}:")
        print(f"    Entangled vs Separable: p={p_ent:.4f}, d={d_ent:.3f}")
        print(f"    Separable vs Tanh:     p={p_sep:.4f}")
        if p_ent < 0.05:
            print(f"    Entanglement provides significant advantage (p={p_ent:.4f})")
        else:
            print(f"    Entanglement not significant (p={p_ent:.4f}) - advantage is trigonometric")

    print(f"\nResults saved to: {SAVE_PATH}")
    print("20-seed entanglement ablation complete.")


if __name__ == "__main__":
    main()
