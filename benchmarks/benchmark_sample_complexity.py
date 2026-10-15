"""Sample complexity and generalization scaling benchmark across training set sizes."""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import sys
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from torch_geometric.loader import DataLoader

from datasets.dataset_qm9 import get_qm9_splits, QM9_TARGET_INDICES
from models.eq_gnn import QPRISMModelV2, EQGNNModelV2
from models.egnn_baseline import count_parameters, create_param_matched_egnn
from benchmarks.benchmark_entanglement_ablation_v2 import (
    SeparableQPRISMModelV2, ClassicalTanhQPRISMModel,
    SeparableEQGNNModelV2, ClassicalTanhEQGNNModel
)


def train_and_evaluate_full(model, target, seed, num_train, num_val=250,
                            num_test=500, epochs=15, batch_size=16, lr=1e-3):
    """Train model and return train and test MAE."""
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

    # --- Evaluate on TRAIN set ---
    model.eval()
    train_mae_total = 0.0
    train_count = 0
    with torch.no_grad():
        for batch in train_loader:
            batch = batch.to(device)
            pred_norm = model(batch.x, batch.pos, batch.edge_index, batch.batch)
            pred = pred_norm * std + mean
            train_mae_total += torch.abs(pred - batch.y).sum().item()
            train_count += batch.num_graphs

    # --- Evaluate on TEST set ---
    test_mae_total = 0.0
    test_count = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred_norm = model(batch.x, batch.pos, batch.edge_index, batch.batch)
            pred = pred_norm * std + mean
            test_mae_total += torch.abs(pred - batch.y).sum().item()
            test_count += batch.num_graphs

    return train_mae_total / train_count, test_mae_total / test_count


MODEL_CONFIGS = {
    "eqgnn_v2": {
        "label": "Q-PRISM (Entangled PQC)",
        "color": "#2196F3",
        "marker": "o",
        "factory": lambda: QPRISMModelV2(
            in_features=11, hidden_features=16, num_layers=2,
            n_q_layers=2, out_dim=1
        ),
    },
    "separable_v2": {
        "label": "Separable Q-PRISM (No Entanglement)",
        "color": "#FF9800",
        "marker": "s",
        "factory": lambda: SeparableQPRISMModelV2(
            in_features=11, hidden_features=16, num_layers=2,
            n_q_layers=2, out_dim=1
        ),
    },
    "tanh_mlp": {
        "label": "Classical tanh-MLP",
        "color": "#F44336",
        "marker": "^",
        "factory": lambda: ClassicalTanhQPRISMModel(
            in_features=11, hidden_features=16, num_layers=2, out_dim=1
        ),
    },
    "egnn_pm": {
        "label": "EGNN Param-Matched",
        "color": "#9E9E9E",
        "marker": "D",
        "factory": lambda: create_param_matched_egnn(
            target_params=2527, in_features=11, num_layers=2, out_dim=1
        )[0],
    },
}


def plot_scaling_curves(results, save_path):
    """Create publication-quality 1×3 figure: test MAE, train MAE, gen gap."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))

    n_values = sorted(results["n_values"])

    for model_key, cfg in MODEL_CONFIGS.items():
        if model_key not in results["models"]:
            continue

        model_data = results["models"][model_key]
        test_means = [model_data[str(n)]["test_mae_mean"] for n in n_values]
        test_stds = [model_data[str(n)]["test_mae_std"] for n in n_values]
        train_means = [model_data[str(n)]["train_mae_mean"] for n in n_values]
        train_stds = [model_data[str(n)]["train_mae_std"] for n in n_values]
        gap_means = [model_data[str(n)]["gen_gap_mean"] for n in n_values]
        gap_stds = [model_data[str(n)]["gen_gap_std"] for n in n_values]

        label = cfg["label"]
        color = cfg["color"]
        marker = cfg["marker"]

        # Test MAE
        ax1.errorbar(n_values, test_means, yerr=test_stds,
                     label=label, color=color, marker=marker,
                     linewidth=2, markersize=7, capsize=4)

        # Train MAE
        ax2.errorbar(n_values, train_means, yerr=train_stds,
                     label=label, color=color, marker=marker,
                     linewidth=2, markersize=7, capsize=4)

        # Generalization gap
        ax3.errorbar(n_values, gap_means, yerr=gap_stds,
                     label=label, color=color, marker=marker,
                     linewidth=2, markersize=7, capsize=4)

    for ax in (ax1, ax2, ax3):
        ax.set_xscale("log")
        ax.set_xlabel("Training set size $N$", fontsize=12)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=9, loc="upper right")

    ax1.set_ylabel("Test MAE (eV)", fontsize=12)
    ax1.set_title("Test Error vs. Sample Size\n(lower = better)", fontsize=13, fontweight="bold")
    ax1.set_yscale("log")

    ax2.set_ylabel("Train MAE (eV)", fontsize=12)
    ax2.set_title("Training Error vs. Sample Size\n(measures fitting capacity)", fontsize=13, fontweight="bold")
    ax2.set_yscale("log")

    ax3.set_ylabel("Gen. Gap = Test MAE − Train MAE (eV)", fontsize=12)
    ax3.set_title("Generalization Gap\n(lower = better generalization)", fontsize=13, fontweight="bold")
    ax3.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")

    fig.suptitle(
        "Sample Complexity Scaling: Quantum vs Classical Models on QM9 HOMO\n"
        "(Caro et al. 2022 — Unitary Regularization Hypothesis)",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {save_path}")


def plot_scaling_exponents(results, save_path):
    """Fit and plot power-law scaling MAE ~ N^{-α} for each model."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    n_values = np.array(sorted(results["n_values"]))
    log_n = np.log(n_values)

    exponents = {}
    for model_key, cfg in MODEL_CONFIGS.items():
        if model_key not in results["models"]:
            continue

        model_data = results["models"][model_key]
        test_means = np.array([model_data[str(n)]["test_mae_mean"] for n in n_values])
        log_mae = np.log(test_means)

        # Fit log(MAE) = -α·log(N) + c  →  MAE ~ N^{-α}
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(log_n, log_mae)
        alpha = -slope
        exponents[model_key] = {
            "alpha": float(alpha),
            "r_squared": float(r_value**2),
            "p_value": float(p_value),
            "std_err": float(std_err),
        }

        # Plot data points
        ax.scatter(n_values, test_means, color=cfg["color"], marker=cfg["marker"],
                   s=80, zorder=5, edgecolor="black", linewidth=0.5)

        # Plot fit line
        fit_line = np.exp(intercept) * n_values ** (-alpha)
        ax.plot(n_values, fit_line, color=cfg["color"], linewidth=2, linestyle="--",
                label=f"{cfg['label']}: α = {alpha:.3f} (R² = {r_value**2:.3f})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training set size $N$", fontsize=13)
    ax.set_ylabel("Test MAE (eV)", fontsize=13)
    ax.set_title("Power-Law Scaling: MAE ~ $N^{-\\alpha}$\n"
                 "(steeper α = better sample efficiency)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {save_path}")

    return exponents


def main():
    print("=" * 70)
    print("SAMPLE COMPLEXITY & GENERALIZATION SCALING BENCHMARK")
    print("(Caro et al. 2022)")
    print("=" * 70)

    TARGET = "homo"
    N_VALUES = [100, 250, 500, 1000, 2000, 5000]
    SEEDS = [42, 43, 44]  # 3 seeds for manageable runtime
    NUM_VAL = 250
    NUM_TEST = 500

    # Verify parameter counts
    print("\nParameter verification:")
    for key, cfg in MODEL_CONFIGS.items():
        m = cfg["factory"]()
        p = count_parameters(m)
        print(f"  {cfg['label']:<35s} {p:,} params")
        del m

    results = {
        "target": TARGET,
        "n_values": N_VALUES,
        "seeds": SEEDS,
        "models": {},
    }

    # --- Main loop ---
    total_runs = len(MODEL_CONFIGS) * len(N_VALUES) * len(SEEDS)
    run_idx = 0

    for model_key, cfg in MODEL_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"MODEL: {cfg['label']}")
        print(f"{'='*60}")

        model_results = {}

        for n_train in N_VALUES:
            train_maes = []
            test_maes = []
            gen_gaps = []

            print(f"\n  N = {n_train:,}")
            for seed in SEEDS:
                run_idx += 1
                t0 = time.time()

                model = cfg["factory"]()
                train_mae, test_mae = train_and_evaluate_full(
                    model, TARGET, seed,
                    num_train=n_train, num_val=NUM_VAL, num_test=NUM_TEST
                )
                elapsed = time.time() - t0

                gen_gap = test_mae - train_mae
                train_maes.append(train_mae)
                test_maes.append(test_mae)
                gen_gaps.append(gen_gap)

                print(f"    Seed {seed}: train={train_mae:.4f}, test={test_mae:.4f}, "
                      f"gap={gen_gap:+.4f} ({elapsed:.1f}s) [{run_idx}/{total_runs}]")

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            model_results[str(n_train)] = {
                "train_maes": train_maes,
                "test_maes": test_maes,
                "gen_gaps": gen_gaps,
                "train_mae_mean": float(np.mean(train_maes)),
                "train_mae_std": float(np.std(train_maes)),
                "test_mae_mean": float(np.mean(test_maes)),
                "test_mae_std": float(np.std(test_maes)),
                "gen_gap_mean": float(np.mean(gen_gaps)),
                "gen_gap_std": float(np.std(gen_gaps)),
            }

            print(f"  >> N={n_train:,}: test={np.mean(test_maes):.4f}±{np.std(test_maes):.4f}, "
                  f"train={np.mean(train_maes):.4f}±{np.std(train_maes):.4f}, "
                  f"gap={np.mean(gen_gaps):+.4f}±{np.std(gen_gaps):.4f}")

        results["models"][model_key] = model_results

    # Generate figures
    print("\nGenerating publication figures...")
    os.makedirs(PROJECT_ROOT / "paper" / "figures", exist_ok=True)

    plot_scaling_curves(
        results,
        PROJECT_ROOT / "paper" / "figures" / "figure_scaling_sample_complexity.png"
    )

    exponents = plot_scaling_exponents(
        results,
        PROJECT_ROOT / "paper" / "figures" / "figure_scaling_exponents.png"
    )

    # Save results
    results["scaling_exponents"] = exponents

    save_path = PROJECT_ROOT / "results" / "sample_complexity_scaling.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {save_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SCALING EXPONENTS SUMMARY (MAE ~ N^{-α})")
    print("=" * 70)
    print(f"\n{'Model':<35s} {'α':<10s} {'R²':<10s} {'Interpretation'}")
    print("-" * 80)
    for model_key in MODEL_CONFIGS:
        if model_key in exponents:
            e = exponents[model_key]
            interp = "better" if e["alpha"] > 0.2 else "moderate" if e["alpha"] > 0.1 else "weak"
            print(f"{MODEL_CONFIGS[model_key]['label']:<35s} {e['alpha']:<10.4f} {e['r_squared']:<10.3f} {interp} sample efficiency")

    print("\nGeneralization gap at N=100 vs N=5000:")
    for model_key in MODEL_CONFIGS:
        if model_key in results["models"]:
            md = results["models"][model_key]
            gap_100 = md["100"]["gen_gap_mean"]
            gap_5000 = md["5000"]["gen_gap_mean"]
            print(f"  {MODEL_CONFIGS[model_key]['label']:<35s} "
                  f"N=100: {gap_100:+.4f}, N=5000: {gap_5000:+.4f}")

    print("\nInterpretation (Caro et al. 2022):")
    print("   • Steeper α = better sample efficiency (learns faster from fewer data)")
    print("   • Smaller gen gap = better generalization (less overfitting)")
    print("   • Quantum unitary constraint acts as geometric regularizer →")
    print("     expect PQC to show steeper α and/or smaller gen gap at low N")
    print("\nSample complexity analysis complete.")


if __name__ == "__main__":
    main()
