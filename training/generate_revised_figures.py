"""
Generate Revised Publication Figures (Phase 2.2)
=================================================
Regenerates:
  - Figure 1: QM9 Scaling Laws with all 6 baselines (Gap + μ)
  - Figure 5: N-body Trajectory Forecasting with Classical Twin
  - Figure 6: MD17 Conformational Energy with Classical Twin

Usage:
    pixi run python training/generate_revised_figures.py
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Paths
RESULTS_DIR = "./results"
FIGURES_DIR = "./paper/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Consistent style ──
COLORS = {
    "eqgnn_v2": "#1f77b4",       # Blue
    "classical_twin": "#e377c2",  # Pink
    "schnet": "#2ca02c",          # Green
    "painn": "#d62728",           # Red
    "egnn_pm": "#9467bd",         # Purple
    "egnn_std": "#ff7f0e",        # Orange
    "eqgnn": "#1f77b4",          # Blue (alias)
    "egnn": "#ff7f0e",           # Orange (alias)
    "noneq": "#7f7f7f",          # Gray
    "static_baseline": "#bcbd22", # Olive
}

LABELS = {
    "eqgnn_v2": "Q-PRISM (Ours)",
    "classical_twin": "Classical Twin",
    "schnet": "SchNet",
    "painn": "PaiNN",
    "egnn_pm": "EGNN Param-Matched",
    "egnn_std": "EGNN Standard",
    "eqgnn": "Q-PRISM (Ours)",
    "egnn": "EGNN Standard",
    "noneq": "Non-Equiv GNN",
    "static_baseline": "Static Baseline",
}

MARKERS = {
    "eqgnn_v2": "o",
    "classical_twin": "s",
    "schnet": "^",
    "painn": "D",
    "egnn_pm": "v",
    "egnn_std": "P",
    "eqgnn": "o",
    "egnn": "P",
    "noneq": "X",
    "static_baseline": "_",
}


def load_json(filename):
    with open(os.path.join(RESULTS_DIR, filename)) as f:
        return json.load(f)


# =============================================================================
# FIGURE 1: QM9 Scaling Laws (Gap + μ) — Log-Log with crossover
# =============================================================================
def generate_figure1_scaling():
    print("Generating Figure 1: QM9 Scaling Laws...")
    data = load_json("qm9_scaling_sweep.json")

    models = ["eqgnn_v2", "classical_twin", "schnet", "painn", "egnn_pm", "egnn_std"]
    n_values = [100, 250, 500, 1000, 2500, 5000, 10000]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    for ax_idx, (target, title, ylabel) in enumerate([
        ("gap", "HOMO-LUMO Gap ($\\Delta\\epsilon$)", "Test MAE (eV)"),
        ("mu", "Dipole Moment ($\\mu$)", "Test MAE (Debye)"),
    ]):
        ax = axes[ax_idx]
        t_data = data.get(target, {})

        for model in models:
            ns_valid = []
            means = []
            stds = []
            for n in n_values:
                key = f"N{n}"
                if key in t_data and model in t_data[key] and "mean_mae" in t_data[key][model]:
                    ns_valid.append(n)
                    means.append(t_data[key][model]["mean_mae"])
                    stds.append(t_data[key][model].get("std_mae", 0))

            if not ns_valid:
                continue

            means = np.array(means)
            stds = np.array(stds)

            ax.plot(ns_valid, means,
                    marker=MARKERS[model], markersize=7,
                    color=COLORS[model], label=LABELS[model],
                    linewidth=2.0, alpha=0.9)
            if np.any(stds > 0):
                ax.fill_between(ns_valid, means - stds, means + stds,
                                color=COLORS[model], alpha=0.1)

        # Find and annotate crossover point (Q-PRISM vs PaiNN)
        eq_data = {}
        painn_data = {}
        for n in n_values:
            key = f"N{n}"
            if key in t_data:
                if "eqgnn_v2" in t_data[key] and "mean_mae" in t_data[key]["eqgnn_v2"]:
                    eq_data[n] = t_data[key]["eqgnn_v2"]["mean_mae"]
                if "painn" in t_data[key] and "mean_mae" in t_data[key]["painn"]:
                    painn_data[n] = t_data[key]["painn"]["mean_mae"]

        # Find approximate crossover N
        common_ns = sorted(set(eq_data.keys()) & set(painn_data.keys()))
        crossover_n = None
        for i in range(len(common_ns) - 1):
            n1, n2 = common_ns[i], common_ns[i + 1]
            diff1 = eq_data[n1] - painn_data[n1]
            diff2 = eq_data[n2] - painn_data[n2]
            if diff1 < 0 and diff2 > 0:
                # Linear interpolation for crossover
                frac = abs(diff1) / (abs(diff1) + abs(diff2))
                crossover_n = n1 + frac * (n2 - n1)
                crossover_mae = eq_data[n1] + frac * (eq_data[n2] - eq_data[n1])
                break

        if crossover_n:
            ax.axvline(x=crossover_n, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            ax.annotate(f'Crossover\n$N \\approx {int(crossover_n)}$',
                        xy=(crossover_n, crossover_mae),
                        xytext=(crossover_n * 1.8, crossover_mae * 1.15),
                        fontsize=8, color='gray',
                        arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
                        ha='center')

            # Shade the Q-PRISM dominance region
            ax.axvspan(ax.get_xlim()[0] if ax.get_xlim()[0] > 0 else 50,
                        crossover_n, color='#1f77b4', alpha=0.04)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Training Samples $N$", fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.set_xticks(n_values)
        ax.set_xticklabels([str(n) for n in n_values], fontsize=9)

        if ax_idx == 0:
            ax.legend(fontsize=8, loc="upper right", framealpha=0.9, ncol=1)

    fig.suptitle("Sample Efficiency Scaling Laws on QM9 (5 Seeds, All Baselines)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "figure1_scaling_law.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {path}")


# =============================================================================
# FIGURE 5: N-Body Trajectory Forecasting (with Classical Twin)
# =============================================================================
def generate_figure5_nbody():
    print("Generating Figure 5: N-Body Trajectory...")
    data = load_json("nbody_benchmark_table2_v2.json")

    model_keys = ["static_baseline", "noneq", "egnn_pm", "classical_twin", "eqgnn", "egnn"]
    labels = [
        "Static Baseline\n($x^T = x^0$)",
        "Non-Equiv GNN\n(Cartesian)",
        "EGNN\nParam-Matched",
        "Classical Twin\n(Iso-Param)",
        "Q-PRISM (Ours)\n(4-Qubit PQC)",
        "EGNN Standard\n(Full Capacity)",
    ]
    colors_list = [
        COLORS["static_baseline"],
        COLORS["noneq"],
        COLORS["egnn_pm"],
        COLORS["classical_twin"],
        COLORS["eqgnn"],
        COLORS["egnn"],
    ]

    means = [data[k]["mean_mse"] for k in model_keys]
    stds = [data[k]["std_mse"] for k in model_keys]
    params = [data[k].get("params", 0) for k in model_keys]

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors_list,
                  width=0.6, edgecolor="black", alpha=0.9, linewidth=0.8)

    # Value labels
    for bar, mean_val, p in zip(bars, means, params):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{mean_val:.4f}", ha="center", fontweight="bold", fontsize=9)
        if p > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, 0.01,
                    f"{p:,}p", ha="center", fontsize=7, color="white", fontweight="bold")

    # Highlight Q-PRISM vs Twin comparison
    twin_idx = model_keys.index("classical_twin")
    eqgnn_idx = model_keys.index("eqgnn")
    twin_mse = means[twin_idx]
    eqgnn_mse = means[eqgnn_idx]
    improvement = (twin_mse - eqgnn_mse) / twin_mse * 100

    # Draw comparison bracket
    bracket_y = max(twin_mse, eqgnn_mse) + stds[twin_idx] + 0.03
    ax.annotate("", xy=(twin_idx, bracket_y), xytext=(eqgnn_idx, bracket_y),
                arrowprops=dict(arrowstyle="<->", color="#1f77b4", linewidth=1.5))
    ax.text((twin_idx + eqgnn_idx) / 2, bracket_y + 0.008,
            f"Q-PRISM wins\n({improvement:.1f}% lower)",
            ha="center", fontsize=8, color="#1f77b4", fontweight="bold")

    ax.set_ylabel("Coordinate Trajectory Prediction MSE ($10^{-2}$)",
                  fontsize=11, fontweight="bold")
    ax.set_title("3D Charged N-Body 5-Particle Trajectory Forecasting (Table 2)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    ax.set_ylim(0, max(means) + max(stds) + 0.12)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "figure5_nbody_trajectory.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {path}")


# =============================================================================
# FIGURE 6: MD17 Conformational Energy (with Classical Twin)
# =============================================================================
def generate_figure6_md17():
    print("Generating Figure 6: MD17 Energy...")
    data = load_json("md17_benchmark_table3_v2.json")

    model_keys = ["noneq", "egnn_pm", "egnn", "classical_twin", "eqgnn"]
    labels_short = ["Non-Equiv\nGNN", "EGNN\nParam-Match", "EGNN\nStandard",
                    "Classical\nTwin", "Q-PRISM\n(Ours)"]
    colors_list = [COLORS[k] for k in model_keys]
    molecules = ["ethanol", "aspirin"]
    mol_titles = ["Ethanol (9 atoms)", "Aspirin (21 atoms)"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), dpi=300)
    fig.subplots_adjust(top=0.88, wspace=0.35)

    for ax_idx, (mol, mol_title) in enumerate(zip(molecules, mol_titles)):
        ax = axes[ax_idx]
        mol_data = data[mol]

        means = [mol_data[k]["mean_mae"] for k in model_keys]
        stds = [mol_data[k]["std_mae"] for k in model_keys]
        params = [mol_data[k]["params"] for k in model_keys]

        x = np.arange(len(model_keys))
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors_list,
                      width=0.6, edgecolor="black", alpha=0.9, linewidth=0.8)

        for i, (bar, mean_val, p) in enumerate(zip(bars, means, params)):
            ax.text(bar.get_x() + bar.get_width() / 2, mean_val - stds[i] - 0.01,
                    f"{mean_val:.3f}", ha="center", va="top", fontweight="bold", fontsize=8.5)
            ax.text(bar.get_x() + bar.get_width() / 2, min(means) - max(stds) - 0.05,
                    f"{p:,}p", ha="center", fontsize=6.5, color="black", fontweight="bold")

        # Highlight Q-PRISM vs Twin
        twin_idx = model_keys.index("classical_twin")
        eqgnn_idx = model_keys.index("eqgnn")
        eq_mae = means[eqgnn_idx]
        tw_mae = means[twin_idx]
        diff = eq_mae - tw_mae

        bracket_y = max(means) + max(stds) + 0.05
        if abs(diff) < 0.02:  # Statistical tie
            ax.text((twin_idx + eqgnn_idx) / 2, bracket_y + 0.02,
                    f"Statistical tie\n($\\Delta$={diff:+.3f})",
                    ha="center", fontsize=7.5, color="gray", style="italic")

        ax.set_xticks(x)
        ax.set_xticklabels(labels_short, fontsize=8.5)
        ax.set_ylabel("Energy MAE (kcal/mol)", fontsize=11, fontweight="bold")
        ax.set_title(mol_title, fontsize=12, fontweight="bold", pad=10)
        ax.grid(True, axis="y", ls="--", alpha=0.4)

        # Tight y-axis around the narrow range
        ymin = min(means) - max(stds) - 0.1
        ymax = max(means) + max(stds) + 0.15
        ax.set_ylim(ymin, ymax)

    fig.suptitle("MD17 Conformational Energy Prediction (Table 3, $N=1000$, 5 Seeds)",
                 fontsize=13, fontweight="bold")

    path = os.path.join(FIGURES_DIR, "figure6_md17_energy.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {path}")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Generating Revised Publication Figures (Phase 2.2)")
    print("=" * 70)

    generate_figure1_scaling()
    generate_figure5_nbody()
    generate_figure6_md17()

    print("\n" + "=" * 70)
    print("All figures saved to paper/figures/")
    print("=" * 70)
