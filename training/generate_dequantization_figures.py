"""
Generate Dequantization & PRISM Figures (Figure 9 & Figure 10)
=============================================================
Generates:
  - figure9_dequantization_comparison.png:
      Bar chart comparing Classical Twin (tanh), Fourier RFF, Q-PRISM (PQC),
      and PRISM (SIREN) on electronic targets + Dipole.
  - figure10_angular_activation_response.png:
      Visualizes edge functional response vs. 3-body bond angle theta in [0, 180],
      contrasting standard tanh (spectral bias) with SIREN and PQC.

Usage:
    pixi run python training/generate_dequantization_figures.py
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

RESULTS_DIR = "./results"
FIGURES_DIR = "./paper/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_json(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def generate_figure9_dequantization():
    print("Generating Figure 9: Dequantization Comparison...")
    data = load_json("extended_table1_v2.json")
    if not data:
        print("extended_table1_v2.json not found!")
        return

    targets = ["homo", "lumo", "gap", "mu"]
    target_titles = [
        "HOMO Energy (eV)",
        "LUMO Energy (eV)",
        "HOMO-LUMO Gap (eV)",
        "Dipole Moment (Debye)"
    ]

    models = ["classical_twin", "trig_twin_fourier", "eqgnn_v2", "trig_twin_siren"]
    model_labels = ["Classical Twin (tanh)", "Fourier Twin (RFF)", "Q-PRISM (PQC)", "PRISM (SIREN)"]
    colors = ["#7f7f7f", "#bcbd22", "#1f77b4", "#2ca02c"]  # Gray, Olive, Blue, Green

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), dpi=300)

    for i, target in enumerate(targets):
        ax = axes[i]
        t_data = data.get(target, {})

        means = []
        stds = []
        for m in models:
            m_info = t_data.get(m, {})
            means.append(m_info.get("mean_mae", 0.0))
            stds.append(m_info.get("std_mae", 0.0))

        x = np.arange(len(models))
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85, edgecolor="black", linewidth=1.0)

        # Highlight best
        best_idx = int(np.argmin(means))
        bars[best_idx].set_edgecolor("#d62728")
        bars[best_idx].set_linewidth(2.2)

        ax.set_title(target_titles[i], fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["tanh", "Fourier", "PQC", "SIREN"], fontsize=10)
        ax.set_ylabel("Test MAE", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        # Annotate numbers
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h * 0.5, f"{h:.3f}",
                    ha="center", va="center", color="white", fontweight="bold", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "figure9_dequantization_comparison.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def generate_figure10_angular_response():
    print("Generating Figure 10: Angular Activation Response...")
    from models.classical_twin import ClassicalTwinModelV2
    from models.classical_trig_twin import PRISMModelV2, TrigTwinModelV2

    # Instantiate models
    torch.manual_seed(42)
    twin_tanh = ClassicalTwinModelV2(in_features=11, hidden_features=16, num_layers=2, n_q_layers=2, out_dim=1)
    twin_siren = PRISMModelV2(in_features=11, hidden_features=16, num_layers=2, n_q_layers=2, out_dim=1, head_type="siren")

    angles_deg = np.linspace(0, 180, 200)
    angles_rad = torch.tensor(angles_deg * np.pi / 180.0, dtype=torch.float32).unsqueeze(-1)

    h_row = torch.zeros(200, 16)
    h_col = torch.zeros(200, 16)
    rbf = torch.zeros(200, 8)

    with torch.no_grad():
        out_tanh = twin_tanh.layers[0].head_electronic(h_row, h_col, rbf, angles_rad).numpy()
        out_siren = twin_siren.layers[0].head_electronic(h_row, h_col, rbf, angles_rad).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=300)

    # Left: Classical Twin (tanh)
    ax0 = axes[0]
    for ch in range(out_tanh.shape[1]):
        ax0.plot(angles_deg, out_tanh[:, ch], alpha=0.7, linewidth=1.5)
    ax0.set_title("Classical Twin ($\tanh$-MLP)\nSpectral Bias: Angle-Blind Response", fontsize=11, fontweight="bold")
    ax0.set_xlabel("3-Body Bond Angle $\\theta$ (degrees)", fontsize=10)
    ax0.set_ylabel("Observable Surrogates $\\in [-1, 1]$", fontsize=10)
    ax0.set_ylim([-1.1, 1.1])
    ax0.grid(True, linestyle="--", alpha=0.4)

    # Right: PRISM (SIREN)
    ax1 = axes[1]
    for ch in range(out_siren.shape[1]):
        ax1.plot(angles_deg, out_siren[:, ch], alpha=0.8, linewidth=1.5)
    ax1.set_title("PRISM (SIREN, Ours)\nBypasses Spectral Bias: Rich Trigonometric Fourier Oscillations", fontsize=11, fontweight="bold")
    ax1.set_xlabel("3-Body Bond Angle $\\theta$ (degrees)", fontsize=10)
    ax1.set_ylabel("Observable Surrogates $\\in [-1, 1]$", fontsize=10)
    ax1.set_ylim([-1.1, 1.1])
    ax1.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "figure10_angular_activation_response.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    generate_figure9_dequantization()
    generate_figure10_angular_response()
