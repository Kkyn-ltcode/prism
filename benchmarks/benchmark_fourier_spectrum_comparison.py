"""Fourier spectrum comparison between PQC and PRISM (SIREN) models."""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import sys
from pathlib import Path
import json
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.eq_gnn import EQGNNModelV2, GaussianRBF, compute_edge_mean_angles, N_RBF
from models.classical_twin import ClassicalTwinModelV2
from models.classical_trig_twin import PRISMModelV2
from models.egnn_baseline import count_parameters
from datasets.dataset_qm9 import get_qm9_splits
from torch_geometric.loader import DataLoader

# Import separable model from ablation benchmark
from benchmarks.benchmark_entanglement_ablation_v2 import SeparableEQGNNModelV2

# Reuse helpers from existing Fourier benchmark
from benchmarks.benchmark_fourier_spectrum import (
    extract_head_response_vs_distance,
    extract_head_response_vs_angle,
    compute_power_spectrum,
    compute_spectral_entropy,
    compute_spectral_concentration,
    train_model,
)


def cosine_similarity_spectra(power_a, power_b):
    """Cosine similarity between two power spectra."""
    a = np.array(power_a).flatten()
    b = np.array(power_b).flatten()
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(dot / (norm_a * norm_b))


def wasserstein_spectra(power_a, power_b):
    """Earth Mover's Distance between normalized power spectra."""
    a = np.array(power_a).flatten()
    b = np.array(power_b).flatten()
    # Normalize to probability distributions
    a_norm = a / (np.sum(a) + 1e-12)
    b_norm = b / (np.sum(b) + 1e-12)
    return float(wasserstein_distance(np.arange(len(a_norm)), np.arange(len(b_norm)),
                                      u_weights=a_norm, v_weights=b_norm))


def dominant_freq_overlap(power_a, power_b, top_k=5):
    """Fraction of A's top-k frequency bins that appear in B's top-k."""
    a = np.array(power_a).flatten()
    b = np.array(power_b).flatten()
    top_a = set(np.argsort(a)[-top_k:])
    top_b = set(np.argsort(b)[-top_k:])
    return float(len(top_a & top_b) / top_k)


MODEL_CONFIGS = [
    ("Q-PRISM (Entangled PQC)", "#2196F3"),
    ("Separable Q-PRISM (No CNOTs)", "#FF9800"),
    ("Classical Twin (tanh-MLP)", "#F44336"),
    ("PRISM (SIREN)", "#4CAF50"),
]


def plot_angular_response(all_results, angles, save_path):
    """1×4 angular response subplot."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for col, (label, color) in enumerate(MODEL_CONFIGS):
        ax = axes[col]
        outputs = np.array(all_results[label]["outputs_vs_angle"])

        for q in range(min(4, outputs.shape[1])):
            ax.plot(np.degrees(angles), outputs[:, q], linewidth=1.2, alpha=0.85,
                    label=f"Out {q}")

        ax.set_xlabel("Bond angle θ (degrees)", fontsize=11)
        ax.set_ylabel("Edge output value", fontsize=11)
        ax.set_title(label.split("(")[0].strip(), fontsize=12, fontweight="bold", color=color)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlim(0, 180)
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
        ax.legend(fontsize=8, loc="lower right", ncol=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Edge Response vs. Bond Angle: PQC vs PRISM (SIREN) Comparison",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {save_path}")


def plot_power_spectra(all_results, save_path):
    """2×2 power spectra subplot."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for idx, (label, color) in enumerate(MODEL_CONFIGS):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]
        data = all_results[label]

        freqs = np.array(data["freqs_angle"])
        all_powers = np.array(data["power_spectra_angle"])  # [8, n_freqs]
        mean_power = np.mean(all_powers, axis=0)

        # Individual channels as thin lines
        for q in range(min(4, all_powers.shape[0])):
            ax.plot(freqs[:60], all_powers[q, :60],
                    linewidth=0.8, alpha=0.4, color=color)

        # Mean power as thick line
        ax.plot(freqs[:60], mean_power[:60],
                linewidth=2.5, color=color, label="Mean power")

        # Annotate metrics
        entropy = data["spectral_entropy"]
        concentration = data["spectral_concentration"]
        ax.text(0.95, 0.95,
                f"Entropy: {entropy:.2f}\nConc.(top-5): {concentration:.1%}",
                transform=ax.transAxes, fontsize=9,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_xlabel("Frequency (cycles / rad)", fontsize=11)
        ax.set_ylabel("Normalized Power", fontsize=11)
        ax.set_title(label, fontsize=11, color=color, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Power Spectra: PQC vs PRISM (SIREN)\n"
                 "(Angular response, fixed d=2.25 Å²)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {save_path}")


def plot_similarity_matrix(sim_matrix, model_labels, save_path):
    """Heatmap of pairwise cosine similarities."""
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(sim_matrix, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Cosine Similarity", fontsize=12)

    ax.set_xticks(range(len(model_labels)))
    ax.set_yticks(range(len(model_labels)))
    short_labels = [l.split("(")[0].strip() for l in model_labels]
    ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(short_labels, fontsize=10)

    # Annotate cells
    for i in range(len(model_labels)):
        for j in range(len(model_labels)):
            ax.text(j, i, f"{sim_matrix[i, j]:.2f}",
                    ha="center", va="center", fontsize=11, fontweight="bold",
                    color="white" if sim_matrix[i, j] > 0.6 else "black")

    ax.set_title("Spectral Cosine Similarity Matrix\n"
                 "(Angular response power spectra, averaged over channels)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {save_path}")


def main():
    print("=" * 70)
    print("FOURIER SPECTRUM COMPARISON: PQC vs PRISM")
    print("=" * 70)

    TARGET = "homo"
    SEED = 42
    N_POINTS = 1024
    D_FIXED = 2.25  # Å² — typical C-C bond distance²

    results = {}

    # Train models
    model_factories = [
        ("Q-PRISM (Entangled PQC)",
         lambda: EQGNNModelV2(in_features=11, hidden_features=16, num_layers=2,
                               n_q_layers=2, out_dim=1)),
        ("Separable Q-PRISM (No CNOTs)",
         lambda: SeparableEQGNNModelV2(in_features=11, hidden_features=16, num_layers=2,
                                        n_q_layers=2, out_dim=1)),
        ("Classical Twin (tanh-MLP)",
         lambda: ClassicalTwinModelV2(in_features=11, hidden_features=16, num_layers=2,
                                      n_q_layers=2, out_dim=1)),
        ("PRISM (SIREN)",
         lambda: PRISMModelV2(in_features=11, hidden_features=16, num_layers=2,
                               n_q_layers=2, out_dim=1, head_type="siren", omega_0=30.0)),
    ]

    trained_models = {}
    for label, factory in model_factories:
        print(f"\nTraining: {label}...")
        t0 = time.time()
        model = factory()
        model, mean, std, test_mae = train_model(model, target=TARGET, seed=SEED)
        elapsed = time.time() - t0
        trained_models[label] = model
        print(f"   Done ({elapsed:.1f}s) | Test MAE = {test_mae:.4f} eV | "
              f"Params = {count_parameters(model):,}")

    # Extract representative fixed inputs
    print("\nExtracting representative node embeddings from training data...")
    train_set, _, _, _ = get_qm9_splits(
        root="./data/QM9", target=TARGET, num_train=100, num_val=50, num_test=50, seed=SEED
    )
    loader = DataLoader(train_set, batch_size=100, shuffle=False)
    batch = next(iter(loader))

    # Use median embedding from the first (entangled) model
    ref_model = trained_models["Q-PRISM (Entangled PQC)"]
    with torch.no_grad():
        h_embedded = ref_model.embedding(batch.x)  # [N, 16]
    h_median = h_embedded.median(dim=0).values.unsqueeze(0)  # [1, 16]
    print(f"   Median node embedding: shape={h_median.shape}, norm={h_median.norm():.3f}")

    # Median bond angle
    with torch.no_grad():
        angles_train = compute_edge_mean_angles(batch.pos, batch.edge_index)
    angle_median = float(angles_train.median())
    print(f"   Median bond angle: {np.degrees(angle_median):.1f}°")

    # Sweep angle for each model
    angles_shared = None
    for label, _ in model_factories:
        print(f"\nExtracting angular response: {label}...")
        model = trained_models[label]
        layer0 = model.layers[0]
        head = layer0.head_electronic
        rbf = layer0.rbf

        # Angular sweep (fixed distance)
        angles, outputs_a = extract_head_response_vs_angle(
            head, rbf, h_median, h_median, D_FIXED,
            angle_range=(0, np.pi), n_points=N_POINTS
        )
        angles_shared = angles

        # Fourier analysis of angular response
        sampling_rate = N_POINTS / np.pi  # cycles per radian
        all_freqs = None
        all_powers = []
        for ch in range(outputs_a.shape[1]):
            freqs, power = compute_power_spectrum(outputs_a[:, ch], sampling_rate)
            all_freqs = freqs
            all_powers.append(power)
        all_powers = np.array(all_powers)  # [8, n_freqs]

        # Spectral metrics (averaged over channels)
        mean_power = np.mean(all_powers, axis=0)
        entropy = compute_spectral_entropy(mean_power)
        concentration = compute_spectral_concentration(mean_power, top_k=5)

        results[label] = {
            "outputs_vs_angle": outputs_a.tolist(),
            "freqs_angle": all_freqs.tolist(),
            "power_spectra_angle": all_powers.tolist(),
            "mean_power_angle": mean_power.tolist(),
            "spectral_entropy": float(entropy),
            "spectral_concentration": float(concentration),
        }
        print(f"   Spectral entropy: {entropy:.2f} bits")
        print(f"   Top-5 concentration: {concentration:.1%}")

    # Compute pairwise similarity matrix
    print("\nComputing pairwise spectral similarities...")
    model_labels = [label for label, _ in model_factories]
    n_models = len(model_labels)
    sim_matrix = np.zeros((n_models, n_models))
    wass_matrix = np.zeros((n_models, n_models))
    overlap_matrix = np.zeros((n_models, n_models))

    for i, label_a in enumerate(model_labels):
        for j, label_b in enumerate(model_labels):
            power_a = results[label_a]["mean_power_angle"]
            power_b = results[label_b]["mean_power_angle"]
            sim_matrix[i, j] = cosine_similarity_spectra(power_a, power_b)
            wass_matrix[i, j] = wasserstein_spectra(power_a, power_b)
            overlap_matrix[i, j] = dominant_freq_overlap(power_a, power_b, top_k=5)

    # Key metric: PQC vs PRISM
    pqc_idx = 0  # Q-PRISM
    prism_idx = 3  # PRISM (SIREN)
    cos_sim = sim_matrix[pqc_idx, prism_idx]
    wass_dist = wass_matrix[pqc_idx, prism_idx]
    freq_overlap = overlap_matrix[pqc_idx, prism_idx]

    # Generate figures
    print("\nGenerating publication figures...")
    fig_dir = PROJECT_ROOT / "paper" / "figures"
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(PROJECT_ROOT / "results", exist_ok=True)

    plot_angular_response(
        results, angles_shared,
        fig_dir / "figure_pqc_vs_prism_angular_response.png"
    )
    plot_power_spectra(
        results,
        fig_dir / "figure_pqc_vs_prism_spectra.png"
    )
    plot_similarity_matrix(
        sim_matrix, model_labels,
        fig_dir / "figure_spectral_similarity_matrix.png"
    )

    # Save JSON results
    results_compact = {}
    for label in model_labels:
        results_compact[label] = {
            "spectral_entropy": results[label]["spectral_entropy"],
            "spectral_concentration": results[label]["spectral_concentration"],
        }
    results_compact["pairwise_metrics"] = {
        "cosine_similarity_matrix": sim_matrix.tolist(),
        "wasserstein_matrix": wass_matrix.tolist(),
        "overlap_matrix": overlap_matrix.tolist(),
        "model_labels": model_labels,
    }
    results_compact["pqc_vs_prism"] = {
        "cosine_similarity": float(cos_sim),
        "wasserstein_distance": float(wass_dist),
        "dominant_freq_overlap_top5": float(freq_overlap),
    }

    save_path = PROJECT_ROOT / "results" / "fourier_spectrum_pqc_vs_prism.json"
    with open(save_path, "w") as f:
        json.dump(results_compact, f, indent=2)
    print(f"\nResults saved: {save_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("SPECTRAL COMPARISON SUMMARY")
    print("=" * 70)

    print(f"\n{'Model':<35s} {'Entropy (bits)':<18s} {'Top-5 Conc.':<15s}")
    print("-" * 68)
    for label in model_labels:
        r = results[label]
        print(f"{label:<35s} {r['spectral_entropy']:<18.2f} {r['spectral_concentration']:<15.1%}")

    print(f"\n{'Pairwise Cosine Similarity Matrix':}")
    print(f"{'':>36s}", end="")
    short = [l.split("(")[0].strip()[:10] for l in model_labels]
    for s in short:
        print(f" {s:>10s}", end="")
    print()
    for i, label in enumerate(model_labels):
        print(f"  {short[i]:>34s}", end="")
        for j in range(n_models):
            print(f" {sim_matrix[i, j]:>10.3f}", end="")
        print()

    print(f"\nKey metric — Q-PRISM vs PRISM (SIREN):")
    print(f"   Cosine similarity:     {cos_sim:.4f}")
    print(f"   Wasserstein-1 distance: {wass_dist:.4f}")
    print(f"   Top-5 freq overlap:    {freq_overlap:.0%}")

    print()
    if cos_sim >= 0.8:
        print("   SIMILAR (cos >= 0.8) -> Supports periodic bias framing")
    elif cos_sim >= 0.5:
        print("   MODERATELY SIMILAR (0.5 <= cos < 0.8) -> Partial support; "
              "consider nuancing the narrative")
    else:
        print("   DIFFER (cos < 0.5) -> Models use "
              "different frequency structures")

    print("\nFourier spectrum comparison complete.")


if __name__ == "__main__":
    main()
