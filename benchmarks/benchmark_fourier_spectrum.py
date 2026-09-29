"""
Fourier Spectrum Analysis: PQC vs Classical tanh-MLP (Strategy 1)
=================================================================
Empirically extracts and compares the frequency content of:
  1. Entangled V2 PQC (full EQ-GNN)
  2. Separable V2 PQC (no CNOTs)
  3. Classical tanh-MLP

All three models are trained on QM9 HOMO (N=1000, seed=42), then the
edge-level head outputs are recorded as functions of continuous distance
d_ij ∈ [0, 10] Å² (fixing node features and angle to training medians).
The discrete Fourier transform reveals the spectral structure.

Theory (Schuld, Sweke & Meyer, 2021, Phys. Rev. A):
  PQC with L layers of data re-uploading via R_y(x), R_z(x) computes:
    f(x) = Σ_{ω ∈ Ω} c_ω cos(ωx) + s_ω sin(ωx)
  where Ω = {sums/differences of eigenvalue gaps of encoding generators}.
  Classical tanh-MLP computes sigmoid-family functions — fundamentally
  different spectral structure.

Outputs:
  - paper/figures/figure_fourier_spectrum.png  (publication-quality)
  - results/fourier_spectrum_analysis.json

Usage:
  cd /Users/kkyn/Documents/Work/Quantum/qgnn
  .pixi/envs/default/bin/python benchmarks/benchmark_fourier_spectrum.py
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
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

from datasets.dataset_qm9 import get_qm9_splits
from models.eq_gnn import (
    EQGNNModelV2, GaussianRBF, compute_edge_mean_angles, N_RBF, n_qubits
)
from models.egnn_baseline import count_parameters

# Import separable + tanh models from the ablation benchmark
from benchmarks.benchmark_entanglement_ablation_v2 import (
    SeparableEQGNNModelV2, ClassicalTanhEQGNNModel
)


# =============================================================================
# 1. TRAIN ALL THREE MODELS
# =============================================================================

def train_model(model, target="homo", seed=42, num_train=1000, epochs=15):
    """Train a model and return it (trained, in eval mode)."""
    train_set, val_set, test_set, (mean, std) = get_qm9_splits(
        root="./data/QM9", target=target,
        num_train=num_train, num_val=250, num_test=500, seed=seed
    )
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)

    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            y_norm = (batch.y - mean) / std
            pred = model(batch.x, batch.pos, batch.edge_index, batch.batch)
            loss = criterion(pred, y_norm)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()

    # Also compute test MAE for reporting
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False)
    total_mae = 0.0
    with torch.no_grad():
        for batch in test_loader:
            pred_norm = model(batch.x, batch.pos, batch.edge_index, batch.batch)
            pred = pred_norm * std + mean
            total_mae += torch.abs(pred - batch.y).sum().item()
    test_mae = total_mae / len(test_set)

    return model, mean, std, test_mae


# =============================================================================
# 2. EXTRACT EDGE-LEVEL RESPONSE FUNCTIONS
# =============================================================================

@torch.no_grad()
def extract_head_response_vs_distance(head, rbf_module, h_row_fixed, h_col_fixed,
                                      angle_fixed, d_range, n_points=1024):
    """
    Sweep d_ij over a continuous range and record the head's 8-dim output.

    Args:
        head: QuantumHead, SeparableQuantumHead, or ClassicalTanhHead (trained)
        rbf_module: trained GaussianRBF to convert d_ij → RBF features
        h_row_fixed: [1, hidden] fixed source node embedding
        h_col_fixed: [1, hidden] fixed target node embedding
        angle_fixed: scalar, fixed 3-body angle
        d_range: (d_min, d_max)
        n_points: number of distance samples

    Returns:
        distances: [n_points] array
        outputs: [n_points, 8] array of head outputs
    """
    d_min, d_max = d_range
    distances = torch.linspace(d_min, d_max, n_points).unsqueeze(1)  # [N, 1]
    rbf_features = rbf_module(distances)  # [N, K]
    angles = torch.full((n_points, 1), angle_fixed)  # [N, 1]

    h_r = h_row_fixed.expand(n_points, -1)  # [N, hidden]
    h_c = h_col_fixed.expand(n_points, -1)  # [N, hidden]

    outputs = head(h_r, h_c, rbf_features, angles)  # [N, 8]

    return distances.squeeze().numpy(), outputs.numpy()


@torch.no_grad()
def extract_head_response_vs_angle(head, rbf_module, h_row_fixed, h_col_fixed,
                                   d_fixed, angle_range=(0, np.pi), n_points=1024):
    """Sweep angle over [0, π] with fixed distance."""
    a_min, a_max = angle_range
    angles_arr = torch.linspace(a_min, a_max, n_points).unsqueeze(1)  # [N, 1]
    d_ij = torch.full((n_points, 1), d_fixed)
    rbf_features = rbf_module(d_ij)  # [N, K]

    h_r = h_row_fixed.expand(n_points, -1)
    h_c = h_col_fixed.expand(n_points, -1)

    outputs = head(h_r, h_c, rbf_features, angles_arr)  # [N, 8]

    return angles_arr.squeeze().numpy(), outputs.numpy()


# =============================================================================
# 3. FOURIER ANALYSIS
# =============================================================================

def compute_power_spectrum(signal, sampling_rate):
    """
    Compute one-sided power spectrum via FFT.

    Returns:
        freqs: frequency axis
        power: |FFT|² normalized
    """
    n = len(signal)
    fft_vals = np.fft.rfft(signal - np.mean(signal))  # remove DC
    power = np.abs(fft_vals) ** 2
    power = power / np.max(power + 1e-12)  # normalize to [0, 1]
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
    return freqs, power


def compute_spectral_entropy(power):
    """Shannon entropy of normalized power spectrum (higher = more diffuse)."""
    p = power / (np.sum(power) + 1e-12)
    p = p[p > 1e-12]
    return -np.sum(p * np.log2(p))


def compute_spectral_concentration(power, top_k=5):
    """Fraction of total power in the top-k frequency bins."""
    sorted_p = np.sort(power)[::-1]
    return np.sum(sorted_p[:top_k]) / (np.sum(power) + 1e-12)


# =============================================================================
# 4. PLOTTING
# =============================================================================

def plot_fourier_analysis(results, save_path):
    """Create publication-quality 2×3 figure."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    model_configs = [
        ("entangled_v2", "Q-PRISM\n(Entangled PQC)", "#2196F3"),
        ("separable_v2", "Separable Q-PRISM\n(No Entanglement)", "#FF9800"),
        ("tanh_mlp", "Classical\ntanh-MLP", "#F44336"),
    ]

    # --- Row 1: Response functions f(d) ---
    for col, (key, label, color) in enumerate(model_configs):
        ax = axes[0, col]
        data = results[key]
        distances = np.array(data["distances"])
        outputs = np.array(data["outputs_vs_distance"])

        # Plot first 4 outputs (⟨Z_0⟩ through ⟨Z_3⟩)
        for q in range(4):
            ax.plot(distances, outputs[:, q], linewidth=1.2, alpha=0.85,
                    label=f"Out {q}")

        ax.set_xlabel("Distance $d_{ij}$ (Å²)", fontsize=11)
        ax.set_ylabel("Edge output value", fontsize=11)
        ax.set_title(label, fontsize=12, fontweight="bold", color=color)
        ax.set_ylim(-1.05, 1.05)
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
        ax.legend(fontsize=8, loc="lower right", ncol=2)
        ax.grid(True, alpha=0.3)

    # --- Row 2: Power spectra ---
    for col, (key, label, color) in enumerate(model_configs):
        ax = axes[1, col]
        data = results[key]

        freqs = np.array(data["freqs_distance"])
        # Average power spectrum across all 8 output channels
        all_powers = np.array(data["power_spectra_distance"])  # [8, n_freqs]
        mean_power = np.mean(all_powers, axis=0)

        # Plot individual channels as thin lines
        for q in range(min(4, all_powers.shape[0])):
            ax.plot(freqs[:60], all_powers[q, :60],
                    linewidth=0.8, alpha=0.4, color=color)

        # Plot mean power as thick line
        ax.plot(freqs[:60], mean_power[:60],
                linewidth=2.5, color=color, label="Mean power")

        # Annotate metrics
        entropy = data["spectral_entropy_distance"]
        concentration = data["spectral_concentration_distance"]
        ax.text(0.95, 0.95,
                f"Entropy: {entropy:.2f}\nConc.(top-5): {concentration:.1%}",
                transform=ax.transAxes, fontsize=9,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_xlabel("Frequency (cycles / Å²)", fontsize=11)
        ax.set_ylabel("Normalized Power", fontsize=11)
        ax.set_title(f"Power Spectrum — {label.split(chr(10))[0]}", fontsize=11, color=color)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Fourier Spectrum Analysis: Quantum vs Classical Edge Functions\n"
        "(Schuld, Sweke & Meyer 2021 — Trigonometric vs Sigmoid Basis)",
        fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"📊 Figure saved: {save_path}")


def plot_angular_response(results, save_path):
    """Supplementary figure: response as function of bond angle."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))

    model_configs = [
        ("entangled_v2", "Q-PRISM (Entangled)", "#2196F3"),
        ("separable_v2", "Separable Q-PRISM (No CNOT)", "#FF9800"),
        ("tanh_mlp", "Classical tanh-MLP", "#F44336"),
    ]

    for col, (key, label, color) in enumerate(model_configs):
        ax = axes[col]
        data = results[key]
        angles = np.array(data["angles"])
        outputs = np.array(data["outputs_vs_angle"])

        for q in range(4):
            ax.plot(np.degrees(angles), outputs[:, q], linewidth=1.2, alpha=0.85,
                    label=f"Out {q}")

        ax.set_xlabel("Bond angle θ (degrees)", fontsize=14)
        ax.set_ylabel("Edge output value", fontsize=14)
        ax.set_title(label, fontsize=14, fontweight="bold", color=color)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlim(0, 180)
        ax.tick_params(axis='both', labelsize=12)
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
        ax.legend(fontsize=10, loc="lower right", ncol=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Edge Response vs. 3-Body Bond Angle (fixed distance = 2.25 Å²)",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"📊 Figure saved: {save_path}")


def plot_spectral_summary(results, save_path):
    """Bar chart comparing spectral entropy and concentration across models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    labels = ["Entangled\nPQC", "Separable\nPQC", "Classical\ntanh-MLP"]
    colors = ["#2196F3", "#FF9800", "#F44336"]
    keys = ["entangled_v2", "separable_v2", "tanh_mlp"]

    entropies = [results[k]["spectral_entropy_distance"] for k in keys]
    concentrations = [results[k]["spectral_concentration_distance"] for k in keys]

    bars1 = ax1.bar(labels, entropies, color=colors, edgecolor="black", linewidth=0.8)
    ax1.set_ylabel("Spectral Entropy (bits)", fontsize=12)
    ax1.set_title("Spectral Entropy\n(lower = more structured)", fontsize=12, fontweight="bold")
    for bar, val in zip(bars1, entropies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{val:.2f}", ha="center", fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")

    bars2 = ax2.bar(labels, concentrations, color=colors, edgecolor="black", linewidth=0.8)
    ax2.set_ylabel("Top-5 Frequency Concentration", fontsize=12)
    ax2.set_title("Spectral Concentration\n(higher = sharper peaks)", fontsize=12, fontweight="bold")
    for bar, val in zip(bars2, concentrations):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.1%}", ha="center", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Fourier Spectral Structure: Quantum (Trigonometric) vs Classical (Sigmoid)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"📊 Figure saved: {save_path}")


# =============================================================================
# 5. MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("FOURIER SPECTRUM ANALYSIS (Schuld et al. 2021 Framework)")
    print("=" * 70)

    TARGET = "homo"
    SEED = 42
    N_POINTS = 1024
    D_RANGE = (0.1, 10.0)  # Å² — typical QM9 interatomic distance² range
    D_FIXED_FOR_ANGLE = 2.25  # Å² — typical C-C bond distance²

    results = {}

    # --- Step 1: Train all three models ---
    model_configs = [
        ("entangled_v2", "Q-PRISM (Entangled)",
         lambda: EQGNNModelV2(in_features=11, hidden_features=16, num_layers=2,
                              n_q_layers=2, out_dim=1)),
        ("separable_v2", "Separable Q-PRISM (No CNOTs)",
         lambda: SeparableEQGNNModelV2(in_features=11, hidden_features=16, num_layers=2,
                                       n_q_layers=2, out_dim=1)),
        ("tanh_mlp", "Classical tanh-MLP",
         lambda: ClassicalTanhEQGNNModel(in_features=11, hidden_features=16,
                                         num_layers=2, out_dim=1)),
    ]

    trained_models = {}
    for key, label, factory in model_configs:
        print(f"\n🔧 Training: {label}...")
        t0 = time.time()
        model = factory()
        model, mean, std, test_mae = train_model(model, target=TARGET, seed=SEED)
        elapsed = time.time() - t0
        trained_models[key] = model
        print(f"   Done ({elapsed:.1f}s) | Test MAE = {test_mae:.4f} eV | Params = {count_parameters(model):,}")

    # --- Step 2: Extract representative fixed inputs ---
    # Use training data median node embedding after the embedding layer
    print("\n📐 Extracting representative node embeddings from training data...")
    train_set, _, _, _ = get_qm9_splits(
        root="./data/QM9", target=TARGET, num_train=100, num_val=50, num_test=50, seed=SEED
    )
    loader = DataLoader(train_set, batch_size=100, shuffle=False)
    batch = next(iter(loader))

    # Get embeddings from the entangled model's embedding layer
    ref_model = trained_models["entangled_v2"]
    with torch.no_grad():
        h_embedded = ref_model.embedding(batch.x)  # [N, 16]

    # Use median node embedding as the representative fixed input
    h_median = h_embedded.median(dim=0).values.unsqueeze(0)  # [1, 16]
    print(f"   Median node embedding: shape={h_median.shape}, norm={h_median.norm():.3f}")

    # Median bond angle from training data
    with torch.no_grad():
        angles_train = compute_edge_mean_angles(batch.pos, batch.edge_index)
    angle_median = float(angles_train.median())
    print(f"   Median bond angle: {np.degrees(angle_median):.1f}°")

    # --- Step 3: Sweep distance and angle for each model ---
    for key, label, _ in model_configs:
        print(f"\n📈 Extracting response functions: {label}...")
        model = trained_models[key]
        layer0 = model.layers[0]  # First GNN layer

        head = layer0.head_electronic  # Use the electronic head
        rbf = layer0.rbf

        # Distance sweep
        distances, outputs_d = extract_head_response_vs_distance(
            head, rbf, h_median, h_median, angle_median, D_RANGE, N_POINTS
        )

        # Angle sweep
        angles, outputs_a = extract_head_response_vs_angle(
            head, rbf, h_median, h_median, D_FIXED_FOR_ANGLE,
            angle_range=(0, np.pi), n_points=N_POINTS
        )

        # Fourier analysis of distance response
        sampling_rate = N_POINTS / (D_RANGE[1] - D_RANGE[0])
        all_freqs = None
        all_powers = []
        for ch in range(outputs_d.shape[1]):
            freqs, power = compute_power_spectrum(outputs_d[:, ch], sampling_rate)
            all_freqs = freqs
            all_powers.append(power)
        all_powers = np.array(all_powers)  # [8, n_freqs]

        # Spectral metrics (averaged over channels)
        mean_power = np.mean(all_powers, axis=0)
        entropy = compute_spectral_entropy(mean_power)
        concentration = compute_spectral_concentration(mean_power, top_k=5)

        results[key] = {
            "label": label,
            "distances": distances.tolist(),
            "outputs_vs_distance": outputs_d.tolist(),
            "angles": angles.tolist(),
            "outputs_vs_angle": outputs_a.tolist(),
            "freqs_distance": all_freqs.tolist(),
            "power_spectra_distance": all_powers.tolist(),
            "spectral_entropy_distance": float(entropy),
            "spectral_concentration_distance": float(concentration),
        }

        print(f"   Spectral entropy: {entropy:.2f} bits")
        print(f"   Top-5 concentration: {concentration:.1%}")

    # --- Step 4: Generate plots ---
    print("\n🎨 Generating publication figures...")
    os.makedirs(PROJECT_ROOT / "paper" / "figures", exist_ok=True)
    os.makedirs(PROJECT_ROOT / "results", exist_ok=True)

    plot_fourier_analysis(
        results,
        PROJECT_ROOT / "paper" / "figures" / "figure_fourier_spectrum.png"
    )

    plot_angular_response(
        results,
        PROJECT_ROOT / "paper" / "figures" / "figure_angular_response.png"
    )

    plot_spectral_summary(
        results,
        PROJECT_ROOT / "paper" / "figures" / "figure_spectral_summary.png"
    )

    # --- Step 5: Save JSON results (strip large arrays for readability) ---
    results_compact = {}
    for key in results:
        results_compact[key] = {
            "label": results[key]["label"],
            "spectral_entropy_distance": results[key]["spectral_entropy_distance"],
            "spectral_concentration_distance": results[key]["spectral_concentration_distance"],
            "n_distance_points": len(results[key]["distances"]),
            "distance_range": list(D_RANGE),
        }

    save_path = PROJECT_ROOT / "results" / "fourier_spectrum_analysis.json"
    with open(save_path, "w") as f:
        json.dump(results_compact, f, indent=2)
    print(f"\n💾 Results saved: {save_path}")

    # --- Step 6: Print summary ---
    print("\n" + "=" * 70)
    print("SPECTRAL STRUCTURE SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<30s} {'Entropy (bits)':<18s} {'Top-5 Conc.':<15s}")
    print("-" * 63)
    for key, label, _ in model_configs:
        r = results[key]
        print(f"{label:<30s} {r['spectral_entropy_distance']:<18.2f} {r['spectral_concentration_distance']:<15.1%}")

    print("\n📖 Interpretation (Schuld et al. 2021):")
    print("   • Lower entropy = more structured frequency content (sharper peaks)")
    print("   • Higher concentration = fewer dominant frequencies (cleaner signal)")
    print("   • PQCs should show LOWER entropy & HIGHER concentration than tanh-MLP")
    print("     because trigonometric gates produce discrete frequency spectra,")
    print("     while sigmoid/tanh produces diffuse, broadband spectra.")

    print("\n✅ Fourier spectrum analysis complete!")


if __name__ == "__main__":
    main()
