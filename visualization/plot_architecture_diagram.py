"""
Q-PRISM Architecture Diagram Generator (Publication Quality)
============================================================
Generates a comprehensive vector-quality diagram illustrating:
1. 3D Molecular Graph Input (Coordinates x_i, Features h_i)
2. Invariant Distance Computation d_ij = ||x_i - x_j||^2
3. 4-Qubit PQC with Multi-Axis Data Re-Uploading & Ring CNOT Entanglement
4. Pauli-Z Expectation Message Generation (m_ij, phi_x^Q)
5. E(3)-Equivariant Coordinate & Invariant Feature Updates
6. Global Molecular Property Prediction

Saves to:
- results/figure_architecture.png
- paper/figures/figure_architecture.png
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle, FancyArrowPatch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def draw_architecture_diagram():
    fig = plt.figure(figsize=(16, 8.5), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    # Color Palette (Nature / Modern Scientific Style)
    c_blue = "#1f77b4"
    c_light_blue = "#e8f1f8"
    c_purple = "#6f42c1"
    c_light_purple = "#f3eefa"
    c_orange = "#d95f02"
    c_light_orange = "#fef2e8"
    c_green = "#2ca02c"
    c_light_green = "#eef8ee"
    c_gray = "#495057"
    c_light_gray = "#f8f9fa"
    c_dark = "#212529"

    # Title Banner
    ax.text(8.0, 8.1, "Q-PRISM: Equivariant Quantum Graph Neural Network Architecture", 
            ha="center", va="center", fontsize=16, fontweight="bold", color=c_dark)
    ax.text(8.0, 7.7, "Embedding Exact Continuous E(3) Physical Symmetries into Parameterized Quantum Circuits", 
            ha="center", va="center", fontsize=11, fontstyle="italic", color=c_gray)

    # ─────────────────────────────────────────────────────────────────────────
    # Panel 1: 3D Molecular Graph Input (x ∈ [0.5, 3.8], y ∈ [0.8, 7.2])
    # ─────────────────────────────────────────────────────────────────────────
    box1 = FancyBboxPatch((0.5, 0.8), 3.3, 6.4, boxstyle="round,pad=0.15", 
                          fc=c_light_blue, ec=c_blue, lw=1.8)
    ax.add_patch(box1)
    ax.text(2.15, 6.85, "1. 3D Geometric Graph Input", ha="center", va="center", 
            fontsize=12, fontweight="bold", color=c_blue)

    # Molecule Sketch (Atoms and 3D Coordinates)
    atoms = [
        (1.4, 5.4, "C", "#2b8cbe"),
        (2.9, 5.7, "O", "#e41a1c"),
        (1.8, 3.8, "C", "#2b8cbe"),
        (2.8, 2.7, "H", "#4daf4a"),
        (0.9, 2.9, "H", "#4daf4a")
    ]
    edges = [(0, 1), (0, 2), (2, 3), (2, 4)]

    for (i, j) in edges:
        x1, y1, _, _ = atoms[i]
        x2, y2, _, _ = atoms[j]
        ax.plot([x1, x2], [y1, y2], color="#999999", lw=2.5, zorder=1)

    for (x, y, sym, col) in atoms:
        circle = plt.Circle((x, y), 0.28, color=col, ec="black", lw=1.2, zorder=2)
        ax.add_patch(circle)
        ax.text(x, y, sym, ha="center", va="center", color="white", fontweight="bold", fontsize=10, zorder=3)

    # Annotations for coordinates and features
    ax.text(2.15, 2.0, r"$\mathbf{x}_i \in \mathbb{R}^3$ (Spatial Coordinates)", ha="center", fontsize=9.5, fontweight="bold")
    ax.text(2.15, 1.5, r"$\mathbf{h}_i \in \mathbb{R}^d$ (Atomic Features)", ha="center", fontsize=9.5, fontweight="bold")
    ax.text(2.15, 1.05, r"$d_{ij} = \|\mathbf{x}_i - \mathbf{x}_j\|^2$ (Invariant)", ha="center", fontsize=9, color=c_purple, fontweight="bold")

    # Arrow 1 -> 2
    arrow1 = FancyArrowPatch((3.85, 4.0), (4.35, 4.0), arrowstyle="-|>", mutation_scale=18, lw=2.0, color=c_dark)
    ax.add_patch(arrow1)

    # ─────────────────────────────────────────────────────────────────────────
    # Panel 2: 4-Qubit PQC with Multi-Axis Re-Uploading (x ∈ [4.4, 11.2], y ∈ [0.8, 7.2])
    # ─────────────────────────────────────────────────────────────────────────
    box2 = FancyBboxPatch((4.4, 0.8), 6.8, 6.4, boxstyle="round,pad=0.15", 
                          fc=c_light_purple, ec=c_purple, lw=1.8)
    ax.add_patch(box2)
    ax.text(7.8, 6.85, "2. 4-Qubit PQC Edge Processor with Multi-Axis Data Re-Uploading", 
            ha="center", va="center", fontsize=12, fontweight="bold", color=c_purple)

    # Circuit Qubit Wires
    wire_y = [5.7, 4.6, 3.5, 2.4]
    qubit_labels = [r"$|0\rangle_0$", r"$|0\rangle_1$", r"$|0\rangle_2$", r"$|0\rangle_3$"]
    input_labels = [r"$R_y(h_{i,0})$", r"$R_y(h_{i,1})$", r"$R_y(h_{j,0})$", r"$R_y(h_{j,1})$"]

    for i, y in enumerate(wire_y):
        ax.plot([4.8, 10.8], [y, y], color=c_dark, lw=1.5, zorder=1)
        ax.text(4.65, y, qubit_labels[i], ha="right", va="center", fontsize=10, fontweight="bold")

    # Stage A: Multi-Axis Feature Encoding (Ry)
    for i, y in enumerate(wire_y):
        rect = FancyBboxPatch((5.0, y - 0.35), 0.95, 0.7, boxstyle="square,pad=0.02", 
                              fc="#e0f2fe", ec="#0284c7", lw=1.2, zorder=2)
        ax.add_patch(rect)
        ax.text(5.475, y, input_labels[i], ha="center", va="center", fontsize=8, fontweight="bold")

    ax.text(5.475, 6.35, "Feature Encoding", ha="center", fontsize=8.5, fontweight="bold", color="#0284c7")

    # Stage B: Invariant Distance Re-Uploading (Rz(w_d * d_ij))
    for i, y in enumerate(wire_y):
        rect = FancyBboxPatch((6.25, y - 0.35), 1.05, 0.7, boxstyle="square,pad=0.02", 
                              fc="#fef3c7", ec="#d97706", lw=1.2, zorder=2)
        ax.add_patch(rect)
        ax.text(6.775, y, r"$R_z(w_d d_{ij})$", ha="center", va="center", fontsize=7.5, fontweight="bold")

    ax.text(6.775, 6.35, "Distance Re-Upload", ha="center", fontsize=8.5, fontweight="bold", color="#d97706")

    # Stage C: Variational Rotations (RX, RY, RZ)
    for i, y in enumerate(wire_y):
        rect = FancyBboxPatch((7.6, y - 0.35), 1.05, 0.7, boxstyle="square,pad=0.02", 
                              fc="#dcfce7", ec="#16a34a", lw=1.2, zorder=2)
        ax.add_patch(rect)
        ax.text(8.125, y, r"$U(\boldsymbol{\theta}_{l,q})$", ha="center", va="center", fontsize=8.5, fontweight="bold")

    ax.text(8.125, 6.35, "Variational Ansatz", ha="center", fontsize=8.5, fontweight="bold", color="#16a34a")

    # Stage D: Circular Ring CNOT Entanglement
    cnot_x = [8.95, 9.3, 9.65, 10.0]
    cnot_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for idx, (q_c, q_t) in enumerate(cnot_pairs):
        cx = cnot_x[idx]
        y_c = wire_y[q_c]
        y_t = wire_y[q_t]
        ax.plot([cx, cx], [y_c, y_t], color=c_purple, lw=1.8, zorder=2)
        # Control dot
        dot = plt.Circle((cx, y_c), 0.08, color=c_purple, zorder=3)
        ax.add_patch(dot)
        # Target target circle
        target_circle = plt.Circle((cx, y_t), 0.14, color="white", ec=c_purple, lw=1.5, zorder=3)
        ax.add_patch(target_circle)
        ax.text(cx, y_t, "+", ha="center", va="center", color=c_purple, fontsize=10, fontweight="bold", zorder=4)

    ax.text(9.475, 6.35, "Ring Entanglement", ha="center", fontsize=8.5, fontweight="bold", color=c_purple)

    # Stage E: Pauli-Z Measurement Readouts
    for i, y in enumerate(wire_y):
        meas = FancyBboxPatch((10.35, y - 0.3), 0.55, 0.6, boxstyle="round,pad=0.02", 
                              fc="#f3f4f6", ec=c_dark, lw=1.2, zorder=2)
        ax.add_patch(meas)
        ax.text(10.625, y, r"$\langle Z \rangle$", ha="center", va="center", fontsize=8, fontweight="bold")

    # Output Expectation Mapping Note
    ax.text(7.8, 1.4, r"Outputs: $\mathbf{m}_{ij} = [\langle Z_0 \rangle, \langle Z_1 \rangle, \langle Z_2 \rangle]^T \in \mathbb{R}^3$ (Invariant Message)", 
            ha="center", fontsize=9.5, fontweight="bold", color=c_dark)
    ax.text(7.8, 1.0, r"$\phi_x^Q(\mathbf{m}_{ij}) = \langle Z_3 \rangle \in [-1, 1]$ (Equivariant Coordinate Weight)", 
            ha="center", fontsize=9.5, fontweight="bold", color=c_orange)

    # Arrow 2 -> 3
    arrow2 = FancyArrowPatch((11.25, 4.0), (11.75, 4.0), arrowstyle="-|>", mutation_scale=18, lw=2.0, color=c_dark)
    ax.add_patch(arrow2)

    # ─────────────────────────────────────────────────────────────────────────
    # Panel 3: Equivariant Update & Readout (x ∈ [11.8, 15.5], y ∈ [0.8, 7.2])
    # ─────────────────────────────────────────────────────────────────────────
    box3 = FancyBboxPatch((11.8, 0.8), 3.7, 6.4, boxstyle="round,pad=0.15", 
                          fc=c_light_green, ec=c_green, lw=1.8)
    ax.add_patch(box3)
    ax.text(13.65, 6.85, "3. Equivariant Update & Readout", ha="center", va="center", 
            fontsize=12, fontweight="bold", color=c_green)

    # Coordinate Update Box
    c_box = FancyBboxPatch((12.05, 4.9), 3.2, 1.4, boxstyle="round,pad=0.08", 
                           fc="white", ec=c_orange, lw=1.5)
    ax.add_patch(c_box)
    ax.text(13.65, 5.95, r"Coordinate Equivariant Update", ha="center", fontsize=9, fontweight="bold", color=c_orange)
    ax.text(13.65, 5.35, r"$\mathbf{x}_i^{l+1} = \mathbf{x}_i^l + \frac{1}{|\mathcal{N}(i)|} \sum_{j} (\mathbf{x}_i^l - \mathbf{x}_j^l) \phi_x^Q$", 
            ha="center", fontsize=8.5, fontweight="bold")

    # Feature Update Box
    f_box = FancyBboxPatch((12.05, 3.1), 3.2, 1.4, boxstyle="round,pad=0.08", 
                           fc="white", ec=c_blue, lw=1.5)
    ax.add_patch(f_box)
    ax.text(13.65, 4.15, r"Node Feature Invariant Update", ha="center", fontsize=9, fontweight="bold", color=c_blue)
    ax.text(13.65, 3.55, r"$\mathbf{h}_i^{l+1} = \mathbf{h}_i^l + \text{MLP}([\mathbf{h}_i^l, \sum_j \mathbf{m}_{ij}])$", 
            ha="center", fontsize=8.5, fontweight="bold")

    # Global Readout Box
    r_box = FancyBboxPatch((12.05, 1.2), 3.2, 1.5, boxstyle="round,pad=0.08", 
                           fc="white", ec=c_green, lw=1.5)
    ax.add_patch(r_box)
    ax.text(13.65, 2.35, r"Invariant Global Readout", ha="center", fontsize=9, fontweight="bold", color=c_green)
    ax.text(13.65, 1.85, r"$\hat{y} = \text{Head}\left(\frac{1}{N} \sum_{i=1}^N \mathbf{h}_i^L\right)$", 
            ha="center", fontsize=9.5, fontweight="bold")
    ax.text(13.65, 1.4, r"Guarantees: $\hat{y}(\mathbf{Q}\mathbf{x} + \mathbf{g}) \equiv \hat{y}(\mathbf{x})$", 
            ha="center", fontsize=8, color=c_purple, fontweight="bold")

    plt.tight_layout()
    save_path = PROJECT_ROOT / "results" / "figure_architecture.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), bbox_inches="tight")
    plt.close()
    print(f"✅ Generated Architecture Diagram: {save_path}")

    # Copy to paper figures
    paper_fig_dir = PROJECT_ROOT / "paper" / "figures"
    paper_fig_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(save_path), str(paper_fig_dir / "figure_architecture.png"))
    print("✅ Copied to paper/figures/figure_architecture.png")


if __name__ == "__main__":
    draw_architecture_diagram()
