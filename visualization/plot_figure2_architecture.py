"""Generate publication architecture diagram for Q-PRISM."""


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

    # Panel 1: 3D Molecular Graph Input (x ∈ [0.5, 3.8], y ∈ [0.8, 7.2])
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
    ax.text(2.15, 2.2, r"$\mathbf{x}_i \in \mathbb{R}^3$ (Spatial Coordinates)", ha="center", fontsize=9, fontweight="bold")
    ax.text(2.15, 1.75, r"$\mathbf{h}_i \in \mathbb{R}^d$ (Atomic Features)", ha="center", fontsize=9, fontweight="bold")
    ax.text(2.15, 1.3, r"$\boldsymbol{\phi}(d_{ij}) \in \mathbb{R}^K$ (Gaussian RBF)", ha="center", fontsize=8.5, color="#d97706", fontweight="bold")
    ax.text(2.15, 0.95, r"$\bar{\theta}_{ij} = \text{MeanAngle}(i,j)$ (3-Body SO(3))", ha="center", fontsize=8.2, color=c_purple, fontweight="bold")

    # Arrow 1 -> 2
    arrow1 = FancyArrowPatch((3.85, 4.0), (4.35, 4.0), arrowstyle="-|>", mutation_scale=18, lw=2.0, color=c_dark)
    ax.add_patch(arrow1)

    # Panel 2: 2 Parallel 4-Qubit PQC Heads (x ∈ [4.4, 11.4], y ∈ [0.8, 7.2])
    box2 = FancyBboxPatch((4.4, 0.8), 7.0, 6.4, boxstyle="round,pad=0.15", 
                          fc=c_light_purple, ec=c_purple, lw=1.8)
    ax.add_patch(box2)
    ax.text(7.9, 6.9, "2. Two Parallel 4-Qubit PQC Edge Heads (Electronic & Topological)", 
            ha="center", va="center", fontsize=11, fontweight="bold", color=c_purple)
    ax.text(7.9, 6.55, r"Independent projections $\mathbf{W}_p^{(s)} \mathbf{h}_i \to \mathbb{R}^2$ with shared RBF $\boldsymbol{\phi}(d_{ij})$ and angular $\bar{\theta}_{ij}$ inputs",
            ha="center", va="center", fontsize=8, fontstyle="italic", color=c_gray)

    # Circuit Qubit Wires
    wire_y = [5.65, 4.65, 3.65, 2.65]
    qubit_labels = [r"$|0\rangle_0$", r"$|0\rangle_1$", r"$|0\rangle_2$", r"$|0\rangle_3$"]
    input_labels = [r"$R_y(h_{i,0}^{p})$", r"$R_y(h_{i,1}^{p})$", r"$R_y(h_{j,0}^{p})$", r"$R_y(h_{j,1}^{p})$"]

    for i, y in enumerate(wire_y):
        ax.plot([4.75, 11.0], [y, y], color=c_dark, lw=1.4, zorder=1)
        ax.text(4.65, y, qubit_labels[i], ha="right", va="center", fontsize=9.5, fontweight="bold")

    # Stage A: Multi-Axis Feature Encoding (Ry)
    for i, y in enumerate(wire_y):
        rect = FancyBboxPatch((4.85, y - 0.35), 0.95, 0.7, boxstyle="square,pad=0.02", 
                              fc="#e0f2fe", ec="#0284c7", lw=1.1, zorder=2)
        ax.add_patch(rect)
        ax.text(5.325, y, input_labels[i], ha="center", va="center", fontsize=7.2, fontweight="bold")

    ax.text(5.325, 6.2, "Feature Enc.", ha="center", fontsize=8, fontweight="bold", color="#0284c7")

    # Stage B: RBF Distance Re-Uploading (Rz(sum w_k phi_k))
    for i, y in enumerate(wire_y):
        rect = FancyBboxPatch((5.95, y - 0.35), 1.05, 0.7, boxstyle="square,pad=0.02", 
                              fc="#fef3c7", ec="#d97706", lw=1.1, zorder=2)
        ax.add_patch(rect)
        ax.text(6.475, y, r"$R_z(\mathbf{w}^T\boldsymbol{\phi})$", ha="center", va="center", fontsize=7.2, fontweight="bold")

    ax.text(6.475, 6.2, "RBF Distance", ha="center", fontsize=8, fontweight="bold", color="#d97706")

    # Stage C: 3-Body Angular Re-Uploading (Ry(theta_bar))
    for i, y in enumerate(wire_y):
        rect = FancyBboxPatch((7.15, y - 0.35), 0.95, 0.7, boxstyle="square,pad=0.02", 
                              fc="#fce7f3", ec="#db2777", lw=1.1, zorder=2)
        ax.add_patch(rect)
        ax.text(7.625, y, r"$R_y(\bar{\theta}_{ij})$", ha="center", va="center", fontsize=7.5, fontweight="bold")

    ax.text(7.625, 6.2, "Angular Re-Up", ha="center", fontsize=8, fontweight="bold", color="#db2777")

    # Stage D: Variational Rotations (Rx, Ry, Rz)
    for i, y in enumerate(wire_y):
        rect = FancyBboxPatch((8.25, y - 0.35), 0.95, 0.7, boxstyle="square,pad=0.02", 
                              fc="#dcfce7", ec="#16a34a", lw=1.1, zorder=2)
        ax.add_patch(rect)
        ax.text(8.725, y, r"$U_{\mathrm{rot}}(\boldsymbol{\theta})$", ha="center", va="center", fontsize=7.5, fontweight="bold")

    ax.text(8.725, 6.2, "Variational", ha="center", fontsize=8, fontweight="bold", color="#16a34a")

    # Stage E: Circular Ring CNOT Entanglement
    cnot_x = [9.35, 9.65, 9.95, 10.25]
    cnot_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for idx, (q_c, q_t) in enumerate(cnot_pairs):
        cx = cnot_x[idx]
        y_c = wire_y[q_c]
        y_t = wire_y[q_t]
        ax.plot([cx, cx], [y_c, y_t], color=c_purple, lw=1.6, zorder=2)
        # Control dot
        dot = plt.Circle((cx, y_c), 0.07, color=c_purple, zorder=3)
        ax.add_patch(dot)
        # Target circle
        target_circle = plt.Circle((cx, y_t), 0.12, color="white", ec=c_purple, lw=1.4, zorder=3)
        ax.add_patch(target_circle)
        ax.text(cx, y_t, "+", ha="center", va="center", color=c_purple, fontsize=9, fontweight="bold", zorder=4)

    ax.text(9.8, 6.2, "Ring CNOT", ha="center", fontsize=8, fontweight="bold", color=c_purple)

    # Stage F: Readout (Single & Correlators)
    for i, y in enumerate(wire_y):
        meas = FancyBboxPatch((10.45, y - 0.3), 0.65, 0.6, boxstyle="round,pad=0.02", 
                              fc="#f3f4f6", ec=c_dark, lw=1.1, zorder=2)
        ax.add_patch(meas)
        ax.text(10.775, y, r"$\langle Z_q \rangle$", ha="center", va="center", fontsize=7.5, fontweight="bold")

    ax.text(10.775, 6.2, "8 Obs.", ha="center", fontsize=8, fontweight="bold", color=c_dark)

    # Output Expectation Mapping Note
    ax.text(7.9, 1.6, r"Per Head: $\mathbf{m}_{ij}^{(s)} = [\langle Z_0 \rangle, \langle Z_1 \rangle, \langle Z_2 \rangle, \langle Z_0 Z_1 \rangle, \langle Z_1 Z_2 \rangle, \langle Z_2 Z_3 \rangle, \langle Z_3 Z_0 \rangle]^T \in \mathbb{R}^7,\quad \phi_x^{(s)} = \langle Z_3 \rangle \in [-1, 1]$", 
            ha="center", fontsize=7.8, fontweight="bold", color=c_dark)
    ax.text(7.9, 1.1, r"Concat 2 Heads: $\mathbf{m}_{ij} = [\mathbf{m}_{ij}^{(1)}, \mathbf{m}_{ij}^{(2)}] \in \mathbb{R}^{14},\qquad \phi_x = \tanh(\mathbf{W}_g [\phi_x^{(1)}, \phi_x^{(2)}]^T) \in (-1, 1)$", 
            ha="center", fontsize=8.2, fontweight="bold", color=c_orange)

    # Arrow 2 -> 3
    arrow2 = FancyArrowPatch((11.45, 4.0), (11.85, 4.0), arrowstyle="-|>", mutation_scale=18, lw=2.0, color=c_dark)
    ax.add_patch(arrow2)

    # Panel 3: Equivariant Update & Readout (x ∈ [11.9, 15.6], y ∈ [0.8, 7.2])
    box3 = FancyBboxPatch((11.9, 0.8), 3.7, 6.4, boxstyle="round,pad=0.15", 
                          fc=c_light_green, ec=c_green, lw=1.8)
    ax.add_patch(box3)
    ax.text(13.75, 6.85, "3. Equivariant Update & Readout", ha="center", va="center", 
            fontsize=12, fontweight="bold", color=c_green)

    # Coordinate Update Box
    c_box = FancyBboxPatch((12.15, 4.9), 3.2, 1.4, boxstyle="round,pad=0.08", 
                           fc="white", ec=c_orange, lw=1.5)
    ax.add_patch(c_box)
    ax.text(13.75, 5.95, r"Coordinate Equivariant Update", ha="center", fontsize=9, fontweight="bold", color=c_orange)
    ax.text(13.75, 5.35, r"$\mathbf{x}_i^{l+1} = \mathbf{x}_i^l + \frac{1}{|\mathcal{N}(i)|} \sum_{j} (\mathbf{x}_i^l - \mathbf{x}_j^l) \phi_x(\mathbf{m}_{ij})$", 
            ha="center", fontsize=8.2, fontweight="bold")

    # Feature Update Box
    f_box = FancyBboxPatch((12.15, 3.1), 3.2, 1.4, boxstyle="round,pad=0.08", 
                           fc="white", ec=c_blue, lw=1.5)
    ax.add_patch(f_box)
    ax.text(13.75, 4.15, r"Node Feature Invariant Update", ha="center", fontsize=9, fontweight="bold", color=c_blue)
    ax.text(13.75, 3.55, r"$\mathbf{h}_i^{l+1} = \mathbf{h}_i^l + \text{MLP}([\mathbf{h}_i^l, \sum_j \mathbf{m}_{ij}])$", 
            ha="center", fontsize=8.5, fontweight="bold")

    # Global Readout Box
    r_box = FancyBboxPatch((12.15, 1.2), 3.2, 1.5, boxstyle="round,pad=0.08", 
                           fc="white", ec=c_green, lw=1.5)
    ax.add_patch(r_box)
    ax.text(13.75, 2.35, r"Invariant Global Readout", ha="center", fontsize=9, fontweight="bold", color=c_green)
    ax.text(13.75, 1.85, r"$\hat{y} = \text{Head}\left(\frac{1}{N} \sum_{i=1}^N \mathbf{h}_i^L\right)$", 
            ha="center", fontsize=9.5, fontweight="bold")
    ax.text(13.75, 1.4, r"Guarantees: $\hat{y}(\mathbf{x}\mathbf{Q}^T + \mathbf{g}) \equiv \hat{y}(\mathbf{x})$", 
            ha="center", fontsize=8, color=c_purple, fontweight="bold")

    plt.tight_layout()
    save_path = PROJECT_ROOT / "results" / "figure_architecture.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), bbox_inches="tight")
    plt.close()
    print(f"Generated architecture diagram: {save_path}")

    # Copy to paper figures
    paper_fig_dir = PROJECT_ROOT / "paper" / "figures"
    paper_fig_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(save_path), str(paper_fig_dir / "figure_architecture.png"))
    print("Copied to paper/figures/figure_architecture.png")


if __name__ == "__main__":
    draw_architecture_diagram()
