# Dequantizing Shallow Equivariant Quantum Graph Networks: Periodic Inductive Biases in Molecular Message Passing

This repository contains the complete research codebase, model architectures, training engines, benchmarks, empirical results, and LaTeX manuscript for:

> **Dequantizing Shallow Equivariant Quantum Graph Networks: Periodic Inductive Biases in Molecular Message Passing**  
> Nguyen Trong Luu, Hieu Minh Pham, Anh Quoc Phan, Cho Xuan Do  
> *Posts and Telecommunications Institute of Technology (PTIT), Hanoi, Vietnam*

---

## 1. Overview & Core Findings

Parameterized quantum circuits (PQCs) integrated into $\mathrm{E}(3)$-equivariant graph neural networks (Q-PRISM) exhibit strong sample efficiency on molecular electronic orbital targets (HOMO, LUMO, HOMO-LUMO gap) at low parameter budgets ($\sim 2{,}500$ parameters). This repository investigates the mechanism driving this performance:

1. **The Source of Advantage is Classical:** Systematic ablation across 20 independent seeds demonstrates that multi-qubit entanglement yields negligible advantage on electronic targets ($<2\%$ on HOMO, statistically indistinguishable on LUMO with $p=0.33$ and Gap with $p=0.16$). The active mechanism is that single-qubit data re-uploading rotations naturally compute truncated Fourier series ($\sin, \cos$), providing a compact periodic basis that standard monotonic activations ($\tanh$, SiLU) cannot express at matching parameter counts.
2. **PRISM Architecture:** We introduce **PRISM**, an $\mathrm{E}(3)$-equivariant message-passing GNN that equips edge MLPs with internal periodic sinusoidal activations ($\sin(\omega_0 \mathbf{W}\mathbf{x} + \mathbf{b})$, SIREN). At identical parameter count ($2{,}527$), PRISM matches or exceeds the 4-qubit PQC across all QM9 electronic targets and outperforms PaiNN on four of six targets.
3. **Asymptotic Convergence with Circuit Scaling:** Scaling PQC depth to $L = 16$ layers and width to $n_q = 8$ qubits shows that deep quantum entanglement asymptotically converges to, but never surpasses, PRISM's classical performance ($p = 0.969$), while PRISM delivers up to a $540\times$ training speedup.
4. **Capacity and Dataset Scaling:** The periodic advantage persists on full-scale QM9 ($N = 110{,}000$, 5 seeds, $q_{\mathrm{FDR}} \le 0.055$) and widens monotonically from $+35.7\%$ to $+51.3\%$ as capacity scales to $100{,}127$ parameters.

---

## 2. Model Architecture Hierarchy

All models maintain strict $\mathrm{E}(3)$ equivariance for atomic coordinates and $\mathrm{O}(3)$ invariance for node embeddings:

| Model | Class Name | Edge / Circuit Type | Parameter Count | Key Reference |
| :--- | :--- | :--- | :---: | :--- |
| **Q-PRISM** | `QPRISMModel` (`models/q_prism.py`) | 4-Qubit PQC (Multi-axis data re-uploading, ring CNOT) | $2{,}527$ | Section 3, Figure 2 |
| **PRISM** | `PRISMModel` (`models/prism.py`) | Classical SIREN periodic activation ($\sin(\omega_0 W x + b)$) | $2{,}527$ | Section 3.3, Eq. 8 |
| **Classical Twin** | `ClassicalTwinModelV2` (`models/classical_twin.py`) | Classical MLP with bounded $\tanh$ / SiLU | $2{,}527$ | Section 3.3, Eq. 7 |
| **RFF Twin** | `PRISMModel(head_type='fourier')` | Random Fourier Features ($\sigma = 10.0$) | $2{,}515$ | Section 3.3 |
| **EGNN-PM** | `EGNNModel` (`models/egnn_baseline.py`) | Parameter-matched EGNN (Satorras et al., 2021) | $2{,}497$ | Table 1 |
| **EGNN-Std** | `EGNNModel` (`models/egnn_baseline.py`) | Standard EGNN configuration | $4{,}289$ | Table 1 |
| **PaiNN** | `PaiNNBaseline` (`models/classical_baselines.py`) | Directional equivariant message passing (Schutt et al., 2021) | $2{,}721$ | Table 1 |
| **SchNet** | `SchNetBaseline` (`models/classical_baselines.py`) | Continuous-filter convolution (Schutt et al., 2017) | $2{,}473$ | Table 1 |
| **Non-Equiv GNN** | `NonEquivariantGNN` (`models/classical_baselines.py`) | Standard MPNN control without geometric equivariance | $2{,}537$ | Table 1 |

---

## 3. Installation & Environment

This repository uses [Pixi](https://pixi.sh/) for deterministic environment management:

```bash
# Clone the repository
git clone https://github.com/username/qgnn.git
cd qgnn

# Install environment and dependencies
pixi install

# Activate the shell or run commands directly via pixi
pixi run python --version
```

Dependencies include PyTorch 2.2+, PyTorch Geometric 2.5+, PennyLane 0.35+, RDKit, SciPy, NumPy, and Matplotlib.

---

## 4. Reproducing Paper Experiments

### Table 1: Multi-Target QM9 Benchmark (8 Models, 5 Seeds)
To run the full 8-model benchmark and compute Benjamini-Hochberg False Discovery Rate (FDR) corrections ($m = 48$ hypotheses):

```bash
# Run 8-model benchmark sweep across HOMO, LUMO, Gap, Dipole, Polarizability, and U0
pixi run python training/run_table1_qm9_benchmark.py --targets homo lumo gap mu alpha u0 --seeds 42 43 44 45 46

# Compute statistical corrections (FDR and Bonferroni)
pixi run python training/compute_fdr_corrections.py
```

### Table 2: 20-Seed Iso-Capacity Entanglement Ablation
Isolates the contribution of multi-qubit entanglement across 20 independent seeds:

```bash
pixi run python training/run_table2_entanglement_ablation.py --targets homo lumo gap --epochs 15
```

### Table 3: 2x2 Factorial Ablation (Activation x Initialization)
Evaluates $\{\sin, \mathrm{SiLU}/\tanh\} \times \{\text{SIREN}, \text{Xavier}\}$ across 10 random seeds:

```bash
pixi run python training/run_table3_factorial_ablation.py --targets homo lumo gap --seeds 42 43 44 45 46 47 48 49 50 51
```

### Table S8 & Figure S9: Parameter Capacity Scaling ($2.5\mathrm{k} \to 100\mathrm{k}+$ Parameters)
Evaluates scaling dynamics across 6 capacity tiers (XS, S, M, L, XL, XXL):

```bash
bash run_experiment_e.sh
# or directly:
pixi run python training/run_capacity_scaling_6tiers.py --force
```

### Table S7 & Figure S8: Full-Scale QM9 Validation ($N = 110{,}000$)
Evaluates PRISM vs Classical Twin on the complete QM9 training partition across 5 seeds:

```bash
pixi run python training/run_full_scale_qm9_110k.py --targets homo lumo gap --seeds 42 43 44 45 46 --epochs 20
```

### Tables S9–S10 & Figures S10–S11: Depth and Multi-Qubit Scaling
Scales PQC circuit depth ($L \in \{1, 2, 4, 8, 16\}$) and register width ($n_q \in \{4, 6, 8, 10, 12\}$):

```bash
# Circuit depth scaling
pixi run python training/run_depth_scaling_pqc.py --seeds 42 43 44 45 46

# Multi-qubit scaling and computational efficiency
pixi run python training/run_qubit_scaling_pqc.py --seeds 42 43 44 45 46
```

### Table S5 & Figure S5: Frequency Sensitivity Grid Sweep ($\omega_0 \in [1, 100]$)
```bash
pixi run python training/run_omega_frequency_sweep.py --omega-values 1 5 10 20 30 50 75 100 --seeds 42 43 44 45 46
```

### Tables S11–S12: Negative Controls (N-Body & MD17)
```bash
# 3D Charged N-body dynamical trajectory prediction
pixi run python training/train_nbody.py --model eqgnn --epochs 30

# MD17 molecular dynamics energy surface prediction (Ethanol, Aspirin)
pixi run python training/train_md17.py --molecule ethanol --epochs 30
```

---

## 5. Generating Publication Figures

All publication figures can be reproduced directly from experiment outputs:

```bash
# Main Fig 1 & Supp Figs S18-S19: Scaling curves and crossover points
pixi run python visualization/plot_figure1_scaling.py --benchmark all

# Main Fig 2: Architecture schematic
pixi run python visualization/plot_figure2_architecture.py

# Main Fig 3: Dequantization comparison
pixi run python visualization/plot_figure3_dequantization.py

# Supp Fig S4 & Table S4: Fourier spectrum analysis
pixi run python benchmarks/benchmark_fourier_spectrum_comparison.py

# Supp Fig S7: Large-scale P2.1 evaluation
pixi run python visualization/plot_figure_s7_p21.py

# Supp Fig S9: Extended capacity scaling (6 tiers)
pixi run python visualization/plot_figure_s9_capacity_scaling.py

# Summary multi-panel figures
pixi run python visualization/plot_summary_figures.py
```

---

## 6. Repository Structure

```
prism/
├── README.md               # Overview, quickstart, and reproduction guide
├── LICENSE                 # MIT Open Source License
├── pixi.toml               # Reproducible environment specification
├── models/                 # Equivariant and classical model implementations
│   ├── prism.py            # PRISM: SIREN periodic equivariant GNN
│   ├── q_prism.py          # Q-PRISM: 4-qubit PQC equivariant GNN
│   ├── classical_twin.py   # Iso-parameter tanh/SiLU baseline
│   ├── classical_trig_twin.py # PRISM and Fourier heads
│   ├── classical_baselines.py # SchNet, PaiNN, Non-Equiv GNN
│   └── egnn_baseline.py    # EGNN-PM and EGNN-Std
├── datasets/               # PyG dataset loaders (QM9, MD17, N-body)
├── training/               # Paper-aligned training pipelines and experiment sweeps
├── benchmarks/             # Hardware noise, ZNE, symmetry, and Fourier tests
├── visualization/          # Paper figure generation scripts (Figs 1-3, S7, S9, Summary)
├── results/                # Published JSON benchmark outputs and LaTeX tables
└── paper/                  # Publication manuscript and supplementary material
    ├── main.pdf            # Pre-compiled Main Manuscript (Nature format)
    ├── supplementary.pdf   # Pre-compiled Supplementary Information
    ├── main.tex            # LaTeX source (11-page Research Article)
    ├── supplementary.tex   # Supplementary LaTeX source (23 pages)
    ├── references.bib      # Complete bibliography
    └── figures/            # 23 active publication figures (PNG)
```

---

## 7. Citation

If you find this work or codebase helpful in your research, please cite:

```bibtex
@article{luu2026dequantizing,
  title={Dequantizing Shallow Equivariant Quantum Graph Networks: Periodic Inductive Biases in Molecular Message Passing},
  author={Luu, Nguyen Trong and Pham, Hieu Minh and Phan, Anh Quoc and Do, Cho Xuan},
  journal={arXiv preprint},
  year={2026}
}
```
