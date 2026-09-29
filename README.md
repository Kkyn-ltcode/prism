# ⚛️ Q-PRISM: Equivariant Quantum Parameterized Resonance Graph Neural Networks for 3D Molecular Property Prediction

This repository contains the complete research suite, mathematical proofs, PyTorch + PennyLane implementations, publication figures, and LaTeX manuscript for **Q-PRISM (Equivariant Quantum Parameterized Resonance Graph Neural Networks)** and **PRISM (Periodic Resonance Invariant Spherical Model)**.

---

## 🚀 Quickstart: Interactive Step-by-Step Tutorials

You can run the interactive tutorial using either **Marimo** (reactive Python app) or **Jupyter**:

### Option A: Run with Marimo (Modern Reactive Notebook / Web App)
To edit and run interactively with Marimo:
```bash
marimo edit notebooks/eq_gnn_tutorial_marimo.py
```
or launch as a standalone reactive web app:
```bash
marimo run notebooks/eq_gnn_tutorial_marimo.py
```
You can also open the `.ipynb` directly in Marimo:
```bash
marimo edit notebooks/EQ_GNN_Master_Tutorial.ipynb
```

### Option B: Run with Jupyter Lab / Notebook
```bash
pixi run jupyter lab notebooks/EQ_GNN_Master_Tutorial.ipynb
```
or
```bash
pixi run jupyter notebook notebooks/EQ_GNN_Master_Tutorial.ipynb
```

Both notebooks guide you step-by-step through:
1. **3D Molecular Graphs (QM9 & MD17):** Interactive 3D visualization of atomic coordinates and chemical bonds.
2. **PennyLane 4-Qubit Circuit:** Visualizing multi-axis data re-uploading and Pauli-$Z$ message readouts.
3. **Live $\mathrm{E}(3)$ Equivariance Verification:** Rotating molecules and numerically verifying Theorem 1 in machine precision.
4. **End-to-End Architecture:** Inspecting the full architecture schematic.
5. **Interactive Training Demo:** Training a Q-PRISM model live on the QM9 HOMO-LUMO gap.
6. **Hardware Noise & ZNE Mitigation:** Simulating depolarizing noise and polynomial zero-noise extrapolation.
7. **Benchmark Results Dashboard:** Exploring raw Tables 1–3 and all publication figures.

---

## 📁 Repository Map

| Category / Path | Description |
|---|---|
| 📓 [`notebooks/EQ_GNN_Master_Tutorial.ipynb`](./notebooks/EQ_GNN_Master_Tutorial.ipynb) | **Master Interactive Notebook:** Step-by-step tutorial covering data, circuits, proofs, training, and figures. |
| 📓 [`notebooks/eq_gnn_tutorial_marimo.py`](./notebooks/eq_gnn_tutorial_marimo.py) | **Marimo Reactive App:** Interactive web app tutorial with live parameter sliders and circuit visualizer. |
| 📄 [`RESEARCH_PROPOSAL.md`](./RESEARCH_PROPOSAL.md) | **Master Research Blueprint:** Theoretical motivation, forensic breakdown, and experimental specifications. |
| 📄 [`paper/main.tex`](./paper/main.tex) | **Publication Manuscript:** Complete LaTeX research paper (3 Tables, 7 Figures, 13 References). |
| 🖼️ [`paper/figures/`](./paper/figures/) | **Publication Figures (1–7):** Architecture, scaling laws, symmetry stress tests, ablations, ZNE, N-body, and MD17. |
| 📚 [`paper/literature/`](./paper/literature/) | **Reference Literature:** Baseline ICML papers and equivariant quantum computing publications. |
| 🧠 [`models/q_prism.py`](./models/q_prism.py) | **Core Model:** PyTorch + PennyLane Q-PRISM implementation with exact degree-normalized $\mathrm{E}(3)$ equivariance. |
| 🧠 [`models/prism.py`](./models/prism.py) | **Classical Periodic Twin:** PRISM architecture with SIREN and Fourier heads bypassing spectral bias. |
| 🧠 [`models/egnn_baseline.py`](./models/egnn_baseline.py) | **Classical Baseline:** $\mathrm{E}(n)$-Equivariant Graph Neural Network (Satorras et al., ICML 2021). |
| 📊 [`datasets/`](./datasets/) | **Data Pipelines:** QM9 (`dataset_qm9.py`), MD17 (`dataset_md17.py`), and 3D Charged N-Body (`dataset_nbody.py`). |
| 🧪 [`training/train_qm9.py`](./training/train_qm9.py) | **QM9 Training Engine:** Benchmarks on $\Delta\epsilon$, $\mu$, $U_0$, and $\alpha$. |
| 🪐 [`training/train_nbody.py`](./training/train_nbody.py) | **3D Charged N-Body Benchmark:** 5-particle dynamical trajectory forecasting (Table 2). |
| 📦 [`training/train_md17.py`](./training/train_md17.py) | **MD17 Benchmark:** Molecular dynamics energy surface prediction for Ethanol & Aspirin (Table 3). |
| 🔄 [`benchmarks/benchmark_symmetry_stress_test.py`](./benchmarks/benchmark_symmetry_stress_test.py) | **Continuous Symmetry Test:** $360^\circ$ continuous rotation sweep and $1{,}000$ random $\mathrm{E}(3)$ transforms. |
| 🔬 [`benchmarks/benchmark_ablations.py`](./benchmarks/benchmark_ablations.py) | **Ablation Suite:** 4 axes (Encoding, CNOT topology, Dynamics, Depth). |
| 🛡️ [`benchmarks/benchmark_hardware_noise_zne.py`](./benchmarks/benchmark_hardware_noise_zne.py) | **Hardware Noise & ZNE:** Zero-noise extrapolation under realistic depolarizing noise. |
| 🎨 [`visualization/`](./visualization/) | **Figure Generators:** Plotting scripts for architecture diagram and power-law scaling curves. |
| 📁 [`results/`](./results/) | **Raw JSON Benchmark Data:** Complete empirical results across all random seeds. |

---

## 📊 Summary of Master Results

### 1. QM9 Multi-Target Benchmark (Table 1, Mean $\pm$ Std over 3 Seeds)
- **HOMO-LUMO Gap ($\Delta\epsilon$):** Q-PRISM: $\mathbf{0.7925 \pm 0.0265\text{ eV}}$ vs EGNN: $0.8098 \pm 0.0351\text{ eV}$ (**$-58.4\%$ parameters**).
- **Dipole Moment ($\mu$):** Q-PRISM: $0.8823 \pm 0.0397\text{ Debye}$ vs EGNN: $0.8607 \pm 0.0735\text{ Debye}$.

### 2. 3D Charged N-Body Trajectory Forecasting (Table 2, Mean $\pm$ Std over 3 Seeds)
- **Q-PRISM (Ours, 1,352 params):** Coordinate $\text{MSE} = \mathbf{0.6649 \pm 0.0137}$ ($-64.9\%$ parameters vs EGNN $0.6168$, outperforming Non-Equiv GNN $0.7161$).

### 3. MD17 Molecular Dynamics Energy Surfaces (Table 3, Mean $\pm$ Std over 3 Seeds)
- **Ethanol:** Q-PRISM: $\mathbf{3.380 \pm 0.071\text{ kcal/mol}}$ (matches EGNN $3.380$ with $-58.6\%$ parameters).
- **Aspirin:** Q-PRISM: $4.544 \pm 0.068\text{ kcal/mol}$ (vs EGNN $4.537$, outperforming Non-Equiv GNN $4.556$).

### 4. Symmetry & Invariance Precision
- **Rotational & Reflection Invariance Error:** $\Delta_{\mathrm{inv}} = \mathbf{8.14 \times 10^{-9}\text{ eV}}$ (machine precision vs $1.62\text{ eV}$ baseline failure).
- **Sample Efficiency Scaling Slope:** $\gamma_{\mathrm{quantum}} = \mathbf{0.224}$ vs $\gamma_{\mathrm{classical}} = 0.185$.
- **NISQ Error Mitigation:** Up to **$97.7\%$ error reduction** via Degree-2 Polynomial ZNE at $\lambda=0.005$.
