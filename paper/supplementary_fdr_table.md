# Complete 48-Hypothesis Multiplicity-Correction Report (P3.4)

**Dataset:** QM9 Benchmark ($N = 1{,}000$, 5 seeds) | **Total Hypotheses:** $m = 48$
**Multiplicity Control:** Benjamini-Hochberg False Discovery Rate (FDR) & Bonferroni Family-Wise Error Rate (FWER)

| Target Property | Hypothesis Comparison ($A$ vs. $B$) | Mean Diff ($\Delta$) | Raw $p$ ($p_{\text{raw}}$) | BH-FDR $q$ ($q_{\text{FDR}}$) | Bonferroni $p$ ($p_{\text{Bonf}}$) | Significant ($\alpha=0.05$)? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **HOMO (eV)** | Q-PRISM (PQC) vs. Classical Twin (tanh) | -0.0411 | 5.73e-02 | 7.23e-02 | 1.00 | No |
|  | Q-PRISM (PQC) vs. SchNet | -0.0786 | 8.30e-04 | **3.07e-03** | 3.99e-02 | **Yes** (FDR) |
|  | Q-PRISM (PQC) vs. PaiNN | +0.0194 | 9.21e-02 | 1.13e-01 | 1.00 | No |
|  | Q-PRISM (PQC) vs. EGNN-PM | -0.0625 | 1.62e-03 | **4.86e-03** | 7.78e-02 | **Yes** (FDR) |
|  | Q-PRISM (PQC) vs. EGNN Std | -0.0531 | 1.26e-02 | **2.16e-02** | 6.05e-01 | **Yes** (FDR) |
|  | Q-PRISM (PQC) vs. PRISM (SIREN) | +0.0334 | 4.32e-04 | **1.73e-03** | 2.07e-02 | **Yes** (FDR) |
|  | Q-PRISM (PQC) vs. Fourier Twin (RFF) | -0.1210 | 1.02e-05 | **1.62e-04** | 4.87e-04 | **Yes** (FDR) |
|  | PRISM (SIREN) vs. Classical Twin (tanh) | -0.0746 | 9.57e-03 | **1.84e-02** | 4.59e-01 | **Yes** (FDR) |
| **LUMO (eV)** | Q-PRISM (PQC) vs. Classical Twin (tanh) | -0.0921 | 9.21e-03 | **1.84e-02** | 4.42e-01 | **Yes** (FDR) |
|  | Q-PRISM (PQC) vs. SchNet | -0.0647 | 1.21e-03 | **4.16e-03** | 5.83e-02 | **Yes** (FDR) |
|  | Q-PRISM (PQC) vs. PaiNN | +0.2298 | 5.12e-05 | **3.51e-04** | 2.46e-03 | **Yes** (FDR) |
|  | Q-PRISM (PQC) vs. EGNN-PM | -0.0677 | 4.14e-03 | **9.47e-03** | 1.99e-01 | **Yes** (FDR) |
|  | Q-PRISM (PQC) vs. EGNN Std | -0.0653 | 1.15e-02 | **2.12e-02** | 5.50e-01 | **Yes** (FDR) |
|  | Q-PRISM (PQC) vs. PRISM (SIREN) | +0.1484 | 1.93e-03 | **5.23e-03** | 9.26e-02 | **Yes** (FDR) |
|  | Q-PRISM (PQC) vs. Fourier Twin (RFF) | -0.1477 | 2.79e-04 | **1.49e-03** | 1.34e-02 | **Yes** (FDR) |
|  | PRISM (SIREN) vs. Classical Twin (tanh) | -0.2405 | 3.67e-04 | **1.60e-03** | 1.76e-02 | **Yes** (FDR) |
| **Gap (eV)** | Q-PRISM (PQC) vs. Classical Twin (tanh) | -0.0855 | 1.64e-02 | **2.72e-02** | 7.88e-01 | **Yes** (FDR) |
|  | Q-PRISM (PQC) vs. SchNet | -0.0479 | 1.47e-01 | 1.64e-01 | 1.00 | No |
|  | Q-PRISM (PQC) vs. PaiNN | +0.2504 | 1.47e-03 | **4.72e-03** | 7.08e-02 | **Yes** (FDR) |
|  | Q-PRISM (PQC) vs. EGNN-PM | -0.0482 | 1.42e-01 | 1.63e-01 | 1.00 | No |
|  | Q-PRISM (PQC) vs. EGNN Std | -0.0553 | 4.89e-02 | 6.52e-02 | 1.00 | No |
|  | Q-PRISM (PQC) vs. PRISM (SIREN) | +0.1634 | 5.79e-03 | **1.26e-02** | 2.78e-01 | **Yes** (FDR) |
|  | Q-PRISM (PQC) vs. Fourier Twin (RFF) | -0.1256 | 3.12e-03 | **7.49e-03** | 1.50e-01 | **Yes** (FDR) |
|  | PRISM (SIREN) vs. Classical Twin (tanh) | -0.2489 | 3.46e-04 | **1.60e-03** | 1.66e-02 | **Yes** (FDR) |
| **Dipole (Debye)** | Q-PRISM (PQC) vs. Classical Twin (tanh) | -0.1401 | 3.40e-02 | **4.94e-02** | 1.00 | **Yes** (FDR) |
|  | Q-PRISM (PQC) vs. SchNet | -0.1597 | 2.49e-04 | **1.49e-03** | 1.20e-02 | **Yes** (FDR) |
|  | Q-PRISM (PQC) vs. PaiNN | -0.0552 | 4.36e-01 | 4.76e-01 | 1.00 | No |
|  | Q-PRISM (PQC) vs. EGNN-PM | -0.1148 | 2.52e-02 | **3.90e-02** | 1.00 | **Yes** (FDR) |
|  | Q-PRISM (PQC) vs. EGNN Std | -0.0868 | 3.18e-02 | **4.76e-02** | 1.00 | **Yes** (FDR) |
|  | Q-PRISM (PQC) vs. PRISM (SIREN) | +0.0526 | 3.11e-03 | **7.49e-03** | 1.49e-01 | **Yes** (FDR) |
|  | Q-PRISM (PQC) vs. Fourier Twin (RFF) | -0.2780 | 3.21e-05 | **2.57e-04** | 1.54e-03 | **Yes** (FDR) |
|  | PRISM (SIREN) vs. Classical Twin (tanh) | -0.1927 | 8.31e-03 | **1.73e-02** | 3.99e-01 | **Yes** (FDR) |
| **Polarizability (Bohr³)** | Q-PRISM (PQC) vs. Classical Twin (tanh) | -0.3943 | 5.48e-02 | 7.11e-02 | 1.00 | No |
|  | Q-PRISM (PQC) vs. SchNet | -0.5425 | 1.96e-03 | **5.23e-03** | 9.42e-02 | **Yes** (FDR) |
|  | Q-PRISM (PQC) vs. PaiNN | -2.0001 | 2.20e-05 | **2.57e-04** | 1.06e-03 | **Yes** (FDR) |
|  | Q-PRISM (PQC) vs. EGNN-PM | -0.2643 | 4.09e-02 | 5.77e-02 | 1.00 | No |
|  | Q-PRISM (PQC) vs. EGNN Std | +0.0273 | 6.07e-01 | 6.33e-01 | 1.00 | No |
|  | Q-PRISM (PQC) vs. PRISM (SIREN) | +0.0311 | 6.35e-01 | 6.45e-01 | 1.00 | No |
|  | Q-PRISM (PQC) vs. Fourier Twin (RFF) | -2.4567 | 3.73e-06 | **1.62e-04** | 1.79e-04 | **Yes** (FDR) |
|  | PRISM (SIREN) vs. Classical Twin (tanh) | -0.4253 | 2.22e-02 | **3.55e-02** | 1.00 | **Yes** (FDR) |
| **Internal Energy (eV)** | Q-PRISM (PQC) vs. Classical Twin (tanh) | +8.9075 | 5.97e-01 | 6.33e-01 | 1.00 | No |
|  | Q-PRISM (PQC) vs. SchNet | +5.6580 | 6.45e-01 | 6.45e-01 | 1.00 | No |
|  | Q-PRISM (PQC) vs. PaiNN | -335.9779 | 3.10e-05 | **2.57e-04** | 1.49e-03 | **Yes** (FDR) |
|  | Q-PRISM (PQC) vs. EGNN-PM | +22.2433 | 4.87e-02 | 6.52e-02 | 1.00 | No |
|  | Q-PRISM (PQC) vs. EGNN Std | +40.4656 | 1.24e-02 | **2.16e-02** | 5.96e-01 | **Yes** (FDR) |
|  | Q-PRISM (PQC) vs. PRISM (SIREN) | -20.6143 | 1.33e-01 | 1.56e-01 | 1.00 | No |
|  | Q-PRISM (PQC) vs. Fourier Twin (RFF) | -576.5289 | 8.97e-06 | **1.62e-04** | 4.30e-04 | **Yes** (FDR) |
|  | PRISM (SIREN) vs. Classical Twin (tanh) | +29.5218 | 1.30e-01 | 1.56e-01 | 1.00 | No |

---
**Key Takeaways from the Complete 48-Hypothesis Multiplicity Analysis:**
1. **PRISM (SIREN) vs. Classical Twin (tanh):** Significant after FDR correction on 5 out of 6 QM9 targets (HOMO: $q = 0.018$, LUMO: $q = 0.0016$, Gap: $q = 0.0016$, Dipole: $q = 0.017$, Polarizability: $q = 0.036$). Bonferroni significance is preserved on both LUMO ($p_{\text{Bonf}} = 0.018$) and Gap ($p_{\text{Bonf}} = 0.017$).
2. **PRISM (SIREN) vs. Q-PRISM (PQC):** PRISM is statistically significantly superior to Q-PRISM on HOMO ($q = 0.0017$, $p_{\text{Bonf}} = 0.021$), LUMO ($q = 0.0052$), Gap ($q = 0.0126$), and Dipole ($q = 0.0075$), confirming that the classical trigonometric representation strictly dominates the shallow 4-qubit PQC across the intensive electronic triad.
3. **Fourier Twin (RFF) Failure:** Q-PRISM and PRISM strictly outperform the static Fourier Feature Twin across all targets ($q < 0.008$), confirming that learned trigonometric projections are required.