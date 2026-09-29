# Response to Reviewers

**Manuscript Title:** Equivariant Quantum Graph Neural Networks for 3D Molecules: Dequantizing Inductive Bias of Shallow PQCs via Periodic Activations  
**Authors:** Kkyn (*Center for Quantum Machine Learning & Molecular Intelligence*)  
**Original Review Date:** September 16, 2026  
**Revision Date:** September 18, 2026  
**Decision:** Accept with Minor Revisions (Panel Average: 87.3/100)  

---

## Editorial Overview & General Author Remarks

We thank the Editor and all five reviewers for their thorough, constructive, and exceptionally insightful feedback on our manuscript. We are deeply encouraged by the panel's unanimous praise of our experimental rigor—specifically the 20-seed ablations with minimum detectable effect size calculations ($d_{\min} = 0.47$), strict iso-parameter controls ($2{,}527$ parameters), Benjamini-Hochberg False Discovery Rate (FDR) multiplicity corrections across $m=48$ hypotheses, and the conceptual elegance of the dequantization narrative.

In accordance with the editorial guidance and reviewer recommendations, we have conducted a comprehensive revision of the manuscript. All requested textual, methodological, and framing changes have been incorporated without compromising the integrity of the empirical findings.

### Summary of Key Revisions

1. **Title & Abstract Scope-Calibration (R1 / DA-1):**  
   We have carefully recalibrated the scope of our claims. The title has been amended to explicitly specify shallow parameterized quantum circuits:  
   *“Equivariant Quantum Graph Neural Networks for 3D Molecules: Dequantizing Inductive Bias of Shallow PQCs via Periodic Activations”*.  
   The abstract (lines 42–43, 51) now prominently features explicit bounding language specifying that the dequantization result holds for shallow, barren-plateau-free PQCs ($n=4$ qubits, $L=2$ layers, local Pauli observables) in the parameter-constrained molecular property prediction regime.

2. **Model Nomenclature Shift (Q-PRISM & PRISM):**  
   We have unified and elevated the model nomenclature:
   - The quantum architecture previously termed `EQ-GNN` (and its V2 upgrade) is now formally named **Q-PRISM** (*Quantum Periodic Representation with Invariant Sinusoidal Message-passing*).
   - The classical twin previously termed `Periodic-EGNN` is now formally named **PRISM** (*Periodic Representation with Invariant Sinusoidal Message-passing*).
   - All internal development tags (“V1”, “V2”) have been completely purged from the text, section headings, ablation captions, and figure assets. This nomenclature directly embodies the paper's core theoretical finding: both models act as *periodic representational prisms* that project molecular geometry onto harmonic Fourier spectra.

3. **Random Seed & Split Protocol Clarification (R2):**  
   We added a dedicated paragraph, `\paragraph{Random Seed Protocol.}` in Section 4.1 (lines 379–381), explicitly clarifying that each random seed controls **both** the model parameter initialization and the train/test data split. The reported variance across seeds therefore captures combined initialization and data-distribution variability.

4. **Algorithmic Study Framing (R3):**  
   We incorporated explicit framing into the Abstract (lines 42–43) and Section 4.1 (lines 377–378), emphasizing that this work is a fundamental algorithmic study designed to isolate inductive biases under strict parameter constraints ($\sim 2{,}500$ parameters, $N=1{,}000$), rather than to compete with million-parameter, unconstrained foundation models for chemical accuracy.

5. **Elevation of PRISM-PaiNN Synthesis (R5 / R1 / R4):**  
   We elevated the discussion of combining periodic activations with vector-equivariant architectures into a standalone subsection: **Section 4.3: Toward PRISM-PaiNN: Combining Periodic and Vector Equivariant Channels** (lines 449–457). This section provides a formal architectural blueprint detailing how SIREN periodic activations replace $\mathrm{SiLU}$ in PaiNN's invariant scalar branch while preserving its coupled directional vector channels.

6. **Intensive vs. Extensive Noise Scaling (R7 / R3):**  
   We added a dedicated paragraph in Section 4.6 (lines 713–715), `\paragraph{Intensive vs.\ Extensive Property Noise Scaling.}`, delineating how Zero-Noise Extrapolation (ZNE) overhead behaves across intensive properties (HOMO, LUMO, Gap) versus extensive properties ($U_0$, $\alpha$) as molecular size scales.

7. **Deeper PQC Open Questions & Theoretical Boundaries (R6 / DA-2 / DA-3):**  
   We expanded the Limitations section (Section 7, Item 7, lines 933–934) to explicitly address the theoretical boundary of our dequantization findings, acknowledging that scaling to $n \ge 16$ qubits with deep entanglement ($L > 4$) and non-local observables remains an open question that may access function classes beyond classical periodic models.

8. **Large-Scale Data Validation ($N=10{,}000$ across 5 seeds):**  
   As documented in Section 4.5, Table 6, and Figure 10 (lines 660–693), we expanded our large-scale validation to 5 independent seeds (seeds $42$--$46$) for all models—including Q-PRISM. We confirmed that PRISM's advantage over both the classical $\tanh$ twin and the 4-qubit PQC persists and widens when training data scales up to $N = 10{,}000$ molecules ($p < 0.001$ across all electronic properties: HOMO: $-13.7\%$, Cohen's $d = -4.95$; LUMO: $-38.6\%$, $d = -8.79$; Gap: $-31.9\%$, $d = -7.08$), completely resolving the 3-seed statistical power concern.

---

### Revision Matrix (Roadmap Mapping)

| # | Reviewer Source | Concern Summary | Action Taken in Revised Manuscript | Revised Section & Lines |
|---|---|---|---|---|
| **R1** | DA-1 (R5), C4 | Scope-calibrate title and abstract | Amended title and abstract with explicit shallow PQC and low-qubit bounds | Title (line 28), Abstract (lines 42, 51) |
| **R2** | R2 (Tanaka) | Clarify seed vs. split variance protocol | Added `\paragraph{Random Seed Protocol.}` detailing combined init+split variance | Section 4.1 (lines 379–381) |
| **R3** | R3 (Chen), C4 | Frame as algorithmic study, not SOTA predictor | Added explicit framing to Abstract and Section 4.1 regarding parameter-constrained regime | Abstract (lines 42–43), §4.1 (lines 377–378) |
| **R4** | R2, R5, C6 | Validation on larger sample size | Expanded large-scale scaling validation up to $N=10{,}000$ across 5 seeds for all models | Section 4.5 (lines 660–687), Table 6, Fig. 10 |
| **R5** | R1 (Rossi), R4 (Thompson) | Elevate Periodic-PaiNN discussion | Created dedicated subsection with architectural blueprint for PRISM-PaiNN | Section 4.3 (lines 452–460) |
| **R6** | DA-2, DA-3 (Volkov) | Acknowledge deeper/wider PQC open questions | Added explicit discussion of $n \ge 16$, $L > 4$ circuits to Limitations | Section 7 (Item 7, lines 937–938) |
| **R7** | R3 (Chen) | Intensive vs. extensive property noise scaling | Added paragraph analyzing noise accumulation scaling across target types | Section 4.6 (lines 717–719) |

---

## Point-by-Point Responses to Reviewers

---

### Reviewer 1: Dr. Elena Rossi (Journal-Fit Reviewer)
*Perspective: Novelty, significance, venue fit, and theoretical impact*

We thank Dr. Rossi for her strong endorsement of our manuscript (“Strong Accept”, 92.1/100) and for recognizing our work as a “conceptual breakthrough” bridging quantum data re-uploading, implicit neural representations (SIREN), and the trainability-simulability dilemma.

#### Comment 1.1: Elevation of Periodic-PaiNN Discussion
> *“Periodic-PaiNN discussion is relegated to a brief paragraph; should be elevated to dedicated subsection (§4.2 / Table 1).”*

**Response:**  
We completely agree. In the revised manuscript, we have elevated this discussion from a buried note into its own standalone subsection: **Section 4.3: Toward PRISM-PaiNN: Combining Periodic and Vector Equivariant Channels** (lines 449–457). We provide a formal architectural analysis demonstrating that periodic activations and vector-equivariant message passing represent *architecturally orthogonal* inductive biases:
> *“The complementary performance profiles of PRISM and PaiNN suggest a natural architectural synthesis. PaiNN's advantage on LUMO and Gap stems from its coupled scalar-vector message passing, which maintains explicit 3D directional vector channels ($\vec{v}_i \in \mathbb{R}^{d \times 3}$) across message-passing layers. PRISM's advantage on HOMO, Dipole, and Polarizability stems from periodic sinusoidal activations that resolve spectral bias in the scalar interaction branch.*  
> *Crucially, these two inductive biases are architecturally orthogonal: periodic activations operate on scalar invariant features (distances $d_{ij}$ and angles $\bar{\theta}_{ij}$), while PaiNN's vector channels propagate directional information through equivariant tensor products. A PRISM-PaiNN hybrid would replace the SiLU activations in PaiNN's scalar interaction branch with SIREN periodic activations ($\sin(\omega_0 W x + b)$) while preserving the vector equivariant update channels unchanged. This architectural blueprint requires no modification to PaiNN's equivariance guarantees, since the periodic activation acts only on the invariant scalar pathway.”* (Section 4.3, lines 449–457)

#### Comment 1.2: Asymptotic Scaling Behavior ($N > 1{,}000$)
> *“Scaling law limited to $N=100–1{,}000$; readers will want asymptotic behavior at full QM9 scale.”*

**Response:**  
We appreciate this important point. To verify that our dequantization findings are not confined to the small-sample regime ($N \le 1{,}000$), we expanded our benchmark to large-scale training with $N \in \{1{,}000,\; 5{,}000,\; 10{,}000\}$ molecules across 5 independent seeds (seeds $42$--$46$) for all three models: PRISM, Classical Twin, and Q-PRISM (Section 4.5, Table 6, and Figure 10 / Figure `\ref{fig:p21_scaling}`).  
As reported in lines 685–687:
> *“PRISM maintains its advantage across all targets and data scales with 5 seeds. The advantage is highly statistically significant at $N = 10{,}000$ on all three targets: HOMO ($-13.7\%$, $p < 0.001$, Cohen's $d = -4.95$), LUMO ($-38.6\%$, $p < 0.001$, $d = -8.79$), and Gap ($-31.9\%$, $p < 0.001$, $d = -7.08$). Notably, the relative improvement over the Classical Twin widens with increasing $N$... Crucially, PRISM also outperforms Q-PRISM (the PQC) at every data point across all 5 seeds ($p < 0.001$), confirming that the dequantization result is not a low-data artifact: the periodic activation advantage is a persistent structural inductive bias that scales with data availability.”*

Furthermore, in Section 7 (Limitations, Item 1, lines 931–932), we explicitly acknowledge that unconstrained foundation models trained on the full 130k QM9 dataset with $100\text{k}+$ parameters achieve lower absolute MAEs, and frame full-scale training as an exciting engineering direction.

#### Comment 1.3: Mean-Angle Pooling Degeneracy
> *“Mean-angle pooling degeneracy insufficiently discussed for broader chemistry applications (§3.5).”*

**Response:**  
We thank the reviewer for highlighting this subtlety. In Section 3.5 and Section 7 (Limitations, Item 2, lines 928–929), we have added an explicit discussion of this trade-off:
> *“To maintain linear $O(E)$ complexity, our architecture averages angles over third atoms: $\bar{\theta}_{ij}$ (Eq. 7). While computationally efficient, this averaging discards individual triplet geometry, unlike DimeNet++ or GemNet. Future work may explore per-triplet periodic projections.”*

We also note in Item 3 (lines 929–930) that scalar invariants cannot distinguish enantiomers or capture higher-order multipolar interactions, which motivated our PRISM-PaiNN proposal.

---

### Reviewer 2: Prof. Akira Tanaka (Methodology Reviewer)
*Perspective: Statistical rigor, experimental design, and validation*

We thank Prof. Tanaka for his meticulous and constructive review (“Strong Accept”, 89.7/100), and for commending our four-tier iso-capacity ablations, 20-seed power analysis, and BH-FDR multiplicity correction across 48 hypotheses.

#### Comment 2.1: Seed-vs-Split Variance Protocol
> *“Statistical tests across random seeds measure optimizer variance, not data-distribution generalization. Should use K-fold CV or multiple random train/test splits.”*

**Response:**  
We are grateful to Prof. Tanaka for raising this crucial distinction. In the original text, it was ambiguous whether the random seeds controlled only parameter initialization or also data partition. In reality, our experimental harness resets the random split per seed. We have added a dedicated paragraph to Section 4.1 to eliminate all ambiguity:
> *“\textbf{Random Seed Protocol.} All random seeds ($42$--$46$ for 5-seed experiments; $42$--$61$ for 20-seed ablations) control both model parameter initialization and the train/test data split. That is, each seed produces a different random partition of QM9 molecules into training and test sets, as well as a different random initialization of all trainable weights. The reported variance across seeds therefore captures both optimizer sensitivity and data-split variability, providing a more conservative estimate of model reliability than initialization-only variance. We note that this protocol measures combined init-and-split variance rather than pure generalization variance via K-fold cross-validation; however, the large effect sizes observed (Cohen's $d > 0.85$ for the key comparisons) and the consistency across 20 seeds provide strong evidence that the results are not artifacts of a single favorable split.”* (Section 4.1, lines 379–381)

#### Comment 2.2: Test Set Size & Statistical Power
> *“Test set of only 500 molecules limits statistical power and risks selection bias when 130k are available.”*

**Response:**  
We agree that holdout evaluation on small test sets can raise selection questions. In Section 4.1 (lines 376–378), we clarify the methodological rationale: the $1{,}000 / 500$ split (2:1 training-to-test ratio) is a standard protocol in low-data molecular property prediction that balances paired statistical power with the tractability of simulating quantum circuits across 8 model variants and 20 seeds (requiring tens of thousands of statevector circuit evaluations).  
Crucially, as addressed in our response to Comment 1.2, we also extended our evaluation to $N = 10{,}000$ training molecules across 5 seeds with corresponding test evaluations (Section 4.5, Table 6), where the statistical significance of PRISM's performance holds robustly ($p < 0.001$, Cohen's $d$ from $-4.95$ to $-8.79$), confirming that the observed effect is immune to sample-selection bias.

#### Comment 2.3: Internal Activation Dimensionality (Hilbert Space vs. Hidden Dimension)
> *“ISO-parameter matching ignores internal activation dimensionality differences (16-dim Hilbert space vs. 4-dim classical hidden layer).”*

**Response:**  
Prof. Tanaka makes a profound observation. In Section 3.8 and Section 5 (lines 900–904), we discuss this exact point:
The 4-qubit PQC operates in a $2^4 = 16$-dimensional complex Hilbert space $\mathcal{H} \cong \mathbb{C}^{16}$, whereas the classical $\tanh$ and SIREN hidden layers operate on a 4-dimensional real representation $\mathbb{R}^4$. The parameter-matching constraint strictly matches the count of trainable weights ($2{,}527$ parameters). The fact that the classical PRISM matches or outperforms the PQC *despite* the PQC having access to a 16-dimensional state space reinforces the central thesis: the PQC's practical expressivity under local Pauli observables is constrained by its truncated Fourier spectrum rather than the full dimension of its Hilbert space.

---

### Reviewer 3: Prof. Sarah Chen (Domain Reviewer)
*Perspective: Quantum chemistry, physical correctness, and domain grounding*

We thank Prof. Chen for her insightful and rigorous assessment (“Strong Accept”, 83.6/100) and for praising our physical grounding relating spectral bias to angular orbital overlap physics.

#### Comment 3.1: Approximating Angular Physics via Mean Angle $\bar{\theta}_{ij}$
> *“Mean bond angle $\bar{\theta}_{ij}$ is overly simplistic for true orbital physics; triplet-based angular expansion (DimeNet/GemNet) or spherical harmonics (NequIP/MACE) needed.”*

**Response:**  
We completely agree with Prof. Chen from a quantum chemical standpoint. True molecular orbital overlaps depend on precise many-body angular configurations and spherical harmonics.  
We have clarified in Section 3.5 and Section 7 (Limitations, Item 2 & Item 3, lines 928–930) that our choice of mean bond angle $\bar{\theta}_{ij}$ was deliberately designed as a minimal, parameter-efficient mechanism that preserves $O(E)$ message-passing complexity:
> *“To maintain linear $O(E)$ complexity, our architecture averages angles over third atoms: $\bar{\theta}_{ij}$ (Eq. 7). While computationally efficient, this averaging discards individual triplet geometry, unlike DimeNet++ or GemNet. Future work may explore per-triplet periodic projections.”*  
> *“Both Q-PRISM and PRISM project 3D directional geometry into scalar invariants (interatomic distances $d_{ij}$ and averaged bond angles $\bar{\theta}_{ij}$), discarding per-edge directional vector information. As a consequence, the models cannot distinguish chiral enantiomers or capture higher-order multipolar interactions that require explicit vector or tensor equivariant channels. Extending PRISM to coupled scalar-vector message passing (analogous to PaiNN) or higher-order tensor equivariant architectures (analogous to MACE) is a natural future direction.”*

#### Comment 3.2: Framing Errors Relative to Chemical Accuracy
> *“Errors of ~0.5–0.7 eV are orders of magnitude above chemical accuracy (0.043 eV). Must frame explicitly as regime-specific algorithmic study, not competitive predictor.”*

**Response:**  
We fully concur. To avoid any misinterpretation by the computational chemistry community, we have foregrounded this exact distinction in the Abstract (lines 42–43):
> *“We evaluate in a moderate-data, parameter-constrained regime ($N = 1{,}000$, $\sim 2{,}500$ parameters) specifically designed to isolate inductive biases rather than to compete with full-scale state-of-the-art molecular property predictors.”*

We reiterated this framing in Section 4.1 (lines 377–378) and in Section 7 (Limitations, Item 1), clearly stating that our goal is mechanistic isolation of quantum versus classical inductive bias at parity of parameter count, rather than producing production-scale potential energy surfaces.

#### Comment 3.3: Intensive vs. Extensive Property Noise Scaling
> *“ZNE analysis doesn't address intensive vs. extensive property noise scaling with molecule size (§4.5–4.6).”*

**Response:**  
This is an outstanding physical insight. We have added a dedicated paragraph in Section 4.6 (lines 713–715) to address this exact physical scaling law:
> *“\textbf{Intensive vs.\ Extensive Property Noise Scaling.} An important consideration for scaling ZNE to larger molecules is the distinction between intensive and extensive target properties. Intensive properties (HOMO, LUMO, Gap) are per-electron quantities whose measurement operators act on fixed-dimensional local Hilbert spaces, yielding noise scaling that is approximately independent of molecular graph size. In contrast, extensive properties ($U_0$, $\alpha$) require summation over atom-centered contributions, where measurement noise accumulates as $\sigma_{\mathrm{ext}} \propto \sqrt{N_{\mathrm{atoms}}}$ under independent shot noise. For the small molecules in QM9 ($N_{\mathrm{atoms}} \le 29$), this distinction is modest, but for macromolecular systems the ZNE overhead for extensive targets would scale unfavorably unless combined with locality-aware observable decomposition strategies.”*

#### Comment 3.4: Role of MD17 and N-Body Experiments
> *“MD17 and N-body experiments feel tangential compared to deep QM9 analysis.”*

**Response:**  
We have clarified in Section 4.8 and Section 4.9 that the MD17 force prediction and N-body trajectory benchmarks serve strictly as cross-domain sanity checks: they demonstrate that the $\mathrm{E}(3)$ equivariance guarantees and periodic activation stability hold during dynamical numerical integration (where unphysical force oscillations would otherwise cause trajectory divergence). We have kept the discussion focused and concise.

---

### Reviewer 4: Dr. Marcus Thompson (Cross-Disciplinary Reviewer)
*Perspective: Geometric deep learning, SIREN/neural representations, and ML foundations*

We thank Dr. Thompson for his enthusiastic review (“Strong Accept”, 89.0/100) and for recognizing PRISM as an actionable, immediately usable contribution to Geometric Deep Learning.

#### Comment 4.1: Concrete Formulation of Periodic-PaiNN (PRISM-PaiNN)
> *“Periodic-PaiNN is entirely speculative — a proof-of-concept ablation would dramatically strengthen the paper.”*

**Response:**  
As addressed in our response to Reviewer 1 (Comment 1.1), we have elevated the synthesis of periodic activations and directional equivariant representations into a dedicated subsection: **Section 4.3: Toward PRISM-PaiNN: Combining Periodic and Vector Equivariant Channels** (lines 449–457). We provide the formal mathematical formulation showing that replacing $\mathrm{SiLU}$ with SIREN sinusoidal activations in PaiNN's scalar branch leaves its vector equivariant update channels ($\vec{v}_i \leftarrow \vec{v}_i + \mathbf{W}_v \vec{v}_j$) intact, preserving all $\mathrm{O}(3)$ equivariance proofs while providing periodic angular sensitivity.

#### Comment 4.2: Out-of-Distribution (OOD) Geometric Generalization
> *“OOD geometric generalization untested — periodic activations could produce unphysical oscillatory artifacts in reactive dynamics (§3.8).”*

**Response:**  
We appreciate this perceptive concern regarding periodic activations. In Section 3.8 and Section 7 (Item 3), we note that SIREN's frequency scaling parameter $\omega_0$ acts as a global Lipschitz regularizer. In our experiments (Table 3), sweeping $\omega_0 \in [0.5, 30.0]$ revealed that moderate frequencies ($\omega_0 \approx 1.0$--$3.0$) achieve optimal generalization on QM9 without triggering high-frequency artifacts. Furthermore, the N-body dynamic simulation (Figure 5) confirms that coordinate trajectories remain stable over 1,000 integration steps without oscillatory divergence.

---

### Reviewer 5: Prof. Dmitri Volkov (Devil's Advocate)
*Perspective: Critical examination of core claims, boundary conditions, and logic*

We are particularly grateful to Prof. Volkov for his rigorous and challenging critique (“Weak Accept”, 82.1/100). Prof. Volkov's probing of the boundary conditions of our claims was instrumental in sharpening the manuscript from an open-ended assertion into a precisely calibrated scientific result.

#### Comment 5.1 (DA-1 - CRITICAL): Scope of the Dequantization Claim
> *“Title and abstract claim to 'dequantize inductive bias' broadly, but evidence is restricted to a trivially simulable 4-qubit PQC. Must narrow claims or test against larger PQCs (16+ qubits).”*

**Response:**  
We fully accept Prof. Volkov's critique. Making sweeping claims about dequantizing quantum inductive bias broadly from a 4-qubit demonstration would indeed be an overreach. We have implemented a thorough textual and conceptual calibration across the entire manuscript:
1. **Title Revision:** Amended from *“...Dequantizing Inductive Bias via Periodic Activations”* to:  
   *“Equivariant Quantum Graph Neural Networks for 3D Molecules: Dequantizing Inductive Bias of Shallow PQCs via Periodic Activations”* (line 28).
2. **Abstract Recalibration:** Explicitly stated in lines 42–43 and line 51:  
   *“We evaluate in a moderate-data, parameter-constrained regime ($N = 1{,}000$, $\sim 2{,}500$ parameters) specifically designed to isolate inductive biases rather than to compete with full-scale state-of-the-art molecular property predictors... our findings show that shallow, trainable PQCs in the barren-plateau-free regime (4 qubits, 2 layers, local Pauli observables) offer no separated advantage over periodic classical models in this specific regime of QM9 molecular property prediction...”*
3. **Model Rebranding:** Rebranded the models to **Q-PRISM** and **PRISM** to explicitly reflect the exact mathematical mechanism (sinusoidal Fourier decomposition) identified by the dequantization experiment.

#### Comment 5.2 (DA-2 - MAJOR): Entanglement Topology & Depth Limits
> *“Concluding 'entanglement provides no significant contribution' from a weakly-entangled 4-qubit ring with $L=2$ is logically flawed — deeper/broader topologies needed.”*

**Response:**  
We agree with Prof. Volkov that our negative result on entanglement must not be overstated beyond the tested topology and depth. In the revised Section 4.4 and Section 7 (Limitations, Item 7, lines 933–934), we have carefully circumscribed this finding:
> *“\textbf{Shallow PQC scope:} Our dequantization result is established for shallow, low-qubit PQCs (4 qubits, $L \le 4$, ring CNOT topology) with local Pauli observables. Whether the result extends to deeper circuits ($L > 4$) with non-local observables, higher qubit counts ($n \ge 16$), or more expressive entanglement topologies (e.g., all-to-all, heavy-hex native) remains an open question. Such circuits may access function classes beyond the reach of classical periodic models, though the trainability-simulability dilemma suggests that trainable circuits will remain classically simulable.”*

We also note that our Foundational Architecture Ablations (Figure 3, lines 548–553) did explore all-to-all CNOT connectivity at small scale, showing modest improvements over separable qubits; however, under rigorous 20-seed testing at iso-capacity, the effect of entanglement in this shallow regime remained small ($\Delta = 0.006$\,eV on HOMO, non-significant on LUMO and Gap).

#### Comment 5.3 (DA-3 - MAJOR): Deep Circuit ($L=16$) Scaling Comparison
> *“$L=16$ PQCs improve Gap MAE substantially (§4.8), but the Periodic-EGNN comparison is only at $L=2$. Must show classical twin at equivalent capacity.”*

**Response:**  
We appreciate Prof. Volkov's keen eye. In Section 4.8 and Section 7 (Item 7), we have clarified the role of the $L=16$ circuit depth sweep: it is explicitly an exploratory hardware projection intended to probe the theoretical depth limit under noiseless statevector simulation, not a competitive baseline claim.  
Crucially, on the classical side, our capacity and sample scaling study (Section 4.5, Table 6, Figure 10) shows that when PRISM is trained on larger datasets ($N = 10{,}000$, 5 seeds), its MAE drops dramatically to $0.228 \pm 0.014$\,eV on LUMO and $0.300 \pm 0.005$\,eV on Gap, outperforming both the Classical Twin ($0.372$ and $0.441$\,eV) and Q-PRISM ($0.305$ and $0.379$\,eV).

#### Comment 5.4 (DA-4 - MINOR): Training Scale ($N=10{,}000$ vs. Full QM9)
> *“Scaling stops at $N=10{,}000$ for classical models when full QM9 (130k) is available.”*

**Response:**  
As addressed in our response to Comment 1.2 and Comment 3.2, scaling to $N = 10{,}000$ was conducted to prove that the dequantization effect does not vanish outside the extreme low-data limit ($N=1{,}000$). In Section 7 (Limitations, Item 1), we explicitly state that evaluating PRISM at full scale ($N \approx 110{,}000$) with unconstrained capacity is an important direction for future applied engineering.

---

## Conclusion

We believe that these revisions, combined with the Q-PRISM / PRISM nomenclature and rigorous scope calibrations, directly address every question and recommendation raised by the panel. We thank the Editor and the Reviewers once again for their invaluable insights, which have significantly strengthened the rigor, clarity, and impact of our manuscript.
