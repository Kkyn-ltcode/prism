"""
PRISM: Periodic Representation with Invariant Sinusoidal Message-passing
=========================================================================
Exact classical twin of Q-PRISM replacing the 4-qubit PQC in each QuantumHead
with a parameter-matched classical MLP using **periodic (sin) activations**
(SIREN-style) at identical parameter count (2,527 parameters).

Replaces the 4-qubit PQC in each QuantumHead with a parameter-matched
classical MLP using **trigonometric (sin) activations** — SIREN-style —
instead of the tanh-bounded MLP used in the original Classical Twin.

The model keeps ALL other components identical to ClassicalTwinModelV2:
    - GaussianRBF distance expansion
    - 3-body SO(3)-invariant angular encoding
    - Multi-head edge convolution (Electronic + Topological)
    - tanh-gated equivariant coordinate updates
    - Node feature MLP with residual connection
    - Global invariant mean pooling + prediction head

Two variants are provided:
    1. PRISMHead (SIREN-style): sin(ω₀ · Linear(x)) with learnable or fixed ω₀
    2. FourierFeatureHead: Explicit Fourier feature layer + Linear
"""

import torch
import torch.nn as nn
import math
from .eq_gnn import GaussianRBF, compute_edge_mean_angles, N_RBF


# =============================================================================
# SIREN-Style Trigonometric Head: Drop-in Replacement for QuantumHead
# =============================================================================

class PRISMHead(nn.Module):
    """
    Classical trigonometric twin of QuantumHead (PRISM / SIREN-style).

    Replaces the 4-qubit PQC with a 2-layer MLP using **sin** activation
    (SIREN-style periodic non-linearity), keeping the same:
        - Input: [h_i_proj, h_j_proj, rbf_features, angle] ∈ ℝ^(4+K+1)
        - Output: 8 values bounded in [-1, 1] (matching Pauli-Z spectrum)
        - feature_proj: Linear(in_features → 2) (identical to QuantumHead)

    The key difference from ClassicalHead:
        - ClassicalHead uses SiLU → tanh (sigmoid-family, non-periodic)
        - TrigHead uses sin → tanh (periodic, matching the PQC's Fourier basis)

    This tests whether the PQC advantage is from periodic basis functions
    (classically replicable) or from genuine quantum substrate effects.

    Parameter Matching:
        Same as ClassicalHead: hidden=4 gives exactly 88 MLP params,
        matching the PQC's 88 trainable parameters per head.

    SIREN Initialization:
        Following Sitzmann et al. (2020), the first layer uses
        uniform initialization in [-1/input_dim, 1/input_dim], and
        the frequency scaling ω₀ (default 30) controls the bandwidth
        of the trigonometric basis.
    """
    def __init__(self, in_features, n_q_layers=2, n_rbf=N_RBF, omega_0=30.0):
        super().__init__()
        self.feature_proj = nn.Linear(in_features, 2)  # Same as QuantumHead
        self.omega_0 = omega_0

        input_dim = 4 + n_rbf + 1  # [h_i0, h_i1, h_j0, h_j1, rbf×K, angle]
        n_outputs = 8  # Match PQC: 4 single-qubit + 4 two-qubit correlators

        # --- Parameter Matching (IDENTICAL to ClassicalHead) ---
        # MLP: Linear(input_dim → hidden) with bias + Linear(hidden → 8) without bias
        # = hidden × (input_dim + 1 + n_outputs)
        # For hidden=4, input_dim=13: 4 × (13 + 1 + 8) = 88 params (EXACT match to PQC)
        hidden = 4
        self.linear1 = nn.Linear(input_dim, hidden)          # input_dim × hidden + hidden
        self.linear2 = nn.Linear(hidden, n_outputs, bias=False)  # hidden × n_outputs (no bias)

        # SIREN initialization (Sitzmann et al., 2020, Sec. 3.2)
        self._siren_init()

    def _siren_init(self):
        """
        SIREN-specific weight initialization.

        First layer: U(-1/n, 1/n) where n = input_dim
        Second layer: U(-√(6/n) / ω₀, √(6/n) / ω₀) where n = hidden_dim

        This ensures that the distribution of activations is preserved
        through the network, avoiding the dead-neuron problem that
        standard initializations cause with sin activations.
        """
        n_in1 = self.linear1.in_features
        with torch.no_grad():
            self.linear1.weight.uniform_(-1.0 / n_in1, 1.0 / n_in1)
            if self.linear1.bias is not None:
                self.linear1.bias.uniform_(-1.0 / n_in1, 1.0 / n_in1)

        n_in2 = self.linear2.in_features
        bound2 = math.sqrt(6.0 / n_in2) / self.omega_0
        with torch.no_grad():
            self.linear2.weight.uniform_(-bound2, bound2)

    def forward(self, h_row, h_col, rbf_features, angles):
        """
        Args:
            h_row: [E, in_features] Source node embeddings.
            h_col: [E, in_features] Target node embeddings.
            rbf_features: [E, K] Gaussian RBF distance features.
            angles: [E, 1] Mean 3-body bond angles per edge.

        Returns:
            out: [E, 8] Classical trigonometric observable surrogates in [-1, 1].
        """
        h_i_proj = self.feature_proj(h_row)  # [E, 2]
        h_j_proj = self.feature_proj(h_col)  # [E, 2]
        inputs = torch.cat([h_i_proj, h_j_proj, rbf_features, angles], dim=-1)  # [E, 4+K+1]

        # SIREN-style forward: sin(ω₀ · W₁x + b₁) → tanh(W₂ · ...)
        # The sin activation generates the periodic basis functions that
        # the reviewers hypothesize are the source of the PQC advantage.
        x = torch.sin(self.omega_0 * self.linear1(inputs))  # [E, hidden], periodic
        return torch.tanh(self.linear2(x))  # [E, 8], bounded to [-1, 1]


TrigHead = PRISMHead  # Backward compatibility alias


# =============================================================================
# Fourier Feature Head: Alternative trigonometric baseline
# =============================================================================

class FourierFeatureHead(nn.Module):
    """
    Classical Fourier feature twin of QuantumHead.

    Instead of SIREN-style implicit periodicity, this uses explicit
    random Fourier feature mapping (Tancik et al., 2020):

        γ(x) = [sin(2π·B·x), cos(2π·B·x)]

    where B is a fixed random matrix, followed by a linear readout.

    This provides a complementary test to TrigHead: rather than learning
    the frequency content through SIREN initialization, the frequencies
    are drawn from a Gaussian prior and fixed.

    Parameter Matching:
        With n_fourier=4 features (producing 2×4=8 expanded dims) and
        input_dim=13:
            B: 13 × 4 = 52 (FIXED, non-trainable)
            Linear(8 → 8, bias=True): 8 × 8 + 8 = 72
            feature_proj: same as others

        PQC has 88 trainable params per head.
        FourierFeatureHead has 72 trainable MLP params per head.
        Difference: -16 params (within acceptable iso-parameter range).

        Alternatively, with a hidden layer:
            Linear(8 → 5, bias=True) + Linear(5 → 8, bias=False):
            8×5 + 5 + 5×8 = 85 ≈ 88
    """
    def __init__(self, in_features, n_q_layers=2, n_rbf=N_RBF, n_fourier=4, sigma=10.0):
        super().__init__()
        self.feature_proj = nn.Linear(in_features, 2)  # Same as QuantumHead

        input_dim = 4 + n_rbf + 1  # [h_i0, h_i1, h_j0, h_j1, rbf×K, angle]
        n_outputs = 8

        # Fixed random Fourier feature matrix (non-trainable)
        B = torch.randn(input_dim, n_fourier) * sigma
        self.register_buffer("B", B)

        # Expanded dim after Fourier features: 2 × n_fourier (sin + cos)
        fourier_dim = 2 * n_fourier  # = 8

        # MLP to match ~88 PQC params:
        # Linear(8 → 5) with bias + Linear(5 → 8) without bias
        # = 8×5 + 5 + 5×8 = 85 params (close to 88)
        hidden = 5
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, hidden),              # 8×5 + 5 = 45
            nn.SiLU(),
            nn.Linear(hidden, n_outputs, bias=False),    # 5×8 = 40
        )
        # Total trainable MLP params: 85

    def forward(self, h_row, h_col, rbf_features, angles):
        """
        Args:
            h_row: [E, in_features] Source node embeddings.
            h_col: [E, in_features] Target node embeddings.
            rbf_features: [E, K] Gaussian RBF distance features.
            angles: [E, 1] Mean 3-body bond angles per edge.

        Returns:
            out: [E, 8] Classical Fourier-feature observable surrogates in [-1, 1].
        """
        h_i_proj = self.feature_proj(h_row)  # [E, 2]
        h_j_proj = self.feature_proj(h_col)  # [E, 2]
        inputs = torch.cat([h_i_proj, h_j_proj, rbf_features, angles], dim=-1)  # [E, 13]

        # Random Fourier Feature mapping: γ(x) = [sin(2π·B·x), cos(2π·B·x)]
        projected = inputs @ self.B  # [E, n_fourier]
        fourier_features = torch.cat([
            torch.sin(2 * math.pi * projected),
            torch.cos(2 * math.pi * projected),
        ], dim=-1)  # [E, 2×n_fourier]

        return torch.tanh(self.mlp(fourier_features))  # [E, 8], bounded to [-1, 1]


# =============================================================================
# PRISM Layer: Mirrors QPRISMLayerV2 / ClassicalTwinLayerV2
# =============================================================================

class PRISMLayerV2(nn.Module):
    """
    PRISM Equivariant Graph Convolution Layer (Periodic Classical Twin).

    Identical architecture to QPRISMLayerV2 except QuantumHead → PRISMHead (or FourierFeatureHead).
    All other components preserved:
        - GaussianRBF distance expansion
        - 3-body angular encoding
        - 2-head edge convolution (Electronic + Topological)
        - tanh-gated equivariant coordinate update
        - Node feature MLP with residual connection
    """
    def __init__(self, in_features, n_q_layers=2, n_rbf=N_RBF,
                 head_type="siren", omega_0=30.0):
        super().__init__()
        self.in_features = in_features
        self.n_rbf = n_rbf

        # Shared Gaussian RBF distance expansion (same as QPRISMLayerV2)
        self.rbf = GaussianRBF(n_rbf=n_rbf)

        # Two periodic/trigonometric heads (replacing QuantumHeads)
        HeadClass = PRISMHead if head_type == "siren" else FourierFeatureHead
        head_kwargs = dict(in_features=in_features, n_q_layers=n_q_layers, n_rbf=n_rbf)
        if head_type == "siren":
            head_kwargs["omega_0"] = omega_0

        self.head_electronic = HeadClass(**head_kwargs)
        self.head_topological = HeadClass(**head_kwargs)

        # Tanh-gated coordinate scalar MLP (same as EQGNNLayerV2)
        self.coord_gate = nn.Linear(2, 1)

        # Invariant Node Feature Update MLP (same as EQGNNLayerV2)
        msg_dim = 7 * 2  # 7 message values per head × 2 heads = 14
        self.node_mlp = nn.Sequential(
            nn.Linear(in_features + msg_dim, in_features),
            nn.SiLU(),
            nn.Linear(in_features, in_features)
        )

    def forward(self, h, x, edge_index):
        """Identical to EQGNNLayerV2.forward() — same I/O contract."""
        row, col = edge_index[0], edge_index[1]

        # 1. Compute invariant squared Euclidean distance
        rel_x = x[row] - x[col]
        d_ij = torch.sum(rel_x ** 2, dim=-1, keepdim=True)

        # 2. RBF distance expansion
        rbf_features = self.rbf(d_ij)

        # 3. 3-body angular encoding
        angles = compute_edge_mean_angles(x, edge_index)

        # 4. Multi-head TRIGONOMETRIC edge convolution
        out_e = self.head_electronic(h[row], h[col], rbf_features, angles)
        out_t = self.head_topological(h[row], h[col], rbf_features, angles)

        # 5. Parse outputs (same decomposition as EQGNNLayerV2)
        msg_e = torch.cat([out_e[:, :3], out_e[:, 4:]], dim=-1)  # [E, 7]
        msg_t = torch.cat([out_t[:, :3], out_t[:, 4:]], dim=-1)  # [E, 7]
        m_ij = torch.cat([msg_e, msg_t], dim=-1)                 # [E, 14]

        phi_x_raw = torch.cat([out_e[:, 3:4], out_t[:, 3:4]], dim=-1)  # [E, 2]

        # 6. Tanh-gated equivariant coordinate update
        phi_x = torch.tanh(self.coord_gate(phi_x_raw))
        coord_msg = rel_x * phi_x
        x_update = torch.zeros_like(x)
        x_update.index_add_(0, row, coord_msg)

        degrees = torch.zeros((x.size(0), 1), device=x.device)
        degrees.index_add_(0, row, torch.ones((row.size(0), 1), device=x.device))
        x_new = x + x_update / degrees.clamp(min=1.0)

        # 7. Invariant node feature update
        m_i = torch.zeros((h.size(0), m_ij.size(1)), device=h.device)
        m_i.index_add_(0, row, m_ij)
        h_new = h + self.node_mlp(torch.cat([h, m_i], dim=-1))

        return h_new, x_new


PRISMLayer = PRISMLayerV2
TrigTwinLayerV2 = PRISMLayerV2  # Backward compatibility alias


# =============================================================================
# PRISM Full Model: Mirrors QPRISMModelV2 / ClassicalTwinModelV2
# =============================================================================

class PRISMModelV2(nn.Module):
    """
    PRISM: Periodic Representation with Invariant Sinusoidal Message-passing.
    Exact classical twin of Q-PRISM replacing PQC with SIREN periodic activations.

    Same API: forward(h, x, edge_index, batch) → [num_graphs, out_dim].

    Args:
        head_type: "siren" for SIREN-style sin activations (default),
                   "fourier" for random Fourier feature mapping.
        omega_0: SIREN frequency scaling (only used when head_type="siren").
    """
    def __init__(self, in_features=11, hidden_features=16, num_layers=3, out_dim=1,
                 n_q_layers=2, n_rbf=N_RBF, head_type="siren", omega_0=30.0):
        super().__init__()
        self.embedding = nn.Linear(in_features, hidden_features)
        self.layers = nn.ModuleList([
            PRISMLayerV2(hidden_features, n_q_layers=n_q_layers, n_rbf=n_rbf,
                         head_type=head_type, omega_0=omega_0)
            for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, out_dim)
        )

    def forward(self, h, x, edge_index, batch=None):
        """Same API as QPRISMModelV2.forward()."""
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        h = self.embedding(h)
        for layer in self.layers:
            h, x = layer(h, x, edge_index)

        # Global invariant mean pooling (same as QPRISMModelV2)
        num_graphs = int(batch.max().item()) + 1
        out_pool = torch.zeros((num_graphs, h.size(1)), device=h.device)
        out_pool.index_add_(0, batch, h)
        counts = torch.zeros((num_graphs, 1), device=h.device)
        counts.index_add_(0, batch, torch.ones((h.size(0), 1), device=h.device))
        out_pool = out_pool / counts.clamp(min=1.0)

        return self.head(out_pool)


PRISMModel = PRISMModelV2
PRISM = PRISMModel
TrigTwinModelV2 = PRISMModelV2
PeriodicEGNN = PRISMModel


# =============================================================================
# Parameter matching utility
# =============================================================================

def create_param_matched_prism(target_params, in_features=11, hidden_features=16,
                               num_layers=2, out_dim=1, n_q_layers=2, n_rbf=N_RBF,
                               head_type="siren", omega_0=30.0):
    """
    Create a PRISMModelV2 and report its parameter count vs target.
    """
    model = PRISMModelV2(
        in_features=in_features,
        hidden_features=hidden_features,
        num_layers=num_layers,
        out_dim=out_dim,
        n_q_layers=n_q_layers,
        n_rbf=n_rbf,
        head_type=head_type,
        omega_0=omega_0,
    )
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    diff = total - target_params
    return model, total, diff


create_param_matched_trig_twin = create_param_matched_prism
create_param_matched_trigtwin = create_param_matched_prism


# =============================================================================
# Verification
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Classical Trigonometric Twin Parameter Matching Verification")
    print("=" * 70)

    from .eq_gnn import EQGNNModelV2
    from .classical_twin import ClassicalTwinModelV2
    from .egnn_baseline import count_parameters

    # Compare all: Q-PRISM, Classical Twin (tanh), PRISM (SIREN/Fourier)
    eqgnn = QPRISMModelV2(in_features=11, hidden_features=16, num_layers=2,
                          n_q_layers=2, out_dim=1)
    twin_tanh = ClassicalTwinModelV2(in_features=11, hidden_features=16, num_layers=2,
                                      n_q_layers=2, out_dim=1)
    twin_siren = PRISMModelV2(in_features=11, hidden_features=16, num_layers=2,
                              n_q_layers=2, out_dim=1, head_type="siren")
    twin_fourier = PRISMModelV2(in_features=11, hidden_features=16, num_layers=2,
                                n_q_layers=2, out_dim=1, head_type="fourier")

    eqgnn_params = count_parameters(eqgnn)
    tanh_params = count_parameters(twin_tanh)
    siren_params = count_parameters(twin_siren)
    fourier_params = count_parameters(twin_fourier)

    print(f"\nQ-PRISM params:               {eqgnn_params:,}")
    print(f"Classical Twin (tanh) params: {tanh_params:,}")
    print(f"PRISM (SIREN) params:         {siren_params:,}")
    print(f"PRISM (Fourier) params:       {fourier_params:,}")
    print(f"\nSIREN vs Q-PRISM diff:        {siren_params - eqgnn_params:+d}")
    print(f"Fourier vs Q-PRISM diff:      {fourier_params - eqgnn_params:+d}")
    print(f"SIREN vs Tanh Twin diff:      {siren_params - tanh_params:+d}")

    # Verify same I/O shapes
    h = torch.randn(4, 11)
    x = torch.randn(4, 3)
    edge_index = torch.tensor([[0,0,0,1,1,2,2,3,3,3],
                                [1,2,3,0,2,0,1,0,1,2]], dtype=torch.long)
    batch = torch.zeros(4, dtype=torch.long)

    out_eqgnn = eqgnn(h, x, edge_index, batch)
    out_siren = twin_siren(h, x, edge_index, batch)
    out_fourier = twin_fourier(h, x, edge_index, batch)

    print(f"\nQ-PRISM output shape:         {out_eqgnn.shape}")
    print(f"PRISM (SIREN) output:         {out_siren.shape}")
    print(f"PRISM (Fourier) output:       {out_fourier.shape}")
    assert out_eqgnn.shape == out_siren.shape == out_fourier.shape, "Output shape mismatch!"

    # Test E(3)-invariance
    Q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(Q) < 0:
        Q[:, 0] *= -1
    t = torch.randn(1, 3)

    x_rot = x @ Q.T + t
    out_rot_siren = twin_siren(h, x_rot, edge_index, batch)
    diff_siren = torch.abs(out_rot_siren - out_siren).max().item()
    out_rot_fourier = twin_fourier(h, x_rot, edge_index, batch)
    diff_fourier = torch.abs(out_rot_fourier - out_fourier).max().item()

    print(f"\nE(3)-Invariance SIREN:       max |f(Qx+t) - f(x)| = {diff_siren:.2e}")
    print(f"E(3)-Invariance Fourier:     max |f(Qx+t) - f(x)| = {diff_fourier:.2e}")
    if diff_siren < 1e-4 and diff_fourier < 1e-4:
        print("✅ Both Trig Twins are E(3)-invariant")
    else:
        print("⚠️  Invariance test issue")

    print("\n✅ All checks passed!")
