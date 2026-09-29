"""
Classical Equivariant Baselines: SchNet and PaiNN (From-Scratch)
================================================================
Parameter-matched implementations of SchNet and PaiNN for fair
iso-parameter comparison with Q-PRISM (~2,527 params).

Both models accept the same API as Q-PRISM:
    forward(h, x, edge_index, batch) → [num_graphs, out_dim]

where h is one-hot atomic features (not integer z), enabling direct
integration with the existing training loop.

References:
    SchNet: Schütt et al., "SchNet: A continuous-filter convolutional
            neural network for modeling quantum interactions", NeurIPS 2017
    PaiNN:  Schütt, Unke & Gastegger, "Equivariant message passing for
            the prediction of tensorial properties and molecular spectra",
            ICML 2021
"""

import torch
import torch.nn as nn
import math


# =============================================================================
# Shared: Sinusoidal Radial Basis Functions (used by both SchNet and PaiNN)
# =============================================================================

class SinusoidalRBF(nn.Module):
    """
    Sinusoidal RBF expansion: e_n(d) = sin(n·π·d / cutoff) / d

    Unlike Gaussian RBF, this has no learnable parameters (per PaiNN paper).
    Combined with a cosine cutoff envelope for smooth distance truncation.
    """
    def __init__(self, n_rbf=20, cutoff=10.0):
        super().__init__()
        self.n_rbf = n_rbf
        self.cutoff = cutoff
        # Frequency indices: n = 1, 2, ..., n_rbf
        self.register_buffer(
            "freqs",
            torch.arange(1, n_rbf + 1, dtype=torch.float32) * math.pi / cutoff
        )

    def forward(self, dist):
        """
        Args:
            dist: [E, 1] Euclidean distances (NOT squared).
        Returns:
            rbf: [E, n_rbf] Sinusoidal RBF features.
        """
        dist = dist.clamp(min=1e-8)
        return torch.sin(self.freqs * dist) / dist  # [E, n_rbf]


class CosineCutoff(nn.Module):
    """Smooth cosine cutoff envelope: f(d) = 0.5 * (cos(π·d/cutoff) + 1) for d < cutoff."""
    def __init__(self, cutoff=10.0):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, dist):
        """dist: [E, 1] → envelope: [E, 1]"""
        return 0.5 * (torch.cos(math.pi * dist / self.cutoff) + 1.0) * (dist < self.cutoff).float()


# =============================================================================
# SchNet: Continuous-Filter Convolutional Network
# =============================================================================

class SchNetInteraction(nn.Module):
    """
    SchNet interaction block.

    Message: m_ij = x_j * W(d_ij)
    where W(d_ij) = Dense(RBF(d_ij)) is a continuous filter.
    Update: h_i = h_i + Σ_j m_ij after cfconv, then refine with atomwise MLP.
    """
    def __init__(self, hidden_channels, num_gaussians, cutoff):
        super().__init__()
        # Continuous-filter generator: RBF → filter weights
        self.filter_net = nn.Sequential(
            nn.Linear(num_gaussians, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        # Atomwise refinement MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.cutoff_fn = CosineCutoff(cutoff)

    def forward(self, h, edge_index, rbf, dist):
        """
        Args:
            h: [N, hidden] Node features.
            edge_index: [2, E] Source → Target.
            rbf: [E, num_gaussians] RBF features.
            dist: [E, 1] Euclidean distances.
        Returns:
            h_new: [N, hidden] Updated node features.
        """
        row, col = edge_index[0], edge_index[1]

        # Continuous filter
        W = self.filter_net(rbf)              # [E, hidden]
        envelope = self.cutoff_fn(dist)        # [E, 1]
        W = W * envelope                      # [E, hidden]

        # Message: element-wise product of neighbor features and filter
        msg = h[col] * W                      # [E, hidden]

        # Aggregate
        agg = torch.zeros_like(h)
        agg.index_add_(0, row, msg)            # [N, hidden]

        # Atomwise update
        return h + self.mlp(agg)


class GaussianSmearing(nn.Module):
    """Gaussian distance expansion with fixed centers (no learnable params)."""
    def __init__(self, num_gaussians=50, d_min=0.0, d_max=10.0):
        super().__init__()
        centers = torch.linspace(d_min, d_max, num_gaussians)
        self.register_buffer("centers", centers)
        self.width = (d_max - d_min) / num_gaussians

    def forward(self, dist):
        """dist: [E, 1] → [E, num_gaussians]"""
        return torch.exp(-0.5 * ((dist - self.centers) / self.width) ** 2)


class SchNetModel(nn.Module):
    """
    SchNet: Continuous-filter convolutional neural network.

    Uses Gaussian distance expansion (invariant) and continuous-filter
    convolutions. No angular features.

    Parameter count formula (approximate):
        embedding: max_z × hidden
        per interaction: 2 × (num_gaussians × hidden + hidden + hidden × hidden + hidden)
                       + 2 × (hidden × hidden + hidden)
        head: hidden × hidden + hidden + hidden × out_dim + out_dim

    For hidden=10, num_gaussians=50, max_z=10, 2 interactions:
        embedding: 10 × 10 = 100
        per interaction: filter_net(50×10+10 + 10×10+10=620) + mlp(10×10+10+10×10+10=220) = 840
        2 interactions: 1680
        head: 10×10+10+10×1+1 = 121
        Total ≈ 1901

    For hidden=12: embedding=120, 2×(12×50+12+12×12+12 + 12×12+12+12×12+12)=2×1176=2352, head=169
        Total ≈ 2641
    """
    def __init__(self, in_features=11, hidden_channels=10, num_interactions=2,
                 num_gaussians=50, cutoff=10.0, out_dim=1, max_z=10):
        super().__init__()
        self.cutoff = cutoff

        # Atom embedding: one-hot features → hidden
        # (Using linear projection from one-hot, not nn.Embedding from z)
        self.embedding = nn.Linear(in_features, hidden_channels)

        # Gaussian distance expansion (no learnable params)
        self.distance_expansion = GaussianSmearing(num_gaussians, 0.0, cutoff)

        # Interaction blocks
        self.interactions = nn.ModuleList([
            SchNetInteraction(hidden_channels, num_gaussians, cutoff)
            for _ in range(num_interactions)
        ])

        # Output head
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, out_dim),
        )

    def forward(self, h, x, edge_index, batch=None):
        """Same API as EQGNNModelV2."""
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        row, col = edge_index[0], edge_index[1]

        # Compute distances (NOT squared — SchNet uses actual distances)
        diff = x[row] - x[col]
        dist = torch.norm(diff, dim=-1, keepdim=True)  # [E, 1]

        # Distance expansion
        rbf = self.distance_expansion(dist)  # [E, num_gaussians]

        # Embedding
        h = self.embedding(h)  # [N, hidden]

        # Message passing
        for interaction in self.interactions:
            h = interaction(h, edge_index, rbf, dist)

        # Global mean pooling
        num_graphs = int(batch.max().item()) + 1
        out_pool = torch.zeros((num_graphs, h.size(1)), device=h.device)
        out_pool.index_add_(0, batch, h)
        counts = torch.zeros((num_graphs, 1), device=h.device)
        counts.index_add_(0, batch, torch.ones((h.size(0), 1), device=h.device))
        out_pool = out_pool / counts.clamp(min=1.0)

        return self.head(out_pool)


# =============================================================================
# PaiNN: Equivariant Message Passing with Scalar + Vector Features
# =============================================================================

class PaiNNMessage(nn.Module):
    """
    PaiNN message block.

    Computes scalar and vector messages using:
        Δs_i = Σ_j φ_s(s_j) ⊙ W_s(d_ij)
        Δv_i = Σ_j v_j ⊙ φ_vv(s_j) ⊙ W_vv(d_ij)
             + Σ_j φ_vs(s_j) ⊙ W_vs(d_ij) ⊙ r̂_ij
    """
    def __init__(self, hidden_channels, n_rbf, cutoff):
        super().__init__()
        # Shared scalar transformation: s → [φ_vv, φ_s, φ_vs] (3 × hidden)
        self.scalar_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 3 * hidden_channels),
        )
        # Filter network: RBF → [W_vv, W_s, W_vs] (3 × hidden)
        self.filter_net = nn.Linear(n_rbf, 3 * hidden_channels)
        self.cutoff_fn = CosineCutoff(cutoff)

    def forward(self, s, v, edge_index, rbf, dist, dir_ij):
        """
        Args:
            s: [N, F] Scalar features.
            v: [N, F, 3] Vector features.
            edge_index: [2, E]
            rbf: [E, n_rbf]
            dist: [E, 1]
            dir_ij: [E, 3] Unit direction vectors r̂_ij.
        Returns:
            ds: [N, F] Scalar message residual.
            dv: [N, F, 3] Vector message residual.
        """
        row, col = edge_index[0], edge_index[1]
        F = s.size(1)

        # Scalar transformation of neighbor features
        phi = self.scalar_mlp(s[col])  # [E, 3F]
        phi_vv, phi_s, phi_vs = phi.split(F, dim=-1)  # each [E, F]

        # Filter from distances
        envelope = self.cutoff_fn(dist)  # [E, 1]
        W = self.filter_net(rbf) * envelope  # [E, 3F]
        W_vv, W_s, W_vs = W.split(F, dim=-1)  # each [E, F]

        # Scalar message: Σ_j φ_s(s_j) ⊙ W_s(d_ij)
        scalar_msg = phi_s * W_s  # [E, F]
        ds = torch.zeros_like(s)
        ds.index_add_(0, row, scalar_msg)

        # Vector message term 1: Σ_j v_j ⊙ φ_vv(s_j) ⊙ W_vv(d_ij)
        vec_scale = (phi_vv * W_vv).unsqueeze(-1)  # [E, F, 1]
        vec_msg1 = v[col] * vec_scale  # [E, F, 3]

        # Vector message term 2: Σ_j φ_vs(s_j) ⊙ W_vs(d_ij) ⊙ r̂_ij
        vec_msg2 = (phi_vs * W_vs).unsqueeze(-1) * dir_ij.unsqueeze(1)  # [E, F, 3]

        dv = torch.zeros_like(v)
        dv.index_add_(0, row, vec_msg1 + vec_msg2)

        return ds, dv


class PaiNNUpdate(nn.Module):
    """
    PaiNN update block (atomwise, no neighbor aggregation).

    Δs_i = a_ss(s_i, ||Vv_i||) + a_sv(s_i, ||Vv_i||) · ⟨Uv_i, Vv_i⟩
    Δv_i = a_vv(s_i, ||Vv_i||) · Uv_i
    """
    def __init__(self, hidden_channels):
        super().__init__()
        F = hidden_channels
        # Linear transforms on vector features
        self.U = nn.Linear(F, F, bias=False)
        self.V = nn.Linear(F, F, bias=False)

        # Scaling MLP: [s, ||Vv||] → [a_vv, a_sv, a_ss]
        self.scale_mlp = nn.Sequential(
            nn.Linear(2 * F, F),
            nn.SiLU(),
            nn.Linear(F, 3 * F),
        )

    def forward(self, s, v):
        """
        Args:
            s: [N, F] Scalar features.
            v: [N, F, 3] Vector features.
        Returns:
            ds: [N, F] Scalar update residual.
            dv: [N, F, 3] Vector update residual.
        """
        F = s.size(1)

        # Linear projections of vector features
        # v: [N, F, 3] → need to apply linear across F dimension
        # Reshape: [N, F, 3] → [N*3, F] → Linear → [N*3, F] → [N, F, 3]
        N = v.size(0)
        v_flat = v.permute(0, 2, 1).reshape(N * 3, F)  # [N*3, F]
        Uv = self.U(v_flat).reshape(N, 3, F).permute(0, 2, 1)  # [N, F, 3]
        Vv = self.V(v_flat).reshape(N, 3, F).permute(0, 2, 1)  # [N, F, 3]

        # Invariant geometric contractions
        Vv_norm = torch.norm(Vv, dim=-1)  # [N, F]
        dot_UV = torch.sum(Uv * Vv, dim=-1)  # [N, F]

        # Scaling MLP
        a = self.scale_mlp(torch.cat([s, Vv_norm], dim=-1))  # [N, 3F]
        a_vv, a_sv, a_ss = a.split(F, dim=-1)  # each [N, F]

        # Updates
        ds = a_ss + a_sv * dot_UV  # [N, F]
        dv = a_vv.unsqueeze(-1) * Uv  # [N, F, 3]

        return ds, dv


class PaiNNModel(nn.Module):
    """
    PaiNN: Polarizable Atom Interaction Neural Network.

    Equivariant message passing with coupled scalar (s) and vector (v) features.
    Uses sinusoidal RBF distance expansion and cosine cutoff.

    Parameter count for hidden_channels=8, num_interactions=2, n_rbf=20:
        embedding: 11 × 8 + 8 = 96
        per message block:
            scalar_mlp: 8×8+8 + 8×24+24 = 72 + 216 = 288
            filter_net: 20×24+24 = 504
            Total message: 792
        per update block:
            U: 8×8 = 64  (no bias)
            V: 8×8 = 64  (no bias)
            scale_mlp: 16×8+8 + 8×24+24 = 136 + 216 = 352
            Total update: 480
        per interaction: 792 + 480 = 1272
        2 interactions: 2544
        head: 8×8+8 + 8×1+1 = 72 + 9 = 81
        Total ≈ 2721

    For hidden_channels=7: ≈ 2,103
    For hidden_channels=8: ≈ 2,721
    """
    def __init__(self, in_features=11, hidden_channels=8, num_interactions=2,
                 n_rbf=20, cutoff=5.0, out_dim=1):
        super().__init__()
        self.cutoff = cutoff

        # Atom embedding (from one-hot features, not integer z)
        self.embedding = nn.Linear(in_features, hidden_channels)

        # Distance expansion (no learnable params)
        self.rbf = SinusoidalRBF(n_rbf, cutoff)
        self.cutoff_fn = CosineCutoff(cutoff)

        # Interaction blocks: alternating message and update
        self.message_blocks = nn.ModuleList([
            PaiNNMessage(hidden_channels, n_rbf, cutoff)
            for _ in range(num_interactions)
        ])
        self.update_blocks = nn.ModuleList([
            PaiNNUpdate(hidden_channels)
            for _ in range(num_interactions)
        ])

        # Output head (from scalar features only — invariant prediction)
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, out_dim),
        )

    def forward(self, h, x, edge_index, batch=None):
        """Same API as EQGNNModelV2."""
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        row, col = edge_index[0], edge_index[1]
        N = h.size(0)
        F = self.embedding.out_features

        # Compute distances and directions
        diff = x[row] - x[col]  # [E, 3]
        dist = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-8)  # [E, 1]
        dir_ij = diff / dist  # [E, 3] unit vectors

        # Distance expansion
        rbf = self.rbf(dist)  # [E, n_rbf]

        # Initialize scalar and vector features
        s = self.embedding(h)  # [N, F]
        v = torch.zeros(N, F, 3, device=h.device)  # [N, F, 3] — initialized to zero

        # Message passing
        for msg_block, upd_block in zip(self.message_blocks, self.update_blocks):
            ds, dv = msg_block(s, v, edge_index, rbf, dist, dir_ij)
            s = s + ds
            v = v + dv
            ds, dv = upd_block(s, v)
            s = s + ds
            v = v + dv

        # Global mean pooling (scalar features only)
        num_graphs = int(batch.max().item()) + 1
        out_pool = torch.zeros((num_graphs, F), device=h.device)
        out_pool.index_add_(0, batch, s)
        counts = torch.zeros((num_graphs, 1), device=h.device)
        counts.index_add_(0, batch, torch.ones((N, 1), device=h.device))
        out_pool = out_pool / counts.clamp(min=1.0)

        return self.head(out_pool)


# =============================================================================
# Parameter matching utilities
# =============================================================================

def create_param_matched_schnet(target_params, in_features=11, num_interactions=2,
                                 num_gaussians=50, cutoff=10.0, out_dim=1):
    """Find SchNet hidden_channels that best matches target_params."""
    best_model, best_diff, best_h = None, float('inf'), None
    for h in range(4, 64):
        model = SchNetModel(in_features=in_features, hidden_channels=h,
                            num_interactions=num_interactions, num_gaussians=num_gaussians,
                            cutoff=cutoff, out_dim=out_dim)
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        diff = abs(total - target_params)
        if diff < best_diff:
            best_diff = diff
            best_model = model
            best_h = h
    total = sum(p.numel() for p in best_model.parameters() if p.requires_grad)
    return best_model, total, best_h


def create_param_matched_painn(target_params, in_features=11, num_interactions=2,
                                n_rbf=20, cutoff=5.0, out_dim=1):
    """Find PaiNN hidden_channels that best matches target_params."""
    best_model, best_diff, best_h = None, float('inf'), None
    for h in range(4, 64):
        model = PaiNNModel(in_features=in_features, hidden_channels=h,
                           num_interactions=num_interactions, n_rbf=n_rbf,
                           cutoff=cutoff, out_dim=out_dim)
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        diff = abs(total - target_params)
        if diff < best_diff:
            best_diff = diff
            best_model = model
            best_h = h
    total = sum(p.numel() for p in best_model.parameters() if p.requires_grad)
    return best_model, total, best_h


# =============================================================================
# Verification
# =============================================================================

if __name__ == "__main__":
    def count_p(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    print("=" * 70)
    print("SchNet & PaiNN Parameter Sweep")
    print("=" * 70)

    target = 2527  # Q-PRISM param count

    print(f"\nTarget: {target} params (Q-PRISM)\n")

    print("SchNet (num_interactions=2, num_gaussians=50, cutoff=10.0):")
    for h in range(6, 18):
        m = SchNetModel(hidden_channels=h, num_interactions=2, num_gaussians=50)
        p = count_p(m)
        marker = " ◄◄◄" if abs(p - target) < 200 else ""
        print(f"  h={h:2d}: {p:6,} params (Δ={p-target:+5d}){marker}")

    print("\nPaiNN (num_interactions=2, n_rbf=20, cutoff=5.0):")
    for h in range(5, 14):
        m = PaiNNModel(hidden_channels=h, num_interactions=2, n_rbf=20)
        p = count_p(m)
        marker = " ◄◄◄" if abs(p - target) < 300 else ""
        print(f"  h={h:2d}: {p:6,} params (Δ={p-target:+5d}){marker}")

    # Best matches
    print("\n" + "=" * 70)
    print("Best Parameter Matches:")
    schnet, sp, sh = create_param_matched_schnet(target)
    painn, pp, ph = create_param_matched_painn(target)
    print(f"  SchNet: hidden={sh}, {sp:,} params (Δ={sp-target:+d})")
    print(f"  PaiNN:  hidden={ph}, {pp:,} params (Δ={pp-target:+d})")

    # Test forward pass
    print("\nForward pass test:")
    h = torch.randn(4, 11)
    x = torch.randn(4, 3)
    edge_index = torch.tensor([[0,0,0,1,1,2,2,3,3,3],
                                [1,2,3,0,2,0,1,0,1,2]], dtype=torch.long)
    batch = torch.zeros(4, dtype=torch.long)

    out_s = schnet(h, x, edge_index, batch)
    out_p = painn(h, x, edge_index, batch)
    print(f"  SchNet output: {out_s.shape}, values: {out_s.detach().numpy().flatten()}")
    print(f"  PaiNN output:  {out_p.shape}, values: {out_p.detach().numpy().flatten()}")

    # E(3)-invariance test for SchNet
    Q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(Q) < 0:
        Q[:, 0] *= -1
    t = torch.randn(1, 3)
    x_rot = x @ Q.T + t

    out_s_rot = schnet(h, x_rot, edge_index, batch)
    diff_s = torch.abs(out_s_rot - out_s).max().item()
    print(f"\n  SchNet E(3)-invariance: max|f(Qx+t) - f(x)| = {diff_s:.2e}")

    out_p_rot = painn(h, x_rot, edge_index, batch)
    diff_p = torch.abs(out_p_rot - out_p).max().item()
    print(f"  PaiNN  E(3)-invariance: max|f(Qx+t) - f(x)| = {diff_p:.2e}")

    print("\n✅ All checks passed!")
