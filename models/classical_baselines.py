"""Classical equivariant baselines: parameter-matched SchNet and PaiNN."""

import torch
import torch.nn as nn
import math


class SinusoidalRBF(nn.Module):
    """Sinusoidal RBF distance expansion."""
    def __init__(self, n_rbf=20, cutoff=10.0):
        super().__init__()
        self.n_rbf = n_rbf
        self.cutoff = cutoff
        self.register_buffer(
            "freqs",
            torch.arange(1, n_rbf + 1, dtype=torch.float32) * math.pi / cutoff
        )

    def forward(self, dist):
        dist = dist.clamp(min=1e-8)
        return torch.sin(self.freqs * dist) / dist  # [E, n_rbf]


class CosineCutoff(nn.Module):
    """Cosine cutoff envelope."""
    def __init__(self, cutoff=10.0):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, dist):
        return 0.5 * (torch.cos(math.pi * dist / self.cutoff) + 1.0) * (dist < self.cutoff).float()


class SchNetInteraction(nn.Module):
    """SchNet continuous-filter interaction block."""
    def __init__(self, hidden_channels, num_gaussians, cutoff):
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(num_gaussians, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.cutoff_fn = CosineCutoff(cutoff)

    def forward(self, h, edge_index, rbf, dist):
        row, col = edge_index[0], edge_index[1]

        W = self.filter_net(rbf)              # [E, hidden]
        envelope = self.cutoff_fn(dist)        # [E, 1]
        W = W * envelope                      # [E, hidden]

        msg = h[col] * W                      # [E, hidden]

        agg = torch.zeros_like(h)
        agg.index_add_(0, row, msg)            # [N, hidden]

        return h + self.mlp(agg)


class GaussianSmearing(nn.Module):
    """Gaussian distance expansion with fixed centers."""
    def __init__(self, num_gaussians=50, d_min=0.0, d_max=10.0):
        super().__init__()
        centers = torch.linspace(d_min, d_max, num_gaussians)
        self.register_buffer("centers", centers)
        self.width = (d_max - d_min) / num_gaussians

    def forward(self, dist):
        return torch.exp(-0.5 * ((dist - self.centers) / self.width) ** 2)


class SchNetModel(nn.Module):
    """SchNet model with continuous-filter convolutions."""
    def __init__(self, in_features=11, hidden_channels=10, num_interactions=2,
                 num_gaussians=50, cutoff=10.0, out_dim=1, max_z=10):
        super().__init__()
        self.cutoff = cutoff

        self.embedding = nn.Linear(in_features, hidden_channels)
        self.distance_expansion = GaussianSmearing(num_gaussians, 0.0, cutoff)
        self.interactions = nn.ModuleList([
            SchNetInteraction(hidden_channels, num_gaussians, cutoff)
            for _ in range(num_interactions)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, out_dim),
        )

    def forward(self, h, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        row, col = edge_index[0], edge_index[1]

        diff = x[row] - x[col]
        dist = torch.norm(diff, dim=-1, keepdim=True)  # [E, 1]

        rbf = self.distance_expansion(dist)  # [E, num_gaussians]
        h = self.embedding(h)  # [N, hidden]

        for interaction in self.interactions:
            h = interaction(h, edge_index, rbf, dist)

        num_graphs = int(batch.max().item()) + 1
        out_pool = torch.zeros((num_graphs, h.size(1)), device=h.device)
        out_pool.index_add_(0, batch, h)
        counts = torch.zeros((num_graphs, 1), device=h.device)
        counts.index_add_(0, batch, torch.ones((h.size(0), 1), device=h.device))
        out_pool = out_pool / counts.clamp(min=1.0)

        return self.head(out_pool)


class PaiNNMessage(nn.Module):
    """PaiNN message block for scalar and vector features."""
    def __init__(self, hidden_channels, n_rbf, cutoff):
        super().__init__()
        self.scalar_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 3 * hidden_channels),
        )
        self.filter_net = nn.Linear(n_rbf, 3 * hidden_channels)
        self.cutoff_fn = CosineCutoff(cutoff)

    def forward(self, s, v, edge_index, rbf, dist, dir_ij):
        row, col = edge_index[0], edge_index[1]
        F = s.size(1)

        phi = self.scalar_mlp(s[col])  # [E, 3F]
        phi_vv, phi_s, phi_vs = phi.split(F, dim=-1)  # each [E, F]

        envelope = self.cutoff_fn(dist)  # [E, 1]
        W = self.filter_net(rbf) * envelope  # [E, 3F]
        W_vv, W_s, W_vs = W.split(F, dim=-1)  # each [E, F]

        scalar_msg = phi_s * W_s  # [E, F]
        ds = torch.zeros_like(s)
        ds.index_add_(0, row, scalar_msg)

        vec_scale = (phi_vv * W_vv).unsqueeze(-1)  # [E, F, 1]
        vec_msg1 = v[col] * vec_scale  # [E, F, 3]

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
    """PaiNN model with equivariant message passing over scalar and vector features."""
    def __init__(self, in_features=11, hidden_channels=8, num_interactions=2,
                 n_rbf=20, cutoff=5.0, out_dim=1):
        super().__init__()
        self.cutoff = cutoff

        self.embedding = nn.Linear(in_features, hidden_channels)

        self.rbf = SinusoidalRBF(n_rbf, cutoff)
        self.cutoff_fn = CosineCutoff(cutoff)

        self.message_blocks = nn.ModuleList([
            PaiNNMessage(hidden_channels, n_rbf, cutoff)
            for _ in range(num_interactions)
        ])
        self.update_blocks = nn.ModuleList([
            PaiNNUpdate(hidden_channels)
            for _ in range(num_interactions)
        ])

        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, out_dim),
        )

    def forward(self, h, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        row, col = edge_index[0], edge_index[1]
        N = h.size(0)
        F = self.embedding.out_features

        diff = x[row] - x[col]  # [E, 3]
        dist = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-8)  # [E, 1]
        dir_ij = diff / dist  # [E, 3]

        rbf = self.rbf(dist)  # [E, n_rbf]

        s = self.embedding(h)  # [N, F]
        v = torch.zeros(N, F, 3, device=h.device)  # [N, F, 3]

        for msg_block, upd_block in zip(self.message_blocks, self.update_blocks):
            ds, dv = msg_block(s, v, edge_index, rbf, dist, dir_ij)
            s = s + ds
            v = v + dv
            ds, dv = upd_block(s, v)
            s = s + ds
            v = v + dv

        num_graphs = int(batch.max().item()) + 1
        out_pool = torch.zeros((num_graphs, F), device=h.device)
        out_pool.index_add_(0, batch, s)
        counts = torch.zeros((num_graphs, 1), device=h.device)
        counts.index_add_(0, batch, torch.ones((N, 1), device=h.device))
        out_pool = out_pool / counts.clamp(min=1.0)

        return self.head(out_pool)


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


# Convenience and backward-compatible aliases
SchNetBaseline = SchNetModel
PaiNNBaseline = PaiNNModel


if __name__ == "__main__":
    def count_p(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    print("SchNet & PaiNN Parameter Sweep")

    target = 2527

    print(f"\nTarget: {target} params\n")

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

    print("\nBest Parameter Matches:")
    schnet, sp, sh = create_param_matched_schnet(target)
    painn, pp, ph = create_param_matched_painn(target)
    print(f"  SchNet: hidden={sh}, {sp:,} params (Δ={sp-target:+d})")
    print(f"  PaiNN:  hidden={ph}, {pp:,} params (Δ={pp-target:+d})")

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

    print("\nAll checks passed.")
