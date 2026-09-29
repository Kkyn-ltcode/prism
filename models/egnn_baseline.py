"""
E(n) Equivariant Graph Neural Network (EGNN) Baseline
=====================================================
Classical baseline implementation from:
"E(n) Equivariant Graph Neural Networks" (Satorras, Hoogeboom, Welling, ICML 2021).
"""

import torch
import torch.nn as nn

class EGNNLayer(nn.Module):
    """
    A single classical E(n) Equivariant Graph Neural Network layer.
    """
    def __init__(self, in_features, hidden_features=16, edge_feat_dim=0):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features

        # Message MLP: phi_e(h_i, h_j, ||x_i - x_j||^2, a_ij) -> m_ij
        msg_in_dim = in_features * 2 + 1 + edge_feat_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(msg_in_dim, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU()
        )

        # Coordinate MLP: phi_x(m_ij) -> scalar weight
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, 1, bias=False)
        )

        # Node Feature MLP: phi_h(h_i, m_i) -> h_i_new
        self.node_mlp = nn.Sequential(
            nn.Linear(in_features + hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, in_features)
        )

    def forward(self, h, x, edge_index):
        """
        Args:
            h: [N, in_features] Node feature matrix.
            x: [N, 3] 3D coordinates.
            edge_index: [2, E] Adjacency list (row, col).
            
        Returns:
            h_new: [N, in_features]
            x_new: [N, 3]
        """
        row, col = edge_index[0], edge_index[1]

        # 1. Compute Invariant Squared Euclidean Distance
        rel_x = x[row] - x[col]                             # [E, 3]
        d_ij = torch.sum(rel_x**2, dim=-1, keepdim=True)   # [E, 1]

        # 2. Compute Edge Message
        edge_input = torch.cat([h[row], h[col], d_ij], dim=-1) # [E, 2*in_feat + 1]
        m_ij = self.edge_mlp(edge_input)                    # [E, hidden_features]

        # 3. Equivariant Coordinate Update
        phi_x = self.coord_mlp(m_ij)                        # [E, 1]
        coord_msg = rel_x * phi_x                           # [E, 3]
        x_update = torch.zeros_like(x)
        x_update.index_add_(0, row, coord_msg)
        
        # Exact per-node degree normalization: |N(i)|
        degrees = torch.zeros((x.size(0), 1), device=x.device)
        degrees.index_add_(0, row, torch.ones((row.size(0), 1), device=x.device))
        x_new = x + x_update / degrees.clamp(min=1.0)

        # 4. Invariant Node Feature Update
        m_i = torch.zeros((h.size(0), m_ij.size(1)), device=h.device)
        m_i.index_add_(0, row, m_ij)
        h_new = h + self.node_mlp(torch.cat([h, m_i], dim=-1))

        return h_new, x_new


class EGNNModel(nn.Module):
    """
    End-to-End Classical EGNN for Molecular Property Prediction.
    """
    def __init__(self, in_features=11, hidden_features=16, num_layers=3, out_dim=1):
        super().__init__()
        self.embedding = nn.Linear(in_features, hidden_features)
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_features, hidden_features=hidden_features) for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, out_dim)
        )

    def forward(self, h, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        h = self.embedding(h)
        for layer in self.layers:
            h, x = layer(h, x, edge_index)

        # Global Invariant Mean Pooling
        num_graphs = int(batch.max().item()) + 1
        out_pool = torch.zeros((num_graphs, h.size(1)), device=h.device)
        out_pool.index_add_(0, batch, h)
        
        counts = torch.zeros((num_graphs, 1), device=h.device)
        counts.index_add_(0, batch, torch.ones((h.size(0), 1), device=h.device))
        out_pool = out_pool / counts.clamp(min=1.0)

        return self.head(out_pool)


def count_parameters(model):
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_param_matched_egnn(target_params, in_features=11, num_layers=2, out_dim=1):
    """
    Create an EGNN model whose parameter count closely matches `target_params`.
    Searches over hidden_features to find the best match.
    
    Args:
        target_params: Target number of trainable parameters to match.
        in_features: Input feature dimension.
        num_layers: Number of EGNN layers.
        out_dim: Output dimension.
        
    Returns:
        (model, actual_params): The EGNN model and its actual parameter count.
    """
    best_model = None
    best_diff = float("inf")
    best_params = 0
    best_hf = 4
    
    for hf in range(4, 128):
        model = EGNNModel(
            in_features=in_features,
            hidden_features=hf,
            num_layers=num_layers,
            out_dim=out_dim
        )
        params = count_parameters(model)
        diff = abs(params - target_params)
        if diff < best_diff:
            best_diff = diff
            best_model = model
            best_params = params
            best_hf = hf
        if params > target_params + 100:
            break
    
    print(f"Param-Matched EGNN: hidden_features={best_hf}, params={best_params:,} (target={target_params:,}, diff={best_diff})")
    return best_model, best_params


class ParamMatchedEGNNModel(EGNNModel):
    """
    Parameter-Matched EGNN Baseline (Tier 1 Iso-Parameter Comparison).
    
    Default configuration: hidden_features=8, num_layers=2
    Expected param count with in_features=11: ~1,185
    
    For exact parameter matching against a specific Q-PRISM configuration,
    use `create_param_matched_egnn(target_params)` factory instead.
    """
    def __init__(self, in_features=11, num_layers=2, out_dim=1, hidden_features=8):
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            num_layers=num_layers,
            out_dim=out_dim
        )


if __name__ == "__main__":
    print("Testing Classical EGNN Invariance...")
    model = EGNNModel(in_features=5, hidden_features=8, num_layers=2)
    model.eval()

    h = torch.randn(4, 5)
    x = torch.randn(4, 3)
    edge_index = torch.tensor([[0, 1, 0, 2, 0, 3, 1, 2, 2, 3], [1, 0, 2, 0, 3, 0, 2, 1, 3, 2]])
    batch = torch.tensor([0, 0, 0, 0])

    pred1 = model(h, x, edge_index, batch)
    
    # SO(3) rotation + translation
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    x_rot = x @ q.T + torch.tensor([[2.0, -1.0, 3.0]])

    pred2 = model(h, x_rot, edge_index, batch)
    err = torch.abs(pred1 - pred2).item()
    print(f"EGNN Invariance Difference: {err:.2e}")
    assert err < 1e-5
    print("✅ Classical EGNN Invariance Test PASSED!")

    # Parameter count verification
    print("\n--- Parameter Count Report ---")
    egnn_std = EGNNModel(in_features=11, hidden_features=16, num_layers=2)
    egnn_pm = ParamMatchedEGNNModel(in_features=11, num_layers=2)
    print(f"Standard EGNN (hidden=16):       {count_parameters(egnn_std):,} params")
    print(f"Param-Matched EGNN (hidden=8):   {count_parameters(egnn_pm):,} params")
    
    # Auto-search demo
    print("\n--- Auto Param-Match Search ---")
    create_param_matched_egnn(target_params=2500, in_features=11, num_layers=2)
    print("✅ All EGNN baseline tests PASSED!")
