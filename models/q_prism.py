"""
Q-PRISM: Quantum Periodic Representation with Invariant Sinusoidal Message-passing
==================================================================================
Drop-in module re-exporting Q-PRISM architecture classes.
"""
from .eq_gnn import (
    QPRISMLayer, QPRISMModel, QPRISM,
    QPRISMLayerV1, QPRISMModelV1,
    QPRISMLayerV2, QPRISMModelV2,
    GaussianRBF, QuantumHead,
    compute_edge_mean_angles,
    # Backward compatibility aliases
    EQGNNLayer, EQGNNModel, EQGNN,
    EQGNNLayerV1, EQGNNModelV1,
    EQGNNLayerV2, EQGNNModelV2,
)

__all__ = [
    "QPRISMLayer", "QPRISMModel", "QPRISM",
    "QPRISMLayerV1", "QPRISMModelV1",
    "QPRISMLayerV2", "QPRISMModelV2",
    "GaussianRBF", "QuantumHead",
    "compute_edge_mean_angles",
    "EQGNNLayer", "EQGNNModel", "EQGNN",
    "EQGNNLayerV1", "EQGNNModelV1",
    "EQGNNLayerV2", "EQGNNModelV2",
]
