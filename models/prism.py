"""
PRISM: Periodic Representation with Invariant Sinusoidal Message-passing
=========================================================================
Drop-in module re-exporting PRISM classical periodic architecture classes.
"""
from .classical_trig_twin import (
    PRISMHead, PRISMLayer, PRISMModel, PRISM,
    PRISMLayerV2, PRISMModelV2,
    FourierFeatureHead,
    create_param_matched_prism,
    # Backward compatibility aliases
    TrigHead, TrigTwinLayerV2, TrigTwinModelV2,
    create_param_matched_trig_twin, create_param_matched_trigtwin, PeriodicEGNN,
)

__all__ = [
    "PRISMHead", "PRISMLayer", "PRISMModel", "PRISM",
    "PRISMLayerV2", "PRISMModelV2",
    "FourierFeatureHead",
    "create_param_matched_prism",
    "TrigHead", "TrigTwinLayerV2", "TrigTwinModelV2",
    "create_param_matched_trig_twin", "create_param_matched_trigtwin", "PeriodicEGNN",
]
