"""PyTorch Fokker–Planck + BAOAB sampling (slim)."""

from .collective_variable import ParticlePairDistance, Torsion
from .integrator import BaoabFokkerPlanckIntegrator, SamplerState
from .kernel import GaussianKernel, PerReplicaRescaleConfig
from .potential import (
    GaffPotential,
    MACE_MODELS,
    MacePotential,
    ParticleDimer,
    load_mace_torch_sim_model,
)
from .system import EnergySystem

__all__ = [
    "BaoabFokkerPlanckIntegrator",
    "EnergySystem",
    "GaffPotential",
    "GaussianKernel",
    "MACE_MODELS",
    "MacePotential",
    "ParticleDimer",
    "ParticlePairDistance",
    "PerReplicaRescaleConfig",
    "SamplerState",
    "Torsion",
    "load_mace_torch_sim_model",
]

try:
    from .integrator import FPTBias, TorchSimFokkerPlanckIntegrator

    __all__ += ["FPTBias", "TorchSimFokkerPlanckIntegrator"]
except ImportError:  # pragma: no cover
    pass

__version__ = "0.1.0"
