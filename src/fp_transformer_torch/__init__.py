"""PyTorch Fokker–Planck + BAOAB sampling (slim)."""

from .collective_variable import (
    CartesianComponent,
    MultiParticlePairDistance,
    PairDistanceDifference,
    ParticlePairDistance,
    SoftDistance,
    SoftDistanceS12,
    Torsion,
)
from .integrator import BaoabFokkerPlanckIntegrator, SamplerState
from .kernel import GaussianKernel, PerReplicaRescaleConfig
from .potential import (
    DoubleWell2D,
    DoubleWellParams,
    GaffPotential,
    GaffTorchSimModel,
    KCAL_PER_MOL_TO_EV,
    MACE_MODELS,
    MacePotential,
    MechanicalEmbeddingMLMM,
    MLMMTopology,
    ParticleDimer,
    ParticleDimerParams,
    ShardedMaceModel,
    load_mace_torch_sim_model,
    load_sharded_mace_torch_sim_model,
)
from .system import EnergySystem

__all__ = [
    "BaoabFokkerPlanckIntegrator",
    "CartesianComponent",
    "DoubleWell2D",
    "DoubleWellParams",
    "EnergySystem",
    "GaffPotential",
    "GaffTorchSimModel",
    "GaussianKernel",
    "KCAL_PER_MOL_TO_EV",
    "MACE_MODELS",
    "MacePotential",
    "MechanicalEmbeddingMLMM",
    "MLMMTopology",
    "MultiParticlePairDistance",
    "PairDistanceDifference",
    "ParticleDimer",
    "ParticleDimerParams",
    "ParticlePairDistance",
    "PerReplicaRescaleConfig",
    "SamplerState",
    "ShardedMaceModel",
    "SoftDistance",
    "SoftDistanceS12",
    "Torsion",
    "load_mace_torch_sim_model",
    "load_sharded_mace_torch_sim_model",
]

try:
    from .integrator import FPTBias, TorchSimFokkerPlanckIntegrator, ShardedBaoabFokkerPlanckIntegrator

    __all__ += [
        "FPTBias",
        "TorchSimFokkerPlanckIntegrator",
        "ShardedBaoabFokkerPlanckIntegrator",
    ]
except ImportError:  # pragma: no cover
    pass

try:
    from .integrator import (
        QuadraticDistanceLowerWall,
        QuadraticDistanceWall,
        SoftminDistanceUpperWall,
    )

    __all__ += [
        "QuadraticDistanceLowerWall",
        "QuadraticDistanceWall",
        "SoftminDistanceUpperWall",
    ]
except ImportError:  # pragma: no cover
    pass

try:
    from .integrator import (
        TorchSimUmbrellaSampling,
        TorchSimWelltemperMetadynamic,
        UmbrellaBias,
        WellTemperedMetadynamicsBias,
        make_umbrella_centers_1d,
        make_umbrella_centers_polyline,
    )

    __all__ += [
        "TorchSimUmbrellaSampling",
        "TorchSimWelltemperMetadynamic",
        "UmbrellaBias",
        "WellTemperedMetadynamicsBias",
        "make_umbrella_centers_1d",
        "make_umbrella_centers_polyline",
    ]
except ImportError:  # pragma: no cover
    pass

__version__ = "0.1.0"
