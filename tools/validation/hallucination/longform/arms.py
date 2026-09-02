"""Public facade for the three long-form synthesis arm adapters.

Implementation responsibilities are split across ``arm_types``,
``endpoint_arm``, and ``official_arm``.  This module intentionally keeps the
original import surface stable for runners and downstream validation tools.
"""

from .arm_types import (
    ArmAdapter,
    AudioChunkRecord,
    CollectedRun,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SPEAKER,
    stable_sampling_seed,
)
from .endpoint_arm import (
    EndpointArmAdapter,
    EngineGrpcArm,
    EngineGrpcArmAdapter,
    TritonGrpcArm,
    TritonGrpcArmAdapter,
)
from .official_arm import OfficialPyTorchArm, OfficialPyTorchArmAdapter


__all__ = [
    "ArmAdapter",
    "AudioChunkRecord",
    "CollectedRun",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_SPEAKER",
    "EndpointArmAdapter",
    "EngineGrpcArm",
    "EngineGrpcArmAdapter",
    "OfficialPyTorchArm",
    "OfficialPyTorchArmAdapter",
    "TritonGrpcArm",
    "TritonGrpcArmAdapter",
    "stable_sampling_seed",
]
