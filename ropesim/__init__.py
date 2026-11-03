"""ropesim — GPU Cosserat rope solver (YarnBall, de-collisioned) for Isaac Lab RL.

Milestone 0: minimal wrap proving the pybind ↔ CUDA ↔ libYarnBall.a pipeline.
A single rope, host-side position readback. Batching, GPU pin-driving, and
zero-copy DLPack tensors come next (see INTEGRATION_PLAN.md).
"""
from ._ropesim import RopeSim  # noqa: F401

__all__ = ["RopeSim"]
