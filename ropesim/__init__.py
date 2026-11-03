"""ropesim — GPU Cosserat rope solver (YarnBall, de-collisioned) for Isaac Lab RL.

- RopeSim:   single rope, host-driven root. Validation / experimentation API
             (the physics test suite runs against it).
- RopeBatch: N ropes in ONE solver. GPU-resident driving (root pos/vel/quat
             from CUDA tensors), contiguous observation buffers exposed
             zero-copy via __cuda_array_interface__ (torch.as_tensor(view)
             shares memory). Keep torch on its default stream (legacy-stream
             ordering is what synchronizes solver and consumer), and keep the
             RopeBatch object alive while views are in use.

Units everywhere: density=rho [kg/m], k_stretch=EA [N], k_bend=EI [N*m^2] (>0).
"""
from ._ropesim import CudaView, RopeBatch, RopeSim  # noqa: F401

__all__ = ["RopeSim", "RopeBatch", "CudaView"]
