"""Shared geometry primitives, independent of policy and dataset contracts.

Import ``numpy_ops`` for data preparation (no torch dependency), or ``torch_ops``
for differentiable training operations. Poses map local coordinates to world;
quaternions use xyzw. Optical cameras use +Z metric depth.
"""
