"""
Torch JIT utilities for DexHand environment.

This module provides JIT-compiled utility functions for vector and quaternion operations.
"""

# Import IsaacGym first
import isaacgym
from isaacgym.torch_utils import to_torch, tensor_clamp, quat_mul, quat_conjugate

# Then import PyTorch
import torch
import numpy as np


@torch.jit.script
def quat_to_euler(q):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw).

    Args:
        q: Quaternion of shape (..., 4)

    Returns:
        Tuple of (roll, pitch, yaw) each of shape (...)
    """
    qx, qy, qz, qw = 0, 1, 2, 3

    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[..., qw] * q[..., qx] + q[..., qy] * q[..., qz])
    cosr_cosp = 1.0 - 2.0 * (q[..., qx] * q[..., qx] + q[..., qy] * q[..., qy])
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[..., qw] * q[..., qy] - q[..., qz] * q[..., qx])
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        torch.sign(sinp) * torch.tensor(np.pi/2, device=q.device, dtype=q.dtype),
        torch.asin(sinp)
    )

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[..., qw] * q[..., qz] + q[..., qx] * q[..., qy])
    cosy_cosp = 1.0 - 2.0 * (q[..., qy] * q[..., qy] + q[..., qz] * q[..., qz])
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


@torch.jit.script
def quat_from_euler(roll, pitch, yaw):
    """
    Convert Euler angles to quaternion.

    Args:
        roll: Roll angle in radians
        pitch: Pitch angle in radians
        yaw: Yaw angle in radians

    Returns:
        Quaternion of shape (..., 4)
    """
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    qw = cy * cp * cr + sy * sp * sr
    qx = cy * cp * sr - sy * sp * cr
    qy = cy * sp * cr + sy * cp * sr
    qz = sy * cp * cr - cy * sp * sr

    return torch.stack([qx, qy, qz, qw], dim=-1)


@torch.jit.script
def axisangle2quat(vec: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Converts scaled axis-angle to quat.

    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    angle = torch.norm(vec, dim=-1, keepdim=True)

    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat(
        [
            vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
            torch.cos(angle[idx, :] / 2.0),
        ],
        dim=-1,
    )

    quat = quat.reshape(list(input_shape) + [4])
    return quat