"""
Torch JIT utilities for DexHand environment.

This module provides JIT-compiled utility functions for vector and quaternion operations.
"""

# Import IsaacGym first

# Then import PyTorch
import torch
import torch.nn.functional as F
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
        torch.sign(sinp) * torch.tensor(np.pi / 2, device=q.device, dtype=q.dtype),
        torch.asin(sinp),
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


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    subgradient is zero where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    mat = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return mat.reshape(quaternions.shape[:-1] + (3, 3))


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.
    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


@torch.jit.script
def rotation_matrix_from_vectors(
    forward: torch.Tensor, up: torch.Tensor
) -> torch.Tensor:
    """
    Create a rotation matrix from forward and up vectors.

    Args:
        forward: Forward direction vector, shape (..., 3)
        up: Up direction vector, shape (..., 3)

    Returns:
        Rotation matrix of shape (..., 3, 3)
    """
    # Normalize forward vector
    forward = forward / torch.norm(forward, dim=-1, keepdim=True).clamp(min=1e-6)

    # Compute right vector (cross product of forward and up)
    right = torch.cross(forward, up, dim=-1)
    right = right / torch.norm(right, dim=-1, keepdim=True).clamp(min=1e-6)

    # Recompute up vector (cross product of right and forward)
    up = torch.cross(right, forward, dim=-1)
    up = up / torch.norm(up, dim=-1, keepdim=True).clamp(min=1e-6)

    # Create rotation matrix [right, up, forward] as columns
    rotation_matrix = torch.stack([right, up, forward], dim=-1)

    return rotation_matrix


def lookat_quaternion(
    cam_pos: torch.Tensor, target_pos: torch.Tensor, up: torch.Tensor = None
) -> torch.Tensor:
    """
    Create a quaternion that represents a camera looking at a target.

    Args:
        cam_pos: Camera position, shape (..., 3)
        target_pos: Target position to look at, shape (..., 3)
        up: Up vector (default: [0, 0, 1] for Z-up), shape (..., 3)

    Returns:
        Quaternion (x, y, z, w) of shape (..., 4)
    """
    if up is None:
        # Default Z-up vector
        batch_shape = cam_pos.shape[:-1]
        up = torch.zeros(*batch_shape, 3, device=cam_pos.device, dtype=cam_pos.dtype)
        up[..., 2] = 1.0

    # Calculate forward vector (from camera to target)
    forward = target_pos - cam_pos
    forward = forward / torch.norm(forward, dim=-1, keepdim=True).clamp(min=1e-6)

    # Create rotation matrix from vectors
    rotation_matrix = rotation_matrix_from_vectors(forward, up)

    # Convert rotation matrix to quaternion
    quaternion = matrix_to_quaternion(rotation_matrix)

    return quaternion
