"""
Coordinate transformation utilities for DexHand environment.

This module provides utilities for transforming points and orientations
between different coordinate frames.
"""

# Import IsaacGym first
from isaacgym.torch_utils import quat_mul, quat_conjugate, quat_rotate, quat_rotate_inverse

# Then import PyTorch
import torch


def point_in_hand_frame(pos_world, hand_pos, hand_rot):
    """
    Convert a point from world frame to hand frame.

    Args:
        pos_world: Position in world frame [batch_size, 3]
        hand_pos: Hand position in world frame [batch_size, 3]
        hand_rot: Hand rotation quaternion [batch_size, 4] in format [x, y, z, w]

    Returns:
        Position in hand frame [batch_size, 3]
    """
    # Vector from hand to point in world frame
    rel_pos = pos_world - hand_pos

    # Use Isaac Gym's optimized quat_rotate_inverse to transform from world to hand frame
    local_pos = quat_rotate_inverse(hand_rot, rel_pos)

    return local_pos


def point_in_world_frame(pos_local, hand_pos, hand_rot):
    """
    Convert a point from hand frame to world frame.

    Args:
        pos_local: Position in hand frame [batch_size, 3]
        hand_pos: Hand position in world frame [batch_size, 3]
        hand_rot: Hand rotation quaternion [batch_size, 4] in format [x, y, z, w]

    Returns:
        Position in world frame [batch_size, 3]
    """
    # Use Isaac Gym's optimized quat_rotate to transform from hand to world frame
    rotated_pos = quat_rotate(hand_rot, pos_local)

    # Add hand position to get world position
    world_pos = rotated_pos + hand_pos

    return world_pos