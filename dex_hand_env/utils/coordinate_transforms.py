"""
Coordinate transformation utilities for DexHand environment.

This module provides utilities for transforming points and orientations
between different coordinate frames.
"""

# Import IsaacGym first
from isaacgym.torch_utils import quat_mul, quat_conjugate

# Then import PyTorch
import torch


def point_in_hand_frame(pos_world, hand_pos, hand_rot):
    """
    Convert a point from world frame to hand frame.
    
    Args:
        pos_world: Position in world frame [batch_size, 3]
        hand_pos: Hand position in world frame [batch_size, 3]
        hand_rot: Hand rotation quaternion [batch_size, 4]
        
    Returns:
        Position in hand frame [batch_size, 3]
    """
    # Vector from hand to point in world frame
    rel_pos = pos_world - hand_pos
    
    # Create pure quaternion from the relative position (0, x, y, z)
    rel_pos_quat = torch.cat([torch.zeros_like(rel_pos[:, :1]), rel_pos], dim=-1)
    
    # Get conjugate of the hand rotation quaternion
    hand_rot_conj = quat_conjugate(hand_rot)
    
    # Apply quaternion rotation: q_conj * p * q
    # This rotates the point from world frame to hand frame
    rot_quat = quat_mul(quat_mul(hand_rot_conj, rel_pos_quat), hand_rot)
    
    # Extract the vector part (x, y, z)
    local_pos = rot_quat[:, 1:4]
    
    # Verify shapes
    assert local_pos.shape == rel_pos.shape, f"Output shape mismatch: {local_pos.shape} vs {rel_pos.shape}"
    
    return local_pos


def point_in_world_frame(pos_local, hand_pos, hand_rot):
    """
    Convert a point from hand frame to world frame.
    
    Args:
        pos_local: Position in hand frame [batch_size, 3]
        hand_pos: Hand position in world frame [batch_size, 3]
        hand_rot: Hand rotation quaternion [batch_size, 4]
        
    Returns:
        Position in world frame [batch_size, 3]
    """
    # Create pure quaternion from the local position (0, x, y, z)
    local_pos_quat = torch.cat([torch.zeros_like(pos_local[:, :1]), pos_local], dim=-1)
    
    # Get conjugate of the hand rotation quaternion
    hand_rot_conj = quat_conjugate(hand_rot)
    
    # Apply inverse quaternion rotation: q * p * q_conj
    # This rotates the point from hand frame to world frame
    rot_quat = quat_mul(quat_mul(hand_rot, local_pos_quat), hand_rot_conj)
    
    # Extract the vector part (x, y, z) and add the hand position
    world_pos = rot_quat[:, 1:4] + hand_pos
    
    # Verify shapes
    assert world_pos.shape == pos_local.shape, f"Output shape mismatch: {world_pos.shape} vs {pos_local.shape}"
    
    return world_pos