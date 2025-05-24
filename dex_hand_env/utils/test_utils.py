"""
Test utilities for DexHand environment.

This module provides utilities for testing the DexHand environment components.
"""

import torch
import numpy as np
from isaacgym.torch_utils import quat_from_euler_xyz, quat_mul, quat_conjugate

from dex_hand.utils.coordinate_transforms import point_in_hand_frame, point_in_world_frame


def test_coordinate_transforms():
    """
    Test the coordinate transformation functions.
    
    This function tests the point_in_hand_frame and point_in_world_frame functions
    with various inputs to ensure they work correctly.
    
    Returns:
        Boolean indicating whether all tests passed
    """
    device = torch.device("cpu")
    batch_size = 3
    
    # Test 1: Identity transformation
    # Hand at origin, no rotation, point at (1,0,0)
    hand_pos = torch.zeros((batch_size, 3), dtype=torch.float32)
    hand_rot = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).repeat(batch_size, 1)  # Identity quaternion
    point_pos = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32).repeat(batch_size, 1)
    
    local_pos = point_in_hand_frame(point_pos, hand_pos, hand_rot)
    expected = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32).repeat(batch_size, 1)
    assert torch.allclose(local_pos, expected, atol=1e-6), f"Identity test failed: {local_pos} != {expected}"
    print("Test 1 (Identity): PASSED")
    
    # Test 2: Translation only
    # Hand at (1,1,1), no rotation, point at (2,1,1)
    hand_pos = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32).repeat(batch_size, 1)
    hand_rot = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).repeat(batch_size, 1)  # Identity quaternion
    point_pos = torch.tensor([[2.0, 1.0, 1.0]], dtype=torch.float32).repeat(batch_size, 1)
    
    local_pos = point_in_hand_frame(point_pos, hand_pos, hand_rot)
    expected = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32).repeat(batch_size, 1)
    assert torch.allclose(local_pos, expected, atol=1e-6), f"Translation test failed: {local_pos} != {expected}"
    print("Test 2 (Translation): PASSED")
    
    # Test 3: Rotation only (90 degrees around Z axis)
    # Hand at origin, rotated 90° around Z, point at (0,1,0)
    hand_pos = torch.zeros((batch_size, 3), dtype=torch.float32)
    # Quaternion for 90° rotation around Z axis: (cos(45°), 0, 0, sin(45°))
    hand_rot = torch.tensor([[0.7071, 0.0, 0.0, 0.7071]], dtype=torch.float32).repeat(batch_size, 1)
    point_pos = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32).repeat(batch_size, 1)
    
    local_pos = point_in_hand_frame(point_pos, hand_pos, hand_rot)
    # Expected: point (0,1,0) in the rotated frame becomes (-1,0,0)
    expected = torch.tensor([[-1.0, 0.0, 0.0]], dtype=torch.float32).repeat(batch_size, 1)
    assert torch.allclose(local_pos, expected, atol=1e-6), f"Rotation test failed: {local_pos} != {expected}"
    print("Test 3 (Rotation Z): PASSED")
    
    # Test 4: Both translation and rotation
    # Hand at (1,1,1), rotated 90° around Z, point at (1,2,1)
    hand_pos = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32).repeat(batch_size, 1)
    hand_rot = torch.tensor([[0.7071, 0.0, 0.0, 0.7071]], dtype=torch.float32).repeat(batch_size, 1)
    point_pos = torch.tensor([[1.0, 2.0, 1.0]], dtype=torch.float32).repeat(batch_size, 1)
    
    local_pos = point_in_hand_frame(point_pos, hand_pos, hand_rot)
    # Expected: relative point (0,1,0) in the rotated frame becomes (-1,0,0)
    expected = torch.tensor([[-1.0, 0.0, 0.0]], dtype=torch.float32).repeat(batch_size, 1)
    assert torch.allclose(local_pos, expected, atol=1e-6), f"Translation + Rotation test failed: {local_pos} != {expected}"
    print("Test 4 (Translation + Rotation): PASSED")
    
    # Test 5: Inverse transformation
    # Test if world_to_local followed by local_to_world returns the original point
    euler_angles = torch.tensor([[0.3, 0.4, 0.5]], dtype=torch.float32).repeat(batch_size, 1)  # Random rotations
    hand_rot = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
    hand_pos = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32).repeat(batch_size, 1)
    point_pos = torch.tensor([[4.0, 5.0, 6.0]], dtype=torch.float32).repeat(batch_size, 1)
    
    # Transform to local
    local_pos = point_in_hand_frame(point_pos, hand_pos, hand_rot)
    
    # Transform back to world
    reconstructed_world_pos = point_in_world_frame(local_pos, hand_pos, hand_rot)
    
    # Should be close to original point_pos
    assert torch.allclose(reconstructed_world_pos, point_pos, atol=1e-6), f"Inverse transform test failed: {reconstructed_world_pos} != {point_pos}"
    print("Test 5 (Inverse transformation): PASSED")
    
    print("All coordinate transform tests PASSED!")
    return True


if __name__ == "__main__":
    test_coordinate_transforms()