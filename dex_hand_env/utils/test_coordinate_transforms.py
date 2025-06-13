#!/usr/bin/env python
"""
Test the world-to-hand frame transformation used in observation encoding.

This tests the coordinate transformation functions to ensure fingertip/pad
poses are correctly transformed from world frame to hand frame.
"""

# Import IsaacGym first
from isaacgym.torch_utils import quat_from_euler_xyz, quat_mul, quat_conjugate

# Then import other modules
import torch
import numpy as np

# Import the transformation functions
from dex_hand_env.utils.coordinate_transforms import point_in_hand_frame, point_in_world_frame


def test_coordinate_transforms():
    """
    Test the coordinate transformation functions.
    
    This function tests the point_in_hand_frame and point_in_world_frame functions
    with various inputs to ensure they work correctly.
    
    Returns:
        Boolean indicating whether all tests passed
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 3
    
    print(f"Running coordinate transform tests on device: {device}")
    print("=" * 60)
    
    # Test 1: Identity transformation
    # Hand at origin, no rotation, point at (1,0,0)
    print("\nTest 1: Identity transformation")
    hand_pos = torch.zeros((batch_size, 3), dtype=torch.float32, device=device)
    hand_rot = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=device).repeat(batch_size, 1)  # Identity quaternion [x,y,z,w]
    point_pos = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32, device=device).repeat(batch_size, 1)
    
    local_pos = point_in_hand_frame(point_pos, hand_pos, hand_rot)
    expected = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32, device=device).repeat(batch_size, 1)
    assert torch.allclose(local_pos, expected, atol=1e-6), f"Identity test failed: {local_pos} != {expected}"
    print("✓ PASSED: Point at (1,0,0) with identity transform → (1,0,0)")
    
    # Test 2: Translation only
    # Hand at (1,1,1), no rotation, point at (2,1,1)
    print("\nTest 2: Translation only")
    hand_pos = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32, device=device).repeat(batch_size, 1)
    hand_rot = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=device).repeat(batch_size, 1)  # Identity quaternion
    point_pos = torch.tensor([[2.0, 1.0, 1.0]], dtype=torch.float32, device=device).repeat(batch_size, 1)
    
    local_pos = point_in_hand_frame(point_pos, hand_pos, hand_rot)
    expected = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32, device=device).repeat(batch_size, 1)
    assert torch.allclose(local_pos, expected, atol=1e-6), f"Translation test failed: {local_pos} != {expected}"
    print("✓ PASSED: Hand at (1,1,1), point at (2,1,1) → local (1,0,0)")
    
    # Test 3: Rotation only (90 degrees around Z axis)
    # Hand at origin, rotated 90° around Z, point at (0,1,0)
    print("\nTest 3: 90° rotation around Z axis")
    hand_pos = torch.zeros((batch_size, 3), dtype=torch.float32, device=device)
    # Quaternion for 90° rotation around Z axis: [x=0, y=0, z=sin(45°), w=cos(45°)]
    sqrt_half = 0.7071067811865476  # sqrt(0.5)
    hand_rot = torch.tensor([[0.0, 0.0, sqrt_half, sqrt_half]], dtype=torch.float32, device=device).repeat(batch_size, 1)
    point_pos = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32, device=device).repeat(batch_size, 1)
    
    local_pos = point_in_hand_frame(point_pos, hand_pos, hand_rot)
    # Expected: point (0,1,0) in world frame appears at (1,0,0) in hand frame
    expected = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32, device=device).repeat(batch_size, 1)
    assert torch.allclose(local_pos, expected, atol=1e-5), f"Rotation test failed: {local_pos} != {expected}"
    print("✓ PASSED: Point at (0,1,0) with 90° Z rotation → local (1,0,0)")
    
    # Test 4: Both translation and rotation
    # Hand at (1,1,1), rotated 90° around Z, point at (1,2,1)
    print("\nTest 4: Translation + Rotation")
    hand_pos = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32, device=device).repeat(batch_size, 1)
    hand_rot = torch.tensor([[0.0, 0.0, sqrt_half, sqrt_half]], dtype=torch.float32, device=device).repeat(batch_size, 1)
    point_pos = torch.tensor([[1.0, 2.0, 1.0]], dtype=torch.float32, device=device).repeat(batch_size, 1)
    
    local_pos = point_in_hand_frame(point_pos, hand_pos, hand_rot)
    # Expected: relative point (0,1,0) in world frame appears at (1,0,0) in hand frame
    expected = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32, device=device).repeat(batch_size, 1)
    assert torch.allclose(local_pos, expected, atol=1e-5), f"Translation + Rotation test failed: {local_pos} != {expected}"
    print("✓ PASSED: Hand at (1,1,1) rotated 90°, point at (1,2,1) → local (1,0,0)")
    
    # Test 5: 90° rotation around Y axis (like the floating hand built-in rotation)
    print("\nTest 5: 90° rotation around Y axis (floating hand case)")
    hand_pos = torch.zeros((batch_size, 3), dtype=torch.float32, device=device)
    # Quaternion for 90° rotation around Y axis: [x=0, y=sin(45°), z=0, w=cos(45°)]
    hand_rot = torch.tensor([[0.0, sqrt_half, 0.0, sqrt_half]], dtype=torch.float32, device=device).repeat(batch_size, 1)
    
    # Test several points
    test_points = [
        ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),   # X-axis point → +Z in hand frame
        ([0.0, 1.0, 0.0], [0.0, 1.0, 0.0]),   # Y-axis point → Y (unchanged)
        ([0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]),  # Z-axis point → -X in hand frame
    ]
    
    for world_point, expected_local in test_points:
        point_pos = torch.tensor([world_point], dtype=torch.float32, device=device).repeat(batch_size, 1)
        local_pos = point_in_hand_frame(point_pos, hand_pos, hand_rot)
        expected = torch.tensor([expected_local], dtype=torch.float32, device=device).repeat(batch_size, 1)
        assert torch.allclose(local_pos, expected, atol=1e-5), f"Y-rotation test failed for {world_point}: {local_pos[0].cpu().numpy()} != {expected_local}"
        print(f"  ✓ Point {world_point} → local {expected_local}")
    
    # Test 6: Inverse transformation
    # Test if world_to_local followed by local_to_world returns the original point
    print("\nTest 6: Inverse transformation (round-trip)")
    euler_angles = torch.tensor([[0.3, 0.4, 0.5]], dtype=torch.float32, device=device).repeat(batch_size, 1)  # Random rotations
    hand_rot = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
    hand_pos = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32, device=device).repeat(batch_size, 1)
    point_pos = torch.tensor([[4.0, 5.0, 6.0]], dtype=torch.float32, device=device).repeat(batch_size, 1)
    
    # Transform to local
    local_pos = point_in_hand_frame(point_pos, hand_pos, hand_rot)
    
    # Transform back to world
    reconstructed_world_pos = point_in_world_frame(local_pos, hand_pos, hand_rot)
    
    # Should be close to original point_pos
    assert torch.allclose(reconstructed_world_pos, point_pos, atol=1e-5), f"Inverse transform test failed: {reconstructed_world_pos} != {point_pos}"
    print("✓ PASSED: World → Hand → World transformation preserves position")
    
    # Test 7: Batch processing with different transformations
    print("\nTest 7: Batch processing with different transformations")
    # Create different poses for each environment in the batch
    hand_pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ], dtype=torch.float32, device=device)
    
    # Different rotations: identity, 90° around Y, 90° around Z
    hand_rot = torch.tensor([
        [0.0, 0.0, 0.0, 1.0],                 # Identity
        [0.0, sqrt_half, 0.0, sqrt_half],     # 90° around Y
        [0.0, 0.0, sqrt_half, sqrt_half]      # 90° around Z
    ], dtype=torch.float32, device=device)
    
    point_pos = torch.tensor([
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],  # Changed to (2,0,0) so relative position is (1,0,0)
        [0.0, 1.0, 0.0]
    ], dtype=torch.float32, device=device)
    
    local_pos = point_in_hand_frame(point_pos, hand_pos, hand_rot)
    
    expected = torch.tensor([
        [1.0, 0.0, 0.0],    # Identity: (1,0,0) → (1,0,0)
        [0.0, 0.0, 1.0],    # 90° Y rotation: (1,0,0) relative to (1,0,0) → (0,0,1)
        [1.0, 0.0, 0.0]     # 90° Z rotation: (0,1,0) → (1,0,0)
    ], dtype=torch.float32, device=device)
    
    assert torch.allclose(local_pos, expected, atol=1e-5), f"Batch test failed:\n{local_pos}\n!=\n{expected}"
    print("✓ PASSED: Batch processing with different transformations per environment")
    
    print("\n" + "=" * 60)
    print("All coordinate transform tests PASSED! ✓")
    return True


if __name__ == "__main__":
    try:
        test_coordinate_transforms()
    except Exception as e:
        print(f"\nTest FAILED with error: {e}")
        import traceback
        traceback.print_exc()