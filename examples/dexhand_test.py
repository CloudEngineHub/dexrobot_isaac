#!/usr/bin/env python
"""
Test script for DexHand environment using the factory.

This script creates a DexHand environment with the BaseTask
and runs a simple test to verify camera controls and visualization.
"""

import os
import sys
import time
import numpy as np
import glob

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import IsaacGym first
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import to_torch

# Then import PyTorch
import torch

# Import factory
from dex_hand_env.factory import create_dex_env

def check_assets():
    """Check if required assets exist and print diagnostic information."""
    print("\n===== ASSET VERIFICATION =====")

    # Check asset root
    root_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
    print(f"Asset root directory: {root_dir}")
    print(f"Directory exists: {os.path.exists(root_dir)}")

    # Check MJCF model path
    model_path = os.path.join(root_dir, "dexrobot_mujoco/dexrobot_mujoco/models")
    print(f"\nModel directory: {model_path}")
    print(f"Directory exists: {os.path.exists(model_path)}")

    # Check specific model file
    model_file = os.path.join(model_path, "dexhand021_right_simplified_floating.xml")
    print(f"\nHand model file: {model_file}")
    print(f"File exists: {os.path.exists(model_file)}")

    if os.path.exists(model_file):
        print(f"File size: {os.path.getsize(model_file)} bytes")

        # Check file content
        with open(model_file, 'r') as f:
            content = f.read()
            print(f"File content length: {len(content)} characters")
            print(f"First 100 characters: {content[:100]}...")

    # List all XML files in the models directory
    print("\nAll XML files in models directory:")
    if os.path.exists(model_path):
        xml_files = glob.glob(os.path.join(model_path, "*.xml"))
        for xml_file in xml_files:
            print(f" - {os.path.basename(xml_file)}")

    # Check mesh directory
    mesh_path = os.path.join(root_dir, "dexrobot_mujoco/dexrobot_mujoco/meshes")
    print(f"\nMesh directory: {mesh_path}")
    print(f"Directory exists: {os.path.exists(mesh_path)}")

    if os.path.exists(mesh_path):
        subdirs = [d for d in os.listdir(mesh_path) if os.path.isdir(os.path.join(mesh_path, d))]
        print(f"Subdirectories: {subdirs}")

        for subdir in subdirs:
            subdir_path = os.path.join(mesh_path, subdir)
            mesh_files = glob.glob(os.path.join(subdir_path, "*.STL"))
            print(f"Found {len(mesh_files)} STL files in {subdir}")

            if len(mesh_files) > 0:
                print(f"Example files: {[os.path.basename(f) for f in mesh_files[:3]]}...")

    print("==============================\n")


def main():
    """Main function to test the DexHand environment."""
    print("Starting DexHand test script...")

    # First verify all assets are available
    check_assets()

    # Create a simple configuration manually
    print("Creating configuration...")
    cfg = {
        "env": {
            "numEnvs": 1,
            "episodeLength": 300,
            "controlMode": "position_delta",
            "controlHandBase": True,
            "controlFingers": True,
            "maxFingerVelocity": 2.0,
            "maxBaseLinearVelocity": 1.0,
            "maxBaseAngularVelocity": 1.5,
            "baseStiffness": 400.0,
            "baseDamping": 40.0,
            "fingerStiffness": 100.0,
            "fingerDamping": 10.0,
            "initialHandPos": [0.0, 0.0, 0.5],
            "initialHandRot": [0.0, 0.0, 0.0, 1.0],
        },
        "sim": {
            "dt": 0.01,
            "substeps": 2,
            "gravity": [0.0, 0.0, -9.81],
            "up_axis": "z",
            # Disable GPU pipeline to avoid CUDA errors
            "use_gpu_pipeline": False,
            "physx": {
                "num_threads": 4,
                "solver_type": 1,
                "num_position_iterations": 8,
                "num_velocity_iterations": 0,
                "contact_offset": 0.005,
                "rest_offset": 0.0,
                "bounce_threshold_velocity": 0.2,
                "max_depenetration_velocity": 1000.0,
                "default_buffer_size_multiplier": 5.0,
                "always_use_articulations": True
            }
        },
        "task": {
            "randomize": False
        },
        "physics_engine": "physx"
    }
    print("Configuration created.")

    # Create the environment
    print("Creating environment...")
    try:
        print("\n===== DETAILED INITIALIZATION SEQUENCE =====")
        env = create_dex_env(
            task_name="Base",
            cfg=cfg,
            rl_device="cuda:0",
            sim_device="cuda:0",
            graphics_device_id=0,
            headless=False,  # Set to False to show visualization
            force_render=True,
            virtual_screen_capture=False
        )
        print("===== INITIALIZATION COMPLETE =====\n")
        print("Environment created successfully!")
    except Exception as e:
        print(f"Error creating environment: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"Environment created with {env.num_envs} environments")
    print(f"Observation space: {env.num_observations}")
    print(f"Action space: {env.num_actions}")
    print(f"Control mode: {env.action_control_mode}")
    print(f"Control hand base: {env.control_hand_base}")
    print(f"Control fingers: {env.control_fingers}")
    print(f"Number of active joints: {env.NUM_ACTIVE_FINGER_DOFS}")
    print(f"Number of base DOFs: {env.NUM_BASE_DOFS}")

    if env.num_actions == 0:
        print("ERROR: num_actions is 0! Environment is improperly initialized.")
        return

    # Define actions
    actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    print(f"Created actions tensor with shape {actions.shape}")

    # Simple sinusoidal movement for testing
    phase = 0.0

    print("Starting simulation loop...")

    # Run the simulation
    for i in range(1000):  # Very reduced iterations for testing
        print(f"Step {i} - setting up actions")

        # Create an action tensor that applies a small upward force
        actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
        
        # If the control_hand_base is True, the first 6 dimensions will be the base DOFs
        if env.control_hand_base:
            # Print current position first (for all steps)
            if hasattr(env, 'hand_pos'):
                hand_z_pos = env.hand_pos[0, 2].item() if hasattr(env, 'hand_pos') else "Unknown"
                dof_z_pos = env.dof_pos[0, 2].item() if hasattr(env, 'dof_pos') else "Unknown"
                print(f"  Current positions: Hand Z: {hand_z_pos:.4f}, DOF Z: {dof_z_pos}")
                
                # Also print target if available
                if hasattr(env, 'current_targets'):
                    target_z = env.current_targets[0, 2].item()
                    print(f"  Current target Z: {target_z:.4f}")
            
            # With the hand base now fixed to the world, we can test each DOF
            # movement individually without worrying about gravity effects
            step_in_cycle = i % 600
            
            # Reset all actions to 0
            actions[:] = 0.0
            
            # In each 100-step phase, move a different DOF
            if step_in_cycle < 100:
                # Try ARTx - first DOF - should move along X axis
                actions[:, 0] = 0.05
                print(f"  Testing DOF 0 (ARTx) (action={actions[0, 0]:.4f})")
            elif step_in_cycle < 200:
                # Try ARTy - second DOF - should move along Y axis
                actions[:, 1] = 0.05
                print(f"  Testing DOF 1 (ARTy) (action={actions[0, 1]:.4f})")
            elif step_in_cycle < 300:
                # Try ARTz - third DOF - should move along Z axis (up/down)
                actions[:, 2] = 0.05
                print(f"  Testing DOF 2 (ARTz) (action={actions[0, 2]:.4f})")
            elif step_in_cycle < 400:
                # Try ARRx - fourth DOF - should rotate around X axis
                actions[:, 3] = 0.05
                print(f"  Testing DOF 3 (ARRx) (action={actions[0, 3]:.4f})")
            elif step_in_cycle < 500:
                # Try ARRy - fifth DOF - should rotate around Y axis
                actions[:, 4] = 0.05
                print(f"  Testing DOF 4 (ARRy) (action={actions[0, 4]:.4f})")
            else:
                # Try ARRz - sixth DOF - should rotate around Z axis
                actions[:, 5] = 0.05
                print(f"  Testing DOF 5 (ARRz) (action={actions[0, 5]:.4f})")
            
            # Print current position info from all sources
            print(f"  Current DOF positions:")
            for dof_idx in range(min(6, env.dof_pos.shape[1])):
                print(f"    DOF {dof_idx}: {env.dof_pos[0, dof_idx]:.4f}")
                
            # Print target info if available
            if hasattr(env, 'prev_active_targets'):
                print(f"  Current targets:")
                for dof_idx in range(min(6, env.prev_active_targets.shape[1])):
                    print(f"    DOF {dof_idx}: {env.prev_active_targets[0, dof_idx]:.4f}")

        # Step the environment with explicit try/except
        print(f"  Stepping environment")
        try:
            # DEBUGGING: If this is step 10, directly set DOF positions to test actuation
            if i == 10:
                print("\n===== MANUALLY SETTING DOF POSITIONS FOR TESTING =====")
                # Create a tensor with the target DOF positions
                dof_positions = env.dof_pos.clone()
                # Set ARTz (vertical position) to 1.0 - higher than initial position
                dof_positions[:, 2] = 1.0
                
                # Get DOF state tensor with positions and velocities
                dof_state = env.dof_state.clone()
                # Set positions (index 0)
                dof_state.view(env.num_envs, -1, 2)[:, :, 0] = dof_positions
                # Set velocities to zero (index 1)
                dof_state.view(env.num_envs, -1, 2)[:, :, 1] = 0
                
                # Set the DOF state directly
                env.gym.set_dof_state_tensor(
                    env.sim,
                    gymtorch.unwrap_tensor(dof_state)
                )
                print("Manually set ARTz DOF position to 1.0")
                print("================================================\n")
            
            # Regular step
            obs, rewards, dones, info = env.step(actions)
            print(f"  Step {i} complete, rewards: {rewards.mean().item()}")
        except Exception as e:
            print(f"  Error during step: {e}")
            import traceback
            traceback.print_exc()
            break

        print(f"  Sleeping between steps")
        time.sleep(0.5)  # Longer sleep to observe any visuals

    print("Test completed - check if visualization window is showing keyboard controls working")


if __name__ == "__main__":
    main()
