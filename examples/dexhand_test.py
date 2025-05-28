#!/usr/bin/env python
"""
Test script for DexHand environment using the factory.

This script creates a DexHand environment with the BaseTask
and runs a simple test to verify hand movement and control.
"""

import os
import sys
import time
import argparse
import yaml

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import IsaacGym first
from isaacgym import gymtorch

# Then import PyTorch
import torch

# Import factory
from dex_hand_env.factory import create_dex_env

def load_config(config_path=None):
    """Load config from YAML file or use default path."""
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "dex_hand_env/cfg/task/BaseTask.yaml"
        )
    
    print(f"Loading config from {config_path}")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        cfg_yaml = yaml.safe_load(f)
    
    # Convert to the expected structure
    cfg = {
        "env": cfg_yaml.get("env", {}),
        "sim": cfg_yaml.get("sim", {}),
        "task": cfg_yaml.get("task", {}),
        "physics_engine": "physx"
    }
    
    # Add required sim parameters that are not in YAML
    if "gravity" not in cfg["sim"]:
        cfg["sim"]["gravity"] = [0.0, 0.0, -9.81]
    
    if "up_axis" not in cfg["sim"]:
        cfg["sim"]["up_axis"] = "z"
    
    return cfg

def main():
    """Main function to test the DexHand environment."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test DexHand environment")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML file")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of environments")
    parser.add_argument("--episode-length", type=int, default=300, help="Maximum episode length")
    parser.add_argument("--headless", action="store_true", help="Run without visualization")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--steps", type=int, default=300, help="Number of steps to run")
    parser.add_argument("--movement-speed", type=float, default=0.05, help="Speed of DOF movement")
    parser.add_argument("--sleep", type=float, default=0.01, help="Sleep time between steps")
    parser.add_argument("--use-gpu-pipeline", action="store_true", help="Enable GPU pipeline for PhysX")
    parser.add_argument("--no-gpu-pipeline", action="store_true", help="Disable GPU pipeline for PhysX")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run simulation on (cuda:0 or cpu)")
    args = parser.parse_args()

    print("Starting DexHand test script...")

    # Load configuration
    cfg = load_config(args.config)
    
    # Override with command line arguments
    cfg["env"]["numEnvs"] = args.num_envs
    cfg["env"]["episodeLength"] = args.episode_length
    
    # Force absolute position control mode for DOF verification
    cfg["env"]["controlMode"] = "position"
    
    # Handle GPU pipeline setting
    if args.use_gpu_pipeline and args.no_gpu_pipeline:
        print("Warning: Both --use-gpu-pipeline and --no-gpu-pipeline specified. Using GPU pipeline.")
        cfg["sim"]["use_gpu_pipeline"] = True
    elif args.use_gpu_pipeline:
        print("Using GPU pipeline for PhysX")
        cfg["sim"]["use_gpu_pipeline"] = True
    elif args.no_gpu_pipeline:
        print("Disabling GPU pipeline for PhysX")
        cfg["sim"]["use_gpu_pipeline"] = False
    
    if args.debug:
        print("Configuration loaded:")
        print(yaml.dump(cfg))

    # Create the environment
    print("Creating environment...")
    
    # Set simulation device based on command line arguments
    sim_device = args.device
    rl_device = args.device
    
    # Use -1 for graphics_device_id if headless or using CPU
    graphics_device_id = -1 if args.headless or sim_device == "cpu" else 0
    
    # Add debug information if requested
    if args.debug:
        print(f"Simulation device: {sim_device}")
        print(f"RL device: {rl_device}")
        print(f"Graphics device ID: {graphics_device_id}")
        print(f"GPU pipeline: {cfg['sim'].get('use_gpu_pipeline', 'not specified')}")
        
        # Set physics debugging options
        if "physx" not in cfg["sim"]:
            cfg["sim"]["physx"] = {}
    
    # Create environment
    try:
        env = create_dex_env(
            task_name="Base",
            cfg=cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=args.headless,
            force_render=not args.headless,
            virtual_screen_capture=False
        )
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
    
    if env.num_actions != 18:
        print(f"ERROR: Expected 18 actions (6 base + 12 finger), got {env.num_actions}")
        return
        
    if env.action_control_mode != "position":
        print(f"ERROR: Expected position control mode, got {env.action_control_mode}")
        return

    # Initialize actions tensor
    actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    
    # Define expected DOF mapping for 18 actions (6 base + 12 finger)
    action_to_dof_map = [
        # Base DOFs (6)
        ("ARTx", "Translation X"),
        ("ARTy", "Translation Y"), 
        ("ARTz", "Translation Z"),
        ("ARRx", "Rotation X"),
        ("ARRy", "Rotation Y"),
        ("ARRz", "Rotation Z"),
        # Finger DOFs (12) - first 3 joints of each of the 4 main fingers
        ("r_f_joint1_1", "Thumb Joint 1"),
        ("r_f_joint1_2", "Thumb Joint 2"),
        ("r_f_joint1_3", "Thumb Joint 3"),
        ("r_f_joint2_1", "Index Joint 1"),
        ("r_f_joint2_2", "Index Joint 2"),
        ("r_f_joint2_3", "Index Joint 3"),
        ("r_f_joint3_1", "Middle Joint 1"),
        ("r_f_joint3_2", "Middle Joint 2"),
        ("r_f_joint3_3", "Middle Joint 3"),
        ("r_f_joint4_1", "Ring Joint 1"),
        ("r_f_joint4_2", "Ring Joint 2"),
        ("r_f_joint4_3", "Ring Joint 3"),
    ]
    
    print(f"\n===== ACTION TO DOF MAPPING VERIFICATION =====")
    print(f"Testing {len(action_to_dof_map)} actions in absolute position control mode")
    print("Each action will be applied individually with a small movement")
    for i, (dof_name, description) in enumerate(action_to_dof_map):
        print(f"  Action {i:2d}: {dof_name:<15} - {description}")
    print("=" * 50)
    
    # Get initial DOF positions for reference
    obs = env.reset()
    initial_dof_pos = env.dof_pos.clone() if hasattr(env, 'dof_pos') else None
    
    print(f"\nStarting action-to-DOF verification test...")
    print(f"Movement magnitude: {args.movement_speed}")
    print(f"Steps per action: 10")
    print(f"Total test duration: {18 * 10} steps")
    print("=" * 50)
    
    # Test with all actions at 0 to verify hand stays still
    print(f"\n>>> Testing with ALL ACTIONS = 0 (Hand should stay still)")
    print("This verifies that the fixed base physics configuration is working correctly.")
    
    # Reset to initial state
    env.reset()
    
    # Get initial DOF positions and hand position
    if hasattr(env, 'dof_pos'):
        initial_dof_pos = env.dof_pos[0].clone()
        print(f"Initial DOF positions (first 6): {initial_dof_pos[:6].cpu().numpy()}")
        print(f"Initial finger DOF positions:")
        finger_dofs = ["r_f_joint1_1", "r_f_joint1_2", "r_f_joint1_3", "r_f_joint1_4",
                      "r_f_joint2_1", "r_f_joint2_2", "r_f_joint2_3", "r_f_joint2_4", 
                      "r_f_joint3_1", "r_f_joint3_2", "r_f_joint3_3", "r_f_joint3_4",
                      "r_f_joint4_1", "r_f_joint4_2", "r_f_joint4_3", "r_f_joint4_4",
                      "r_f_joint5_1", "r_f_joint5_2", "r_f_joint5_3", "r_f_joint5_4"]
        for i, name in enumerate(finger_dofs):
            dof_idx = 6 + i  # finger DOFs start at index 6
            print(f"  {name}: {initial_dof_pos[dof_idx].item():.6f}")
            
        # Check if we have DOF properties to see default/rest positions
        if hasattr(env, 'action_processor') and hasattr(env.action_processor, 'dof_props'):
            print(f"\nDOF Properties shape: {env.action_processor.dof_props.shape}")
            print("DOF properties (stiffness, damping, friction, armature, min, max):")
            for i in range(min(26, env.action_processor.dof_props.shape[0])):
                props = env.action_processor.dof_props[i].cpu().numpy()
                joint_names = ["ARTx", "ARTy", "ARTz", "ARRx", "ARRy", "ARRz"] + [f"finger_{j}" for j in range(6, 26)]
                print(f"  {joint_names[i]:>12}: stiff={props[0]:6.1f}, damp={props[1]:5.1f}, min={props[4]:+6.3f}, max={props[5]:+6.3f}")
                
        # Also check if original DOF properties were extracted correctly
        if hasattr(env, 'dof_properties_from_asset'):
            print(f"\nOriginal DOF properties from asset: {env.dof_properties_from_asset is not None}")
            if env.dof_properties_from_asset is not None:
                print(f"Shape: {env.dof_properties_from_asset.shape if hasattr(env.dof_properties_from_asset, 'shape') else 'N/A'}")
                print(f"Type: {type(env.dof_properties_from_asset)}")
        
    if hasattr(env, 'hand_pos'):
        initial_hand_pos = env.hand_pos[0].clone()
        print(f"Initial hand position: {initial_hand_pos.cpu().numpy()}")
    
    # Apply zero actions for 50 steps and monitor stability
    for step in range(50):
        actions[:] = 0.0  # All actions are zero
        
        obs, rewards, dones, info = env.step(actions)
        env.render()
        
        # Check every 10 steps
        if step % 10 == 9 and hasattr(env, 'dof_pos'):
            current_dof_pos = env.dof_pos[0]
            dof_change = current_dof_pos - initial_dof_pos  # Keep signed changes
            max_dof_change = torch.max(torch.abs(dof_change)).item()
            
            # Print detailed info about base DOFs (ARTx, ARTy, ARTz, ARRx, ARRy, ARRz)
            print(f"  Step {step+1}:")
            print(f"    ARTx: {current_dof_pos[0].item():.6f} (change: {dof_change[0].item():+.6f})")
            print(f"    ARTy: {current_dof_pos[1].item():.6f} (change: {dof_change[1].item():+.6f})")
            print(f"    ARTz: {current_dof_pos[2].item():.6f} (change: {dof_change[2].item():+.6f}) ← Z position")
            print(f"    ARRx: {current_dof_pos[3].item():.6f} (change: {dof_change[3].item():+.6f})")
            print(f"    ARRy: {current_dof_pos[4].item():.6f} (change: {dof_change[4].item():+.6f})")
            print(f"    ARRz: {current_dof_pos[5].item():.6f} (change: {dof_change[5].item():+.6f})")
            print(f"    Max DOF change = {max_dof_change:.6f}")
            
            # Check if hand is staying still (very small changes are acceptable due to numerical precision)
            if max_dof_change < 1e-4:
                print(f"    ✓ GOOD: Hand is staying still (max change < 1e-4)")
            else:
                print(f"    ⚠ WARNING: Hand is moving when it should be still (max change = {max_dof_change:.6f})")
                # Find which DOF is changing the most
                max_change_idx = torch.argmax(torch.abs(dof_change)).item()
                
                # Use actual DOF names from verification output
                actual_dof_names = ["ARTx", "ARTy", "ARTz", "ARRx", "ARRy", "ARRz", 
                                  "r_f_joint1_1", "r_f_joint1_2", "r_f_joint1_3", "r_f_joint1_4",
                                  "r_f_joint2_1", "r_f_joint2_2", "r_f_joint2_3", "r_f_joint2_4", 
                                  "r_f_joint3_1", "r_f_joint3_2", "r_f_joint3_3", "r_f_joint3_4",
                                  "r_f_joint4_1", "r_f_joint4_2", "r_f_joint4_3", "r_f_joint4_4",
                                  "r_f_joint5_1", "r_f_joint5_2", "r_f_joint5_3", "r_f_joint5_4"]
                
                print(f"    Most changing DOF: {actual_dof_names[max_change_idx]} (index {max_change_idx})")
                print(f"    This DOF changed by: {dof_change[max_change_idx].item():+.6f}")
                
                # Print top 5 changing DOFs for better analysis
                abs_changes = torch.abs(dof_change)
                top_5_indices = torch.topk(abs_changes, k=min(5, len(abs_changes))).indices
                print(f"    Top 5 changing DOFs:")
                for i, idx in enumerate(top_5_indices):
                    print(f"      {i+1}. {actual_dof_names[idx]} (index {idx}): {dof_change[idx].item():+.6f}")
        
        time.sleep(args.sleep)
    
    print(f"\nPhysics stability test completed. The hand should have remained stationary.")

    print("Test completed")

if __name__ == "__main__":
    main()