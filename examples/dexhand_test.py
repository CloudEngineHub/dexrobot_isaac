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
    
    if env.num_actions == 0:
        print("ERROR: num_actions is 0! Environment is improperly initialized.")
        return

    # Initialize actions tensor
    actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    
    print("Starting simulation loop...")
    for i in range(args.steps):
        if args.debug:
            print(f"Step {i} - setting up actions")

        # Reset action tensor to zeros each step
        actions[:] = 0.0
        
        # Cycle through DOFs, moving one at a time
        dof_index = i % env.num_actions
        actions[:, dof_index] = args.movement_speed
        
        dof_name = None
        if env.control_hand_base and dof_index < env.NUM_BASE_DOFS:
            base_names = ["ARTx", "ARTy", "ARTz", "ARRx", "ARRy", "ARRz"]
            dof_name = base_names[dof_index]
        elif env.control_fingers and dof_index >= env.NUM_BASE_DOFS:
            finger_idx = dof_index - env.NUM_BASE_DOFS
            if hasattr(env, 'active_joint_names') and finger_idx < len(env.active_joint_names):
                dof_name = env.active_joint_names[finger_idx]
        
        if args.debug:
            print(f"  Moving DOF {dof_index}: {dof_name} with value {args.movement_speed}")
            
            # Print current hand position if available
            if hasattr(env, 'hand_pos'):
                print(f"  Hand position: {env.hand_pos[0].cpu().numpy()}")
                
            # Print current DOF positions if available
            if hasattr(env, 'dof_pos') and dof_index < env.dof_pos.shape[1]:
                print(f"  Current DOF value: {env.dof_pos[0, dof_index].item():.4f}")

        # Step the environment
        try:
            obs, rewards, dones, info = env.step(actions)
            
            if args.debug:
                print(f"  Step complete, rewards: {rewards.mean().item():.4f}")
        except Exception as e:
            print(f"Error during step: {e}")
            import traceback
            traceback.print_exc()
            break

        # Sleep between steps
        time.sleep(args.sleep)

    print("Test completed")

if __name__ == "__main__":
    main()