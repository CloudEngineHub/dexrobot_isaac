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
    parser.add_argument("--episode-length", type=int, default=1200, help="Maximum episode length")
    parser.add_argument("--headless", action="store_true", help="Run without visualization")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--steps", type=int, default=1200, help="Number of steps to run")
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
    
    if env.num_actions != 12:
        print(f"ERROR: Expected 12 actions (12 finger controls), got {env.num_actions}")
        return
        
    if env.action_control_mode != "position":
        print(f"ERROR: Expected position control mode, got {env.action_control_mode}")
        return

    # Initialize actions tensor
    actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    
    # Define expected action mapping for 12 finger controls with coupling
    action_to_dof_map = [
        # Finger Controls (12) with coupling logic:
        ("thumb_spread", "Thumb Spread (→ r_f_joint1_1)"),
        ("thumb_mcp", "Thumb MCP (→ r_f_joint1_2)"),
        ("thumb_dip", "Thumb DIP (→ r_f_joint1_3+1_4 coupled)"),
        ("finger_spread", "Finger Spread (→ r_f_joint2_1+4_1+5_1, 5_1=2x)"),
        ("index_mcp", "Index MCP (→ r_f_joint2_2)"),
        ("index_dip", "Index DIP (→ r_f_joint2_3+2_4 coupled)"),
        ("middle_mcp", "Middle MCP (→ r_f_joint3_2)"),
        ("middle_dip", "Middle DIP (→ r_f_joint3_3+3_4 coupled)"),
        ("ring_mcp", "Ring MCP (→ r_f_joint4_2)"),
        ("ring_dip", "Ring DIP (→ r_f_joint4_3+4_4 coupled)"),
        ("pinky_mcp", "Pinky MCP (→ r_f_joint5_2)"),
        ("pinky_dip", "Pinky DIP (→ r_f_joint5_3+5_4 coupled)"),
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
    print(f"Steps per action: 100")
    print(f"Total test duration: {12 * 100} steps")
    print("=" * 50)
    
    # Sequential action testing: each action goes from -1 → 1 → -1 over 100 steps
    print(f"\n>>> Testing Sequential Action Movement")
    print("Each action will move from -1 → 1 → -1 over 100 steps")
    print("Starting with all actions at -1, then testing each action individually")
    
    # Reset to initial state
    env.reset()
    
    # Get initial DOF positions for reference
    if hasattr(env, 'dof_pos'):
        initial_dof_pos = env.dof_pos[0].clone()
        print(f"Initial finger DOF positions:")
        finger_dofs = ["r_f_joint1_1", "r_f_joint1_2", "r_f_joint1_3", "r_f_joint1_4",
                      "r_f_joint2_1", "r_f_joint2_2", "r_f_joint2_3", "r_f_joint2_4", 
                      "r_f_joint3_1", "r_f_joint3_2", "r_f_joint3_3", "r_f_joint3_4",
                      "r_f_joint4_1", "r_f_joint4_2", "r_f_joint4_3", "r_f_joint4_4",
                      "r_f_joint5_1", "r_f_joint5_2", "r_f_joint5_3", "r_f_joint5_4"]
        for i, name in enumerate(finger_dofs):
            dof_idx = 6 + i  # finger DOFs start at index 6
            print(f"  {name}: {initial_dof_pos[dof_idx].item():.6f}")
    
    # Total steps: 12 actions × 100 steps each = 1200 steps
    total_steps = 12 * 100
    steps_per_action = 100
    
    print(f"\nStarting sequential action test:")
    print(f"Total steps: {total_steps}")
    print(f"Steps per action: {steps_per_action}")
    print("=" * 60)
    
    for step in range(total_steps):
        # Initialize all actions to -1
        actions[:] = -1.0
        
        # Determine which action to move
        current_action_idx = step // steps_per_action
        step_in_action = step % steps_per_action
        
        # Create a triangular wave: -1 → 1 → -1 over 100 steps
        # Steps 0-49: -1 → 1 (linear increase)
        # Steps 50-99: 1 → -1 (linear decrease)
        if step_in_action < 50:
            # -1 → 1 over first 50 steps
            progress = step_in_action / 49.0  # 0 to 1
            action_value = -1.0 + 2.0 * progress  # -1 to 1
        else:
            # 1 → -1 over last 50 steps
            progress = (step_in_action - 50) / 49.0  # 0 to 1
            action_value = 1.0 - 2.0 * progress  # 1 to -1
        
        # Apply the action value to the current action
        if current_action_idx < 12:
            actions[0, current_action_idx] = action_value
        
        # Step the simulation
        obs, rewards, dones, info = env.step(actions)
        env.render()
        
        # Print progress every 25 steps and at key transitions
        if (step_in_action % 25 == 0 or step_in_action == 49 or step_in_action == 99):
            action_name = action_to_dof_map[current_action_idx][0] if current_action_idx < 12 else "completed"
            print(f"  Step {step+1:4d}: Action {current_action_idx} ({action_name:>13}) = {action_value:+6.3f} (substep {step_in_action+1:2d}/100)")
            
            # At transitions, show DOF changes for the current action
            if hasattr(env, 'dof_pos') and (step_in_action == 0 or step_in_action == 49 or step_in_action == 99):
                current_dof_pos = env.dof_pos[0]
                dof_change = current_dof_pos - initial_dof_pos
                
                # Show changes for the joints controlled by this action
                if current_action_idx < 12:
                    action_name, description = action_to_dof_map[current_action_idx]
                    print(f"    {description}")
                    
                    # Show finger DOF changes
                    finger_changes = dof_change[6:]  # Skip base DOFs
                    max_finger_change = torch.max(torch.abs(finger_changes)).item()
                    
                    # Find the most changed finger DOF
                    if max_finger_change > 1e-6:
                        max_finger_idx = torch.argmax(torch.abs(finger_changes)).item()
                        finger_dof_name = finger_dofs[max_finger_idx]
                        finger_change_value = finger_changes[max_finger_idx].item()
                        print(f"    Max finger DOF change: {finger_dof_name} = {finger_change_value:+.6f}")
                    
                    # Show coupling verification for specific actions
                    if current_action_idx == 2:  # thumb_dip (should affect joints 1_3 and 1_4)
                        joint1_3_change = finger_changes[2].item()  # r_f_joint1_3 (index 8-6=2)
                        joint1_4_change = finger_changes[3].item()  # r_f_joint1_4 (index 9-6=3)
                        print(f"    Coupling check: r_f_joint1_3={joint1_3_change:+.6f}, r_f_joint1_4={joint1_4_change:+.6f}")
                        if abs(joint1_3_change - joint1_4_change) < 1e-5:
                            print(f"    ✓ Coupling verified: both joints move together")
                        else:
                            print(f"    ⚠ Coupling issue: joints should move together")
                    
                    elif current_action_idx == 3:  # finger_spread (should affect 2_1, 4_1, 5_1 with 5_1=2x)
                        joint2_1_change = finger_changes[4].item()   # r_f_joint2_1 (index 10-6=4)
                        joint4_1_change = finger_changes[12].item()  # r_f_joint4_1 (index 18-6=12)
                        joint5_1_change = finger_changes[16].item()  # r_f_joint5_1 (index 22-6=16)
                        joint3_1_change = finger_changes[8].item()   # r_f_joint3_1 (index 14-6=8, should stay 0)
                        print(f"    Spread coupling: 2_1={joint2_1_change:+.6f}, 4_1={joint4_1_change:+.6f}, 5_1={joint5_1_change:+.6f}")
                        print(f"    Fixed joint 3_1: {joint3_1_change:+.6f} (should be 0)")
                        if abs(joint5_1_change - 2.0 * joint2_1_change) < 1e-5 and abs(joint4_1_change - joint2_1_change) < 1e-5:
                            print(f"    ✓ Spread coupling verified: 5_1 ≈ 2×(2_1,4_1)")
                        else:
                            print(f"    ⚠ Spread coupling issue")
        
        time.sleep(args.sleep)
    
    print(f"\nSequential action test completed!")
    print(f"All 12 actions have been tested with the -1 → 1 → -1 pattern.")
    
    # Final DOF state summary
    if hasattr(env, 'dof_pos'):
        final_dof_pos = env.dof_pos[0]
        final_dof_change = final_dof_pos - initial_dof_pos
        final_finger_changes = final_dof_change[6:]
        max_final_change = torch.max(torch.abs(final_finger_changes)).item()
        
        print(f"\nFinal finger DOF changes from initial state:")
        for i, name in enumerate(finger_dofs):
            change = final_finger_changes[i].item()
            print(f"  {name}: {change:+.6f}")
        
        print(f"Maximum final change: {max_final_change:.6f}")
        if max_final_change < 0.1:
            print("✓ Hand returned close to initial position (good)")
        else:
            print("⚠ Hand position significantly changed from initial")

    print("Test completed")

if __name__ == "__main__":
    main()