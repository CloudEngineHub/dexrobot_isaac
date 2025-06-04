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
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import IsaacGym first
from isaacgym import gymtorch

# Then import PyTorch
import torch

# Import factory
from dex_hand_env.factory import create_dex_env
import math

# Import Rerun for plotting (optional)
try:
    import rerun as rr
    RERUN_AVAILABLE = True
except ImportError:
    RERUN_AVAILABLE = False
    print("Rerun not available. Install with 'pip install rerun-sdk' for real-time plotting.")

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

def create_rule_based_base_controller():
    """
    Create a rule-based controller function for hand base.
    Returns a function that takes env and returns base targets.
    """
    def base_controller(env):
        """
        Generate rule-based control for hand base (6 DOFs: x, y, z, rx, ry, rz).
        Returns raw DOF targets in physically meaningful units (meters, radians).
        """
        base_targets = torch.zeros((env.num_envs, 6), device=env.device)

        # Get actual simulation time from environment
        # progress_buf counts control steps, not physics steps
        # control_dt = physics_dt * physics_steps_per_control_step
        control_dt = env.physics_manager.control_dt

        # Get simulation time in seconds
        sim_time = env.progress_buf[0].item() * control_dt

        # Use 2π for 1 second period for circular motion
        t = sim_time * 2.0 * math.pi  # 1 Hz base frequency

        # Circular motion (relative displacement in meters)
        base_targets[:, 0] = 0.1 * math.sin(t)      # ARTx (±10cm)
        base_targets[:, 1] = 0.1 * math.cos(t)      # ARTy (±10cm)
        base_targets[:, 2] = 0.1 * math.sin(t * 0.5)  # ARTz (±10cm, slower)

        # Rotation oscillations (30 degrees = 0.5236 radians)
        base_targets[:, 3] = 0.5236 * math.sin(t * 0.7)   # ARRx (±30 degrees)
        base_targets[:, 4] = 0.5236 * math.cos(t * 0.8)   # ARRy (±30 degrees)
        base_targets[:, 5] = 0.5236 * math.sin(t * 1.2)   # ARRz (±30 degrees)

        return base_targets

    return base_controller

def create_rule_based_finger_controller():
    """
    Create a rule-based controller function for fingers.
    Returns a function that takes env and returns finger targets.
    """
    # Create a mutable object to track state
    state = {'call_count': 0, 'start_time': None}

    def finger_controller(env):
        """
        Generate rule-based control for fingers (12 finger controls).
        Returns raw finger targets in physically meaningful units (radians).
        """
        finger_targets = torch.zeros((env.num_envs, 12), device=env.device)

        # Get actual simulation time from environment
        control_dt = env.physics_manager.control_dt

        # Calculate time based on progress_buf (control steps)
        sim_time = env.progress_buf[0].item() * control_dt

        # Use 2π for 0.5 second period for grasping (2 Hz)
        t = sim_time * 2.0 * math.pi * 2.0  # 2 Hz for faster finger motion

        # Debug: Print time info when progress_buf changes or first few calls
        last_progress = getattr(finger_controller, 'last_progress', -1)
        current_progress = env.progress_buf[0].item()

        if state['call_count'] < 5 or current_progress != last_progress:
            print(f"[Finger Controller] Call #{state['call_count']}, progress_buf: {current_progress}, sim_time: {sim_time:.6f}, t: {t:.6f}, sin(t): {math.sin(t):.6f}")

        finger_controller.last_progress = current_progress
        state['call_count'] += 1

        # Coordinated finger motion - simulate grasping
        grasp_wave = 0.5 * (1 + math.sin(t))  # 0 to 1 wave

        # Check if we have contact (can use env state)
        if hasattr(env, 'contact_forces'):
            # Sum contact forces across fingertips
            total_contact = env.contact_forces.sum(dim=(1, 2))  # Sum over fingers and force dims
            # If we have significant contact, increase grasp strength
            contact_detected = total_contact > 0.1
            grasp_scale = torch.where(contact_detected, 1.2, 1.0).unsqueeze(1)
        else:
            grasp_scale = 1.0

        # Thumb motion (actions 0-2) - physical joint ranges
        finger_targets[:, 0] = 0.2 * grasp_wave        # thumb_spread (0-0.2 rad)
        finger_targets[:, 1] = 1.0 * grasp_wave * grasp_scale.squeeze()  # thumb_mcp (0-1.0 rad)
        finger_targets[:, 2] = 1.2 * grasp_wave * grasp_scale.squeeze()  # thumb_dip (0-1.2 rad)

        # Finger spread (action 3) - physical range for spread joints
        finger_targets[:, 3] = 0.1 * grasp_wave        # finger_spread (0-0.1 rad)

        # Index finger (actions 4-5)
        finger_targets[:, 4] = 1.0 * grasp_wave * grasp_scale.squeeze()  # index_mcp (0-1.0 rad)
        finger_targets[:, 5] = 1.3 * grasp_wave * grasp_scale.squeeze()  # index_dip (0-1.3 rad)

        # Middle finger (actions 6-7)
        finger_targets[:, 6] = 1.0 * grasp_wave * grasp_scale.squeeze()  # middle_mcp (0-1.0 rad)
        finger_targets[:, 7] = 1.3 * grasp_wave * grasp_scale.squeeze()  # middle_dip (0-1.3 rad)

        # Ring finger (actions 8-9)
        finger_targets[:, 8] = 0.9 * grasp_wave * grasp_scale.squeeze()  # ring_mcp (0-0.9 rad)
        finger_targets[:, 9] = 1.2 * grasp_wave * grasp_scale.squeeze()  # ring_dip (0-1.2 rad)

        # Pinky finger (actions 10-11) - less motion
        finger_targets[:, 10] = 0.8 * grasp_wave * grasp_scale.squeeze()  # pinky_mcp (0-0.8 rad)
        finger_targets[:, 11] = 1.1 * grasp_wave * grasp_scale.squeeze()  # pinky_dip (0-1.1 rad)

        return finger_targets

    return finger_controller


def setup_rerun_logging():
    """Initialize Rerun logging."""
    if not RERUN_AVAILABLE:
        return False
    
    # Initialize the main Rerun application
    rr.init("dexhand_test", spawn=True)
    
    # Create 14 separate plots using different entity paths that will appear as separate plots
    # Use the root level and separate recording IDs for each plot type
    plot_configs = [
        ("Plot_1_ARTx", "ARTx position and target"),
        ("Plot_2_ART_vels", "ART velocities"), 
        ("Plot_3_ARRx", "ARRx position and target"),
        ("Plot_4_ARR_vels", "ARR velocities"),
        ("Plot_5_finger_DOFs", "Active finger DOFs"),
        ("Plot_6_hand_pos", "Hand position"),
        ("Plot_7_hand_quat", "Hand quaternion"),
        ("Plot_8_contact", "Contact forces"),
        ("Plot_9_actions", "Actions comparison"),
        ("Plot_10_world_pos", "World frame positions"),
        ("Plot_11_world_quat", "World frame quaternions"), 
        ("Plot_12_hand_pos", "Hand frame positions"),
        ("Plot_13_hand_quat", "Hand frame quaternions"),
        ("Plot_14_raw_DOFs", "Raw vs active DOFs")
    ]
    
    # Set up each plot with a clear structure
    for plot_name, description in plot_configs:
        # Log a text description for each plot
        rr.log(f"{plot_name}/description", rr.TextLog(description))
    
    return True


def log_observation_data(env, step, env_idx=0):
    """Log observation data to Rerun for visualization."""
    if not RERUN_AVAILABLE:
        return
    
    try:
        # Get observation dictionary for convenient access
        obs_dict = env.get_observations_dict()
        obs_encoder = env.observation_encoder
        
        # Plot 1: ARTx pos, ARTx target
        artx_pos = obs_encoder.get_base_dof_value("ARTx", "pos", obs_dict, env_idx)
        artx_target = obs_encoder.get_base_dof_value("ARTx", "target", obs_dict, env_idx)
        rr.log("Plot_1_ARTx/pos", rr.Scalar(float(artx_pos)))
        rr.log("Plot_1_ARTx/target", rr.Scalar(float(artx_target)))
        
        # Plot 2: ARTx vel, ARTy vel, ARTz vel
        artx_vel = obs_encoder.get_base_dof_value("ARTx", "vel", obs_dict, env_idx)
        arty_vel = obs_encoder.get_base_dof_value("ARTy", "vel", obs_dict, env_idx)
        artz_vel = obs_encoder.get_base_dof_value("ARTz", "vel", obs_dict, env_idx)
        rr.log("Plot_2_ART_vels/artx_vel", rr.Scalar(float(artx_vel)))
        rr.log("Plot_2_ART_vels/arty_vel", rr.Scalar(float(arty_vel)))
        rr.log("Plot_2_ART_vels/artz_vel", rr.Scalar(float(artz_vel)))
        
        # Plot 3: ARRx pos, ARRx target
        arrx_pos = obs_encoder.get_base_dof_value("ARRx", "pos", obs_dict, env_idx)
        arrx_target = obs_encoder.get_base_dof_value("ARRx", "target", obs_dict, env_idx)
        rr.log("Plot_3_ARRx/pos", rr.Scalar(float(arrx_pos)))
        rr.log("Plot_3_ARRx/target", rr.Scalar(float(arrx_target)))
        
        # Plot 4: ARRx vel, ARRy vel, ARRz vel
        arrx_vel = obs_encoder.get_base_dof_value("ARRx", "vel", obs_dict, env_idx)
        arry_vel = obs_encoder.get_base_dof_value("ARRy", "vel", obs_dict, env_idx)
        arrz_vel = obs_encoder.get_base_dof_value("ARRz", "vel", obs_dict, env_idx)
        rr.log("Plot_4_ARR_vels/arrx_vel", rr.Scalar(float(arrx_vel)))
        rr.log("Plot_4_ARR_vels/arry_vel", rr.Scalar(float(arry_vel)))
        rr.log("Plot_4_ARR_vels/arrz_vel", rr.Scalar(float(arrz_vel)))
        
        # Plot 5: (active finger dof) th_rot pos, th_rot target, mf_mcp pos, mf_mcp target
        th_rot_pos = obs_encoder.get_active_finger_dof_value("th_rot", "pos", obs_dict, env_idx)
        th_rot_target = obs_encoder.get_active_finger_dof_value("th_rot", "target", obs_dict, env_idx)
        mf_mcp_pos = obs_encoder.get_active_finger_dof_value("mf_mcp", "pos", obs_dict, env_idx)
        mf_mcp_target = obs_encoder.get_active_finger_dof_value("mf_mcp", "target", obs_dict, env_idx)
        rr.log("Plot_5_finger_DOFs/th_rot_pos", rr.Scalar(float(th_rot_pos)))
        rr.log("Plot_5_finger_DOFs/th_rot_target", rr.Scalar(float(th_rot_target)))
        rr.log("Plot_5_finger_DOFs/mf_mcp_pos", rr.Scalar(float(mf_mcp_pos)))
        rr.log("Plot_5_finger_DOFs/mf_mcp_target", rr.Scalar(float(mf_mcp_target)))
        
        # Plot 6: hand_pose x,y,z
        if "hand_pose" in obs_dict:
            hand_pose = obs_dict["hand_pose"][env_idx]
            rr.log("Plot_6_hand_pos/x", rr.Scalar(float(hand_pose[0])))
            rr.log("Plot_6_hand_pos/y", rr.Scalar(float(hand_pose[1])))
            rr.log("Plot_6_hand_pos/z", rr.Scalar(float(hand_pose[2])))
        
        # Plot 7: hand_pose quat w,x,y,z
        if "hand_pose" in obs_dict:
            hand_quat = obs_dict["hand_pose"][env_idx, 3:7]
            rr.log("Plot_7_hand_quat/w", rr.Scalar(float(hand_quat[3])))  # w is last in Isaac Gym
            rr.log("Plot_7_hand_quat/x", rr.Scalar(float(hand_quat[0])))
            rr.log("Plot_7_hand_quat/y", rr.Scalar(float(hand_quat[1])))
            rr.log("Plot_7_hand_quat/z", rr.Scalar(float(hand_quat[2])))
        
        # Plot 8: contact_force x,y,z component and magnitude for middle finger (index 2)
        if "contact_forces" in obs_dict:
            middle_finger_force = obs_encoder.get_contact_force_value(2, obs_dict, env_idx)  # Middle finger
            rr.log("Plot_8_contact/x", rr.Scalar(float(middle_finger_force[0])))
            rr.log("Plot_8_contact/y", rr.Scalar(float(middle_finger_force[1])))
            rr.log("Plot_8_contact/z", rr.Scalar(float(middle_finger_force[2])))
            magnitude = np.linalg.norm(middle_finger_force)
            rr.log("Plot_8_contact/magnitude", rr.Scalar(float(magnitude)))
        
        # Plot 9: unscaled previous action corresponding to ARTx, ARTx vel times control_dt
        if "prev_actions" in obs_dict and hasattr(env.action_processor, 'unscale_actions'):
            prev_actions = obs_dict["prev_actions"][env_idx:env_idx+1]  # Keep batch dimension
            unscaled_actions = env.action_processor.unscale_actions(prev_actions)
            if unscaled_actions.shape[1] > 0:
                unscaled_artx_action = float(unscaled_actions[0, 0])  # First action is ARTx
                artx_vel_times_dt = float(artx_vel * env.action_processor.control_dt) if env.action_processor.control_dt else 0.0
                rr.log("Plot_9_actions/unscaled_artx", rr.Scalar(unscaled_artx_action))
                rr.log("Plot_9_actions/artx_vel_times_dt", rr.Scalar(artx_vel_times_dt))
        
        # Plot 10: middle finger fingerpad and fingertip x,y,z in world frame
        mf_tip_world = obs_encoder.get_finger_pose_value("r_f_link3_tip", "world", obs_dict, env_idx)
        mf_pad_world = obs_encoder.get_finger_pose_value("r_f_link3_pad", "world", obs_dict, env_idx)
        rr.log("Plot_10_world_pos/tip_x", rr.Scalar(float(mf_tip_world['position'][0])))
        rr.log("Plot_10_world_pos/tip_y", rr.Scalar(float(mf_tip_world['position'][1])))
        rr.log("Plot_10_world_pos/tip_z", rr.Scalar(float(mf_tip_world['position'][2])))
        rr.log("Plot_10_world_pos/pad_x", rr.Scalar(float(mf_pad_world['position'][0])))
        rr.log("Plot_10_world_pos/pad_y", rr.Scalar(float(mf_pad_world['position'][1])))
        rr.log("Plot_10_world_pos/pad_z", rr.Scalar(float(mf_pad_world['position'][2])))
        
        # Plot 11: middle finger fingerpad and fingertip quad in world frame
        rr.log("Plot_11_world_quat/tip_x", rr.Scalar(float(mf_tip_world['orientation'][0])))
        rr.log("Plot_11_world_quat/tip_y", rr.Scalar(float(mf_tip_world['orientation'][1])))
        rr.log("Plot_11_world_quat/tip_z", rr.Scalar(float(mf_tip_world['orientation'][2])))
        rr.log("Plot_11_world_quat/tip_w", rr.Scalar(float(mf_tip_world['orientation'][3])))
        rr.log("Plot_11_world_quat/pad_x", rr.Scalar(float(mf_pad_world['orientation'][0])))
        rr.log("Plot_11_world_quat/pad_y", rr.Scalar(float(mf_pad_world['orientation'][1])))
        rr.log("Plot_11_world_quat/pad_z", rr.Scalar(float(mf_pad_world['orientation'][2])))
        rr.log("Plot_11_world_quat/pad_w", rr.Scalar(float(mf_pad_world['orientation'][3])))
        
        # Plot 12: middle finger fingerpad and fingertip x,y,z in hand frame
        mf_tip_hand = obs_encoder.get_finger_pose_value("r_f_link3_tip", "hand", obs_dict, env_idx)
        mf_pad_hand = obs_encoder.get_finger_pose_value("r_f_link3_pad", "hand", obs_dict, env_idx)
        rr.log("Plot_12_hand_pos/tip_x", rr.Scalar(float(mf_tip_hand['position'][0])))
        rr.log("Plot_12_hand_pos/tip_y", rr.Scalar(float(mf_tip_hand['position'][1])))
        rr.log("Plot_12_hand_pos/tip_z", rr.Scalar(float(mf_tip_hand['position'][2])))
        rr.log("Plot_12_hand_pos/pad_x", rr.Scalar(float(mf_pad_hand['position'][0])))
        rr.log("Plot_12_hand_pos/pad_y", rr.Scalar(float(mf_pad_hand['position'][1])))
        rr.log("Plot_12_hand_pos/pad_z", rr.Scalar(float(mf_pad_hand['position'][2])))
        
        # Plot 13: middle finger fingerpad and fingertip quad in hand frame
        rr.log("Plot_13_hand_quat/tip_x", rr.Scalar(float(mf_tip_hand['orientation'][0])))
        rr.log("Plot_13_hand_quat/tip_y", rr.Scalar(float(mf_tip_hand['orientation'][1])))
        rr.log("Plot_13_hand_quat/tip_z", rr.Scalar(float(mf_tip_hand['orientation'][2])))
        rr.log("Plot_13_hand_quat/tip_w", rr.Scalar(float(mf_tip_hand['orientation'][3])))
        rr.log("Plot_13_hand_quat/pad_x", rr.Scalar(float(mf_pad_hand['orientation'][0])))
        rr.log("Plot_13_hand_quat/pad_y", rr.Scalar(float(mf_pad_hand['orientation'][1])))
        rr.log("Plot_13_hand_quat/pad_z", rr.Scalar(float(mf_pad_hand['orientation'][2])))
        rr.log("Plot_13_hand_quat/pad_w", rr.Scalar(float(mf_pad_hand['orientation'][3])))
        
        # Plot 14: r_f_joint5_1 pos and target; active finger ff_spr pos and target
        joint5_1_pos = obs_encoder.get_raw_finger_dof("r_f_joint5_1", "pos", obs_dict, env_idx)
        joint5_1_target = obs_encoder.get_raw_finger_dof("r_f_joint5_1", "target", obs_dict, env_idx)
        ff_spr_pos = obs_encoder.get_active_finger_dof_value("ff_spr", "pos", obs_dict, env_idx)
        ff_spr_target = obs_encoder.get_active_finger_dof_value("ff_spr", "target", obs_dict, env_idx)
        rr.log("Plot_14_raw_DOFs/joint5_1_pos", rr.Scalar(float(joint5_1_pos)))
        rr.log("Plot_14_raw_DOFs/joint5_1_target", rr.Scalar(float(joint5_1_target)))
        rr.log("Plot_14_raw_DOFs/ff_spr_pos", rr.Scalar(float(ff_spr_pos)))
        rr.log("Plot_14_raw_DOFs/ff_spr_target", rr.Scalar(float(ff_spr_target)))
        
    except Exception as e:
        print(f"Error logging data at step {step}: {e}")
        import traceback
        traceback.print_exc()

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

    # Action mode control arguments
    parser.add_argument("--control-mode", type=str, choices=["position", "position_delta"], default="position",
                       help="Control mode: position (absolute) or position_delta (incremental)")
    parser.add_argument("--policy-controls-base", type=str, default="false", choices=["true", "false"],
                       help="Include hand base in policy action space (default: false)")
    parser.add_argument("--policy-controls-fingers", type=str, default="true", choices=["true", "false"],
                       help="Include fingers in policy action space (default: true)")
    parser.add_argument("--enable-plotting", action="store_true", help="Enable real-time plotting with Rerun")
    parser.add_argument("--plot-env-idx", type=int, default=0, help="Environment index to plot (default: 0)")
    args = parser.parse_args()

    print("Starting DexHand test script...")
    
    # Initialize plotting if requested
    plotting_enabled = False
    if args.enable_plotting:
        plotting_enabled = setup_rerun_logging()
        if plotting_enabled:
            print(f"Real-time plotting enabled for environment index {args.plot_env_idx}")
        else:
            print("Plotting requested but Rerun not available")

    # Load configuration
    cfg = load_config(args.config)

    # Override with command line arguments
    cfg["env"]["numEnvs"] = args.num_envs
    cfg["env"]["episodeLength"] = args.episode_length

    # Apply action mode configuration
    cfg["env"]["controlMode"] = args.control_mode
    cfg["env"]["policyControlsHandBase"] = args.policy_controls_base.lower() == "true"
    cfg["env"]["policyControlsFingers"] = args.policy_controls_fingers.lower() == "true"

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
    print(f"Policy controls hand base: {env.policy_controls_hand_base}")
    print(f"Policy controls fingers: {env.policy_controls_fingers}")

    # Calculate expected action space size
    expected_actions = 0
    if env.policy_controls_hand_base:
        expected_actions += 6  # base DOFs
    if env.policy_controls_fingers:
        expected_actions += 12  # finger controls

    if env.num_actions != expected_actions:
        print(f"ERROR: Expected {expected_actions} actions, got {env.num_actions}")
        return

    if env.action_control_mode != args.control_mode:
        print(f"ERROR: Expected {args.control_mode} control mode, got {env.action_control_mode}")
        return

    # Set up rule-based controllers for uncontrolled DOFs
    base_controller = None if env.policy_controls_hand_base else create_rule_based_base_controller()
    finger_controller = None if env.policy_controls_fingers else create_rule_based_finger_controller()

    if base_controller or finger_controller:
        print("\nSetting up rule-based controllers:")
        if base_controller:
            print("- Base controller: Active (circular motion)")
        if finger_controller:
            print("- Finger controller: Active (adaptive grasping with 5x speed)")
        env.set_rule_based_controllers(
            base_controller=base_controller,
            finger_controller=finger_controller
        )

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

    print(f"\n===== ACTION MODE VERIFICATION =====")
    print(f"Control Mode: {env.action_control_mode}")
    print(f"Policy controls base: {env.policy_controls_hand_base}")
    print(f"Policy controls fingers: {env.policy_controls_fingers}")
    print(f"Action space size: {env.num_actions}")

    if not env.policy_controls_hand_base:
        print("- Hand base will use RULE-BASED control (circular motion)")
    if not env.policy_controls_fingers:
        print("- Fingers will use RULE-BASED control (grasping motion)")

    if env.policy_controls_fingers:
        print(f"Testing {len(action_to_dof_map)} finger actions:")
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
        # Exit early if we've reached the requested number of steps
        if step >= args.steps:
            break
        # Initialize policy actions with proper defaults:
        # - Base DOFs: 0.0 (middle of range, neutral position)
        # - Finger DOFs: -1.0 (minimum of range, closed/contracted position)
        actions[:] = 0.0  # Initialize all to 0 first

        # Set finger actions to -1.0 (closed position) as default
        if env.policy_controls_fingers:
            # Find where finger actions start in the action space
            finger_start_idx = env.action_processor.NUM_BASE_DOFS if env.policy_controls_hand_base else 0
            finger_end_idx = finger_start_idx + env.action_processor.NUM_ACTIVE_FINGER_DOFS
            actions[:, finger_start_idx:finger_end_idx] = -1.0

        # Initialize action tracking variables (needed for progress display)
        finger_actions_count = env.action_processor.NUM_ACTIVE_FINGER_DOFS
        base_actions_count = env.action_processor.NUM_BASE_DOFS
        current_action_idx = step // steps_per_action
        step_in_action = step % steps_per_action
        action_value = 0.0  # Default for rule-based control

        # For base joints: Create pattern 0 → -1 → 1 → 0 over 100 steps (base middle is default)
        # For finger joints: Create pattern -1 → 1 → -1 over 100 steps (finger 0 is closed)
        if env.policy_controls_hand_base and current_action_idx < base_actions_count:
            # Base joints: 0 → -1 → 1 → 0 pattern
            if step_in_action < 25:
                # 0 → -1 (first quarter)
                progress = step_in_action / 24.0
                action_value = 0.0 - progress
            elif step_in_action < 75:
                # -1 → 1 (middle half)
                progress = (step_in_action - 25) / 49.0
                action_value = -1.0 + 2.0 * progress
            else:
                # 1 → 0 (last quarter)
                progress = (step_in_action - 75) / 24.0
                action_value = 1.0 - progress
        else:
            # Finger joints: -1 → 1 → -1 pattern
            if step_in_action < 50:
                progress = step_in_action / 49.0
                action_value = -1.0 + 2.0 * progress
            else:
                progress = (step_in_action - 50) / 49.0
                action_value = 1.0 - 2.0 * progress

        # For policy-controlled base
        if env.policy_controls_hand_base:
            base_action_idx = current_action_idx

            if base_action_idx < base_actions_count:
                # Use range [-0.5, 0.5] for base joints to reduce fierce movements
                scaled_base_action = action_value * 0.5
                actions[0, base_action_idx] = scaled_base_action

        # For policy-controlled fingers
        if env.policy_controls_fingers:
            # Apply to policy actions
            action_start_idx = base_actions_count if env.policy_controls_hand_base else 0
            finger_action_idx = current_action_idx

            if finger_action_idx < finger_actions_count:
                actions[0, action_start_idx + finger_action_idx] = action_value

        # Step the simulation (rule-based control is applied automatically in pre_physics_step)
        if step < 5:
            print(f"[Test] Before step {step}: progress_buf = {env.progress_buf[0].item()}, reset_buf = {env.reset_buf[0].item()}")
        obs, rewards, dones, info = env.step(actions)
        if step < 5:
            print(f"[Test] After step {step}: progress_buf = {env.progress_buf[0].item()}, reset_buf = {env.reset_buf[0].item()}, done = {dones[0].item()}")
        env.render()
        
        # Log observation data for plotting
        if plotting_enabled:
            rr.set_time_sequence("step", step)
            log_observation_data(env, step, args.plot_env_idx)

        # Print progress every 25 steps and at key transitions
        if (step_in_action % 25 == 0 or step_in_action == 49 or step_in_action == 99):
            if env.policy_controls_hand_base and current_action_idx < base_actions_count:
                base_action_names = ["ARTx", "ARTy", "ARTz", "ARRx", "ARRy", "ARRz"]
                action_name = base_action_names[current_action_idx]
                action_sent = actions[0, current_action_idx].item() if current_action_idx < actions.shape[1] else action_value
                print(f"  Step {step+1:4d}: Base Action {current_action_idx} ({action_name:>13}) = {action_sent:+6.3f} (substep {step_in_action+1:2d}/100)")
            elif env.policy_controls_fingers and current_action_idx < finger_actions_count:
                action_name = action_to_dof_map[current_action_idx][0]
                print(f"  Step {step+1:4d}: Finger Action {current_action_idx} ({action_name:>13}) = {action_value:+6.3f} (substep {step_in_action+1:2d}/100)")
            elif not env.policy_controls_fingers:
                print(f"  Step {step+1:4d}: RULE-BASED finger control (substep {step_in_action+1:2d}/100)")
            elif not env.policy_controls_hand_base:
                print(f"  Step {step+1:4d}: RULE-BASED base control (substep {step_in_action+1:2d}/100)")
            else:
                print(f"  Step {step+1:4d}: Test completed (substep {step_in_action+1:2d}/100)")

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
