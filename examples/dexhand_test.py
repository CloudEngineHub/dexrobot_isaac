#!/usr/bin/env python
"""
Test script for DexHand environment using BaseTask only.

This script creates a DexHand environment hardcoded to use BaseTask
and runs tests to verify hand movement and control functionality.
Specifically designed for BaseTask testing - does not support other tasks.

IMPORTANT: To test position control mode, use `task.controlMode=position`
(not `env.controlMode`) to properly override the BaseTask configuration.

Note: This script is hardcoded to BaseTask. Use train.py for other tasks.
"""

import os
import sys
import time
import numpy as np
import math
import hydra
from omegaconf import DictConfig, OmegaConf

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import IsaacGym first
from isaacgym import gymapi  # noqa: E402

# Then import PyTorch
import torch  # noqa: E402

# Import loguru
from loguru import logger  # noqa: E402

# Import factory and task
from dexhand_env.factory import create_dex_env  # noqa: E402
from dexhand_env.tasks.base_task import BaseTask  # noqa: E402

# Global variables for velocity integration tracking
arr_integrated_pos = None
arr_initial_pos = None
physics_dt = None
finger_integrated_pos = None
finger_initial_pos = None

# Import scipy for quaternion conversion
try:
    from scipy.spatial.transform import Rotation

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.info(
        "Scipy not available. Install with 'pip install scipy' for quaternion to Euler conversion."
    )

# Import Rerun for plotting (optional)
try:
    import rerun as rr

    RERUN_AVAILABLE = True
except ImportError:
    RERUN_AVAILABLE = False
    logger.info(
        "Rerun not available. Install with 'pip install rerun-sdk' for real-time plotting."
    )


def setup_logging(level):
    """Configure loguru based on verbosity level."""
    # Remove default handler
    logger.remove()

    # Add new handler with appropriate level
    if level == "debug":
        logger.add(
            sys.stderr,
            level="DEBUG",
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        )
    elif level == "info":
        logger.add(
            sys.stderr,
            level="INFO",
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        )
    elif level == "warning":
        logger.add(
            sys.stderr,
            level="WARNING",
            format="<level>{level: <8}</level> | <level>{message}</level>",
        )
    elif level == "error":
        logger.add(
            sys.stderr,
            level="ERROR",
            format="<level>{level: <8}</level> | <level>{message}</level>",
        )
    else:
        logger.add(
            sys.stderr,
            level="INFO",
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        )


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

        # Get episode time using the proper interface
        sim_time = env.get_episode_time(env_id=0)

        # Use 2π for 1 second period for circular motion
        t = sim_time * 2.0 * math.pi  # 1 Hz base frequency

        # Circular motion (relative displacement in meters)
        base_targets[:, 0] = 0.1 * math.sin(t)  # ARTx (±10cm)
        base_targets[:, 1] = 0.1 * math.cos(t)  # ARTy (±10cm)
        base_targets[:, 2] = 0.1 * math.sin(t * 0.5)  # ARTz (±10cm, slower)

        # Rotation oscillations (30 degrees = 0.5236 radians)
        base_targets[:, 3] = 0.5236 * math.sin(t * 0.7)  # ARRx (±30 degrees)
        base_targets[:, 4] = 0.5236 * math.cos(t * 0.8)  # ARRy (±30 degrees)
        base_targets[:, 5] = 0.5236 * math.sin(t * 1.2)  # ARRz (±30 degrees)

        return base_targets

    return base_controller


def create_rule_based_finger_controller(debug=False):
    """
    Create a rule-based controller function for fingers.
    Returns a function that takes env and returns finger targets.

    Args:
        debug: Enable debug logging
    """
    # Create a mutable object to track state
    state = {"call_count": 0, "start_time": None}

    def finger_controller(env):
        """
        Generate rule-based control for fingers (12 finger controls).
        Returns raw finger targets in physically meaningful units (radians).
        """
        finger_targets = torch.zeros((env.num_envs, 12), device=env.device)

        # Get episode time using the proper interface
        sim_time = env.get_episode_time(env_id=0)

        # Use 2π for 0.5 second period for grasping (2 Hz)
        t = sim_time * 2.0 * math.pi * 2.0  # 2 Hz for faster finger motion

        # Debug: Log time info when episode_step_count changes or first few calls
        last_progress = getattr(finger_controller, "last_progress", -1)
        current_progress = env.episode_step_count[0].item()

        if debug and (state["call_count"] < 3 or current_progress != last_progress):
            logger.debug(
                f"[Finger Controller] Call #{state['call_count']}, episode_step_count: {current_progress}, sim_time: {sim_time:.6f}, t: {t:.6f}, sin(t): {math.sin(t):.6f}"
            )

        finger_controller.last_progress = current_progress
        state["call_count"] += 1

        # Coordinated finger motion - simulate grasping
        grasp_wave = 0.5 * (1 + math.sin(t))  # 0 to 1 wave

        # Check if we have contact (can use env state)
        if hasattr(env, "contact_forces"):
            # Sum contact forces across fingertips
            total_contact = env.contact_forces.sum(
                dim=(1, 2)
            )  # Sum over fingers and force dims
            # If we have significant contact, increase grasp strength
            contact_detected = total_contact > 0.1
            grasp_scale = torch.where(contact_detected, 1.2, 1.0).unsqueeze(1)
        else:
            grasp_scale = 1.0

        # Thumb motion (actions 0-2) - physical joint ranges
        finger_targets[:, 0] = 0.2 * grasp_wave  # thumb_spread (0-0.2 rad)
        finger_targets[:, 1] = (
            1.0 * grasp_wave * grasp_scale.squeeze()
        )  # thumb_mcp (0-1.0 rad)
        finger_targets[:, 2] = (
            1.2 * grasp_wave * grasp_scale.squeeze()
        )  # thumb_dip (0-1.2 rad)

        # Finger spread (action 3) - physical range for spread joints
        finger_targets[:, 3] = 0.1 * grasp_wave  # finger_spread (0-0.1 rad)

        # Index finger (actions 4-5)
        finger_targets[:, 4] = (
            1.0 * grasp_wave * grasp_scale.squeeze()
        )  # index_mcp (0-1.0 rad)
        finger_targets[:, 5] = (
            1.3 * grasp_wave * grasp_scale.squeeze()
        )  # index_dip (0-1.3 rad)

        # Middle finger (actions 6-7)
        finger_targets[:, 6] = (
            1.0 * grasp_wave * grasp_scale.squeeze()
        )  # middle_mcp (0-1.0 rad)
        finger_targets[:, 7] = (
            1.3 * grasp_wave * grasp_scale.squeeze()
        )  # middle_dip (0-1.3 rad)

        # Ring finger (actions 8-9)
        finger_targets[:, 8] = (
            0.9 * grasp_wave * grasp_scale.squeeze()
        )  # ring_mcp (0-0.9 rad)
        finger_targets[:, 9] = (
            1.2 * grasp_wave * grasp_scale.squeeze()
        )  # ring_dip (0-1.2 rad)

        # Pinky finger (actions 10-11) - less motion
        finger_targets[:, 10] = (
            0.8 * grasp_wave * grasp_scale.squeeze()
        )  # pinky_mcp (0-0.8 rad)
        finger_targets[:, 11] = (
            1.1 * grasp_wave * grasp_scale.squeeze()
        )  # pinky_dip (0-1.1 rad)

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
        ("Plot_7_hand_quat_euler", "Hand quaternion and Euler angles"),
        ("Plot_8_contact", "Contact forces"),
        ("Plot_9_actions", "Actions comparison"),
        ("Plot_10_world_pos", "World frame positions"),
        ("Plot_11_world_quat", "World frame quaternions and Euler angles"),
        ("Plot_12_hand_pos", "Hand frame positions"),
        ("Plot_13_hand_quat", "Hand frame quaternions and Euler angles"),
        ("Plot_14_raw_DOFs", "Raw vs active DOFs"),
        ("Plot_15_ARR_integration", "ARR velocity integration check (Issue #1)"),
        ("Plot_16_finger_integration", "Finger joint velocity integration check"),
        # Reward plots
        ("Plot_17_rewards_common", "Common reward components"),
        ("Plot_18_rewards_weighted", "Weighted reward contributions"),
        ("Plot_19_rewards_total", "Total reward over time"),
        ("Plot_20_reward_alive", "Alive and height safety rewards"),
        ("Plot_21_reward_velocities", "Velocity penalty rewards"),
        ("Plot_22_reward_limits", "Joint limit penalty rewards"),
        ("Plot_23_reward_accelerations", "Acceleration penalty rewards"),
        ("Plot_24_reward_contact", "Contact stability reward"),
        ("Plot_25_reward_breakdown", "Reward breakdown by category"),
        ("Plot_26_reward_cumulative", "Cumulative rewards"),
    ]

    # Set up each plot with a clear structure
    for plot_name, description in plot_configs:
        # Log a text description for each plot
        rr.log(f"{plot_name}/description", rr.TextLog(description))

    return True


def log_observation_data(env, step, cfg, env_idx=0, debug=False):
    """Log observation data to Rerun for visualization."""
    if not RERUN_AVAILABLE:
        return

    try:
        # Get observation dictionary for convenient access
        obs_dict = env.get_observations_dict()
        obs_encoder = env.observation_encoder

        # Check if environment was just reset (episode_step_count == 0)
        episode_step = env.episode_step_count[env_idx].item()
        just_reset = episode_step == 0

        # Plot 1: ARTx pos, ARTx target
        artx_pos = obs_encoder.get_base_dof_value("ARTx", "pos", obs_dict, env_idx)
        artx_target = obs_encoder.get_base_dof_value(
            "ARTx", "target", obs_dict, env_idx
        )
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
        arrx_target = obs_encoder.get_base_dof_value(
            "ARRx", "target", obs_dict, env_idx
        )
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
        th_rot_pos = obs_encoder.get_active_finger_dof_value(
            "th_rot", "pos", obs_dict, env_idx
        )
        th_rot_target = obs_encoder.get_active_finger_dof_value(
            "th_rot", "target", obs_dict, env_idx
        )
        mf_mcp_pos = obs_encoder.get_active_finger_dof_value(
            "mf_mcp", "pos", obs_dict, env_idx
        )
        mf_mcp_target = obs_encoder.get_active_finger_dof_value(
            "mf_mcp", "target", obs_dict, env_idx
        )
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

        # Plot 7: hand_pose quat w,x,y,z AND Euler angles (both raw and ARR-aligned)
        if "hand_pose" in obs_dict:
            hand_quat = obs_dict["hand_pose"][env_idx, 3:7]

            # Debug: log quaternion values at first few steps
            if debug and step < 3:
                logger.debug(f"Step {step} - hand_quat raw: {hand_quat}")
                logger.debug(
                    f"  As list: [x={hand_quat[0]:.6f}, y={hand_quat[1]:.6f}, z={hand_quat[2]:.6f}, w={hand_quat[3]:.6f}]"
                )

            # Log quaternion components - Isaac Gym format is [x, y, z, w]
            rr.log("Plot_7_hand_quat_euler/quat_x", rr.Scalar(float(hand_quat[0])))
            rr.log("Plot_7_hand_quat_euler/quat_y", rr.Scalar(float(hand_quat[1])))
            rr.log("Plot_7_hand_quat_euler/quat_z", rr.Scalar(float(hand_quat[2])))
            rr.log("Plot_7_hand_quat_euler/quat_w", rr.Scalar(float(hand_quat[3])))

            # Convert quaternion to Euler angles and log them
            if SCIPY_AVAILABLE:
                # Isaac Gym uses [x, y, z, w] format which matches scipy's default
                quat_numpy = (
                    hand_quat.cpu().numpy()
                    if hasattr(hand_quat, "cpu")
                    else np.array(hand_quat)
                )
                rotation = Rotation.from_quat(
                    quat_numpy
                )  # Isaac Gym format matches scipy default [x,y,z,w]

                # Get Euler angles in radians (roll, pitch, yaw)
                euler_angles = rotation.as_euler("xyz", degrees=False)

                # Debug: print Euler angles at first few steps
                if debug and step < 3:
                    print(
                        f"  Euler angles (rad): [roll={euler_angles[0]:.6f}, pitch={euler_angles[1]:.6f}, yaw={euler_angles[2]:.6f}]"
                    )
                    euler_degrees = rotation.as_euler("xyz", degrees=True)
                    print(
                        f"  Euler angles (deg): [roll={euler_degrees[0]:.2f}°, pitch={euler_degrees[1]:.2f}°, yaw={euler_degrees[2]:.2f}°]"
                    )

                rr.log(
                    "Plot_7_hand_quat_euler/euler_roll_rad", rr.Scalar(euler_angles[0])
                )
                rr.log(
                    "Plot_7_hand_quat_euler/euler_pitch_rad", rr.Scalar(euler_angles[1])
                )
                rr.log(
                    "Plot_7_hand_quat_euler/euler_yaw_rad", rr.Scalar(euler_angles[2])
                )

        # Also check ARR-aligned pose if available
        if "hand_pose_arr_aligned" in obs_dict:
            arr_aligned_quat = obs_dict["hand_pose_arr_aligned"][env_idx, 3:7]

            if debug and step < 3:
                logger.debug(
                    f"  ARR-aligned quat: [x={arr_aligned_quat[0]:.6f}, y={arr_aligned_quat[1]:.6f}, z={arr_aligned_quat[2]:.6f}, w={arr_aligned_quat[3]:.6f}]"
                )

            # Log ARR-aligned quaternion
            rr.log(
                "Plot_7_hand_quat_euler/arr_quat_x",
                rr.Scalar(float(arr_aligned_quat[0])),
            )
            rr.log(
                "Plot_7_hand_quat_euler/arr_quat_y",
                rr.Scalar(float(arr_aligned_quat[1])),
            )
            rr.log(
                "Plot_7_hand_quat_euler/arr_quat_z",
                rr.Scalar(float(arr_aligned_quat[2])),
            )
            rr.log(
                "Plot_7_hand_quat_euler/arr_quat_w",
                rr.Scalar(float(arr_aligned_quat[3])),
            )

            if SCIPY_AVAILABLE:
                arr_quat_numpy = (
                    arr_aligned_quat.cpu().numpy()
                    if hasattr(arr_aligned_quat, "cpu")
                    else np.array(arr_aligned_quat)
                )
                arr_rotation = Rotation.from_quat(arr_quat_numpy)
                arr_euler = arr_rotation.as_euler("xyz", degrees=False)

                if debug and step < 3:
                    print(
                        f"  ARR-aligned Euler (rad): [roll={arr_euler[0]:.6f}, pitch={arr_euler[1]:.6f}, yaw={arr_euler[2]:.6f}]"
                    )
                    arr_euler_deg = arr_rotation.as_euler("xyz", degrees=True)
                    print(
                        f"  ARR-aligned Euler (deg): [roll={arr_euler_deg[0]:.2f}°, pitch={arr_euler_deg[1]:.2f}°, yaw={arr_euler_deg[2]:.2f}°]"
                    )

                # Log ARR-aligned Euler angles
                rr.log(
                    "Plot_7_hand_quat_euler/arr_euler_roll_rad", rr.Scalar(arr_euler[0])
                )
                rr.log(
                    "Plot_7_hand_quat_euler/arr_euler_pitch_rad",
                    rr.Scalar(arr_euler[1]),
                )
                rr.log(
                    "Plot_7_hand_quat_euler/arr_euler_yaw_rad", rr.Scalar(arr_euler[2])
                )

        # Plot 8: contact_force x,y,z component and magnitude for middle finger distal phalanx
        if "contact_forces" in obs_dict:
            middle_finger_force = obs_encoder.get_contact_force_value(
                "r_f_link3_4", obs_dict, env_idx
            )  # Middle finger distal phalanx
            rr.log("Plot_8_contact/x", rr.Scalar(float(middle_finger_force[0])))
            rr.log("Plot_8_contact/y", rr.Scalar(float(middle_finger_force[1])))
            rr.log("Plot_8_contact/z", rr.Scalar(float(middle_finger_force[2])))
            magnitude = np.linalg.norm(middle_finger_force)
            rr.log("Plot_8_contact/magnitude", rr.Scalar(float(magnitude)))

        # Plot 9: unscaled previous action corresponding to ARTx, ARTx vel times control_dt
        if "prev_actions" in obs_dict and hasattr(
            env.action_processor, "unscale_actions"
        ):
            prev_actions = obs_dict["prev_actions"][
                env_idx : env_idx + 1
            ]  # Keep batch dimension
            unscaled_actions = env.action_processor.unscale_actions(prev_actions)
            if unscaled_actions.shape[1] > 0:
                unscaled_artx_action = float(
                    unscaled_actions[0, 0]
                )  # First action is ARTx
                artx_vel_times_dt = (
                    float(artx_vel * env.action_processor.control_dt)
                    if env.action_processor.control_dt
                    else 0.0
                )
                rr.log("Plot_9_actions/unscaled_artx", rr.Scalar(unscaled_artx_action))
                rr.log("Plot_9_actions/artx_vel_times_dt", rr.Scalar(artx_vel_times_dt))

        # Plot 10: middle finger fingerpad and fingertip x,y,z in world frame
        mf_tip_world = obs_encoder.get_finger_pose_value(
            "r_f_link3_tip", "world", obs_dict, env_idx
        )
        mf_pad_world = obs_encoder.get_finger_pose_value(
            "r_f_link3_pad", "world", obs_dict, env_idx
        )
        rr.log("Plot_10_world_pos/tip_x", rr.Scalar(float(mf_tip_world["position"][0])))
        rr.log("Plot_10_world_pos/tip_y", rr.Scalar(float(mf_tip_world["position"][1])))
        rr.log("Plot_10_world_pos/tip_z", rr.Scalar(float(mf_tip_world["position"][2])))
        rr.log("Plot_10_world_pos/pad_x", rr.Scalar(float(mf_pad_world["position"][0])))
        rr.log("Plot_10_world_pos/pad_y", rr.Scalar(float(mf_pad_world["position"][1])))
        rr.log("Plot_10_world_pos/pad_z", rr.Scalar(float(mf_pad_world["position"][2])))

        # Plot 11: middle finger fingerpad and fingertip quat in world frame (with Euler)
        tip_quat = mf_tip_world["orientation"]
        pad_quat = mf_pad_world["orientation"]

        # Log quaternions
        rr.log("Plot_11_world_quat/tip_quat_x", rr.Scalar(float(tip_quat[0])))
        rr.log("Plot_11_world_quat/tip_quat_y", rr.Scalar(float(tip_quat[1])))
        rr.log("Plot_11_world_quat/tip_quat_z", rr.Scalar(float(tip_quat[2])))
        rr.log("Plot_11_world_quat/tip_quat_w", rr.Scalar(float(tip_quat[3])))
        rr.log("Plot_11_world_quat/pad_quat_x", rr.Scalar(float(pad_quat[0])))
        rr.log("Plot_11_world_quat/pad_quat_y", rr.Scalar(float(pad_quat[1])))
        rr.log("Plot_11_world_quat/pad_quat_z", rr.Scalar(float(pad_quat[2])))
        rr.log("Plot_11_world_quat/pad_quat_w", rr.Scalar(float(pad_quat[3])))

        # Convert to Euler if scipy available
        if SCIPY_AVAILABLE:
            # Tip Euler angles in radians
            tip_rotation = Rotation.from_quat(tip_quat)
            tip_euler_rad = tip_rotation.as_euler("xyz", degrees=False)
            rr.log("Plot_11_world_quat/tip_euler_roll_rad", rr.Scalar(tip_euler_rad[0]))
            rr.log(
                "Plot_11_world_quat/tip_euler_pitch_rad", rr.Scalar(tip_euler_rad[1])
            )
            rr.log("Plot_11_world_quat/tip_euler_yaw_rad", rr.Scalar(tip_euler_rad[2]))

            # Pad Euler angles in radians
            pad_rotation = Rotation.from_quat(pad_quat)
            pad_euler_rad = pad_rotation.as_euler("xyz", degrees=False)
            rr.log("Plot_11_world_quat/pad_euler_roll_rad", rr.Scalar(pad_euler_rad[0]))
            rr.log(
                "Plot_11_world_quat/pad_euler_pitch_rad", rr.Scalar(pad_euler_rad[1])
            )
            rr.log("Plot_11_world_quat/pad_euler_yaw_rad", rr.Scalar(pad_euler_rad[2]))

        # Plot 12: middle finger fingerpad and fingertip x,y,z in hand frame
        mf_tip_hand = obs_encoder.get_finger_pose_value(
            "r_f_link3_tip", "hand", obs_dict, env_idx
        )
        mf_pad_hand = obs_encoder.get_finger_pose_value(
            "r_f_link3_pad", "hand", obs_dict, env_idx
        )
        rr.log("Plot_12_hand_pos/tip_x", rr.Scalar(float(mf_tip_hand["position"][0])))
        rr.log("Plot_12_hand_pos/tip_y", rr.Scalar(float(mf_tip_hand["position"][1])))
        rr.log("Plot_12_hand_pos/tip_z", rr.Scalar(float(mf_tip_hand["position"][2])))
        rr.log("Plot_12_hand_pos/pad_x", rr.Scalar(float(mf_pad_hand["position"][0])))
        rr.log("Plot_12_hand_pos/pad_y", rr.Scalar(float(mf_pad_hand["position"][1])))
        rr.log("Plot_12_hand_pos/pad_z", rr.Scalar(float(mf_pad_hand["position"][2])))

        # Plot 13: middle finger fingerpad and fingertip quat in hand frame (with Euler)
        tip_quat_hand = mf_tip_hand["orientation"]
        pad_quat_hand = mf_pad_hand["orientation"]

        # Log quaternions
        rr.log("Plot_13_hand_quat/tip_quat_x", rr.Scalar(float(tip_quat_hand[0])))
        rr.log("Plot_13_hand_quat/tip_quat_y", rr.Scalar(float(tip_quat_hand[1])))
        rr.log("Plot_13_hand_quat/tip_quat_z", rr.Scalar(float(tip_quat_hand[2])))
        rr.log("Plot_13_hand_quat/tip_quat_w", rr.Scalar(float(tip_quat_hand[3])))
        rr.log("Plot_13_hand_quat/pad_quat_x", rr.Scalar(float(pad_quat_hand[0])))
        rr.log("Plot_13_hand_quat/pad_quat_y", rr.Scalar(float(pad_quat_hand[1])))
        rr.log("Plot_13_hand_quat/pad_quat_z", rr.Scalar(float(pad_quat_hand[2])))
        rr.log("Plot_13_hand_quat/pad_quat_w", rr.Scalar(float(pad_quat_hand[3])))

        # Convert to Euler if scipy available
        if SCIPY_AVAILABLE:
            # Tip Euler angles in radians
            tip_rotation_hand = Rotation.from_quat(tip_quat_hand)
            tip_euler_hand_rad = tip_rotation_hand.as_euler("xyz", degrees=False)
            rr.log(
                "Plot_13_hand_quat/tip_euler_roll_rad", rr.Scalar(tip_euler_hand_rad[0])
            )
            rr.log(
                "Plot_13_hand_quat/tip_euler_pitch_rad",
                rr.Scalar(tip_euler_hand_rad[1]),
            )
            rr.log(
                "Plot_13_hand_quat/tip_euler_yaw_rad", rr.Scalar(tip_euler_hand_rad[2])
            )

            # Pad Euler angles in radians
            pad_rotation_hand = Rotation.from_quat(pad_quat_hand)
            pad_euler_hand_rad = pad_rotation_hand.as_euler("xyz", degrees=False)
            rr.log(
                "Plot_13_hand_quat/pad_euler_roll_rad", rr.Scalar(pad_euler_hand_rad[0])
            )
            rr.log(
                "Plot_13_hand_quat/pad_euler_pitch_rad",
                rr.Scalar(pad_euler_hand_rad[1]),
            )
            rr.log(
                "Plot_13_hand_quat/pad_euler_yaw_rad", rr.Scalar(pad_euler_hand_rad[2])
            )

        # Plot 14: r_f_joint5_1 pos and target; active finger ff_spr pos and target
        joint5_1_pos = obs_encoder.get_raw_finger_dof(
            "r_f_joint5_1", "pos", obs_dict, env_idx
        )
        joint5_1_target = obs_encoder.get_raw_finger_dof(
            "r_f_joint5_1", "target", obs_dict, env_idx
        )
        ff_spr_pos = obs_encoder.get_active_finger_dof_value(
            "ff_spr", "pos", obs_dict, env_idx
        )
        ff_spr_target = obs_encoder.get_active_finger_dof_value(
            "ff_spr", "target", obs_dict, env_idx
        )
        rr.log("Plot_14_raw_DOFs/joint5_1_pos", rr.Scalar(float(joint5_1_pos)))
        rr.log("Plot_14_raw_DOFs/joint5_1_target", rr.Scalar(float(joint5_1_target)))
        rr.log("Plot_14_raw_DOFs/ff_spr_pos", rr.Scalar(float(ff_spr_pos)))
        rr.log("Plot_14_raw_DOFs/ff_spr_target", rr.Scalar(float(ff_spr_target)))

        # Plot 15: ARR Velocity Integration Check - Issue #1
        # Initialize integration tracking if needed (use global variables)
        global arr_integrated_pos, arr_initial_pos, physics_dt
        if arr_integrated_pos is None or physics_dt is None or just_reset:
            arr_integrated_pos = np.array([0.0, 0.0, 0.0])
            arr_initial_pos = None
            # Get physics_dt from config with fallback
            physics_dt = cfg.sim.get("dt", 0.0083) if "sim" in cfg else 0.0083
            if just_reset:
                logger.debug(
                    f"Reset ARR integration tracking due to environment reset (episode_step={episode_step})"
                )
            else:
                logger.debug(
                    f"Initialized ARR integration tracking with physics_dt = {physics_dt}"
                )

        # Always try to log even if there's an error, to see if the plot appears
        try:
            # Get current ARR positions and velocities
            arrx_pos = obs_encoder.get_base_dof_value("ARRx", "pos", obs_dict, env_idx)
            arry_pos = obs_encoder.get_base_dof_value("ARRy", "pos", obs_dict, env_idx)
            arrz_pos = obs_encoder.get_base_dof_value("ARRz", "pos", obs_dict, env_idx)
            arrx_vel = obs_encoder.get_base_dof_value("ARRx", "vel", obs_dict, env_idx)
            arry_vel = obs_encoder.get_base_dof_value("ARRy", "vel", obs_dict, env_idx)
            arrz_vel = obs_encoder.get_base_dof_value("ARRz", "vel", obs_dict, env_idx)

            # Track initial position
            if arr_initial_pos is None:
                arr_initial_pos = np.array([arrx_pos, arry_pos, arrz_pos])

            # Integrate velocities (skip first step of episode)
            if episode_step > 0:
                arr_integrated_pos[0] += arrx_vel * physics_dt
                arr_integrated_pos[1] += arry_vel * physics_dt
                arr_integrated_pos[2] += arrz_vel * physics_dt

            # Calculate actual position change from initial
            actual_pos_change = (
                np.array([arrx_pos, arry_pos, arrz_pos]) - arr_initial_pos
            )

            # Plot actual vs integrated positions
            rr.log(
                "Plot_15_ARR_integration/actual_arrx_change",
                rr.Scalar(float(actual_pos_change[0])),
            )
            rr.log(
                "Plot_15_ARR_integration/integrated_arrx_change",
                rr.Scalar(float(arr_integrated_pos[0])),
            )
            rr.log(
                "Plot_15_ARR_integration/actual_arry_change",
                rr.Scalar(float(actual_pos_change[1])),
            )
            rr.log(
                "Plot_15_ARR_integration/integrated_arry_change",
                rr.Scalar(float(arr_integrated_pos[1])),
            )
            rr.log(
                "Plot_15_ARR_integration/actual_arrz_change",
                rr.Scalar(float(actual_pos_change[2])),
            )
            rr.log(
                "Plot_15_ARR_integration/integrated_arrz_change",
                rr.Scalar(float(arr_integrated_pos[2])),
            )

            # Plot integration errors
            error_x = actual_pos_change[0] - arr_integrated_pos[0]
            error_y = actual_pos_change[1] - arr_integrated_pos[1]
            error_z = actual_pos_change[2] - arr_integrated_pos[2]
            rr.log("Plot_15_ARR_integration/error_arrx", rr.Scalar(float(error_x)))
            rr.log("Plot_15_ARR_integration/error_arry", rr.Scalar(float(error_y)))
            rr.log("Plot_15_ARR_integration/error_arrz", rr.Scalar(float(error_z)))

        except Exception as plot15_error:
            logger.error(f"Plot 15 error at step {step}: {plot15_error}")

        # Plot 16: Finger Joint Velocity Integration Check
        # Initialize finger integration tracking
        global finger_integrated_pos, finger_initial_pos
        if episode_step == 0 or finger_integrated_pos is None or just_reset:
            finger_integrated_pos = 0.0
            finger_initial_pos = None
            if just_reset:
                logger.debug(
                    f"Reset finger integration tracking due to environment reset (episode_step={episode_step})"
                )
            else:
                logger.debug("Initialized finger integration tracking")

        try:
            # Test thumb rotation joint (r_f_joint1_1) - a finger rotation joint
            finger_pos = obs_encoder.get_raw_finger_dof(
                "r_f_joint1_1", "pos", obs_dict, env_idx
            )
            finger_vel = obs_encoder.get_raw_finger_dof(
                "r_f_joint1_1", "vel", obs_dict, env_idx
            )

            # Track initial position
            if finger_initial_pos is None:
                finger_initial_pos = finger_pos

            # Integrate velocity (skip first step of episode)
            if episode_step > 0:
                finger_integrated_pos += finger_vel * physics_dt

            # Calculate actual position change from initial
            actual_finger_change = finger_pos - finger_initial_pos

            # Plot actual vs integrated positions for finger joint
            rr.log(
                "Plot_16_finger_integration/actual_r_f_joint1_1_change",
                rr.Scalar(float(actual_finger_change)),
            )
            rr.log(
                "Plot_16_finger_integration/integrated_r_f_joint1_1_change",
                rr.Scalar(float(finger_integrated_pos)),
            )

            # Plot integration error
            finger_error = actual_finger_change - finger_integrated_pos
            rr.log(
                "Plot_16_finger_integration/error_r_f_joint1_1",
                rr.Scalar(float(finger_error)),
            )

            if step == 1:
                logger.debug(f"Plot 16 logged successfully at step {step}")

        except Exception as plot16_error:
            logger.error(f"Plot 16 error at step {step}: {plot16_error}")

    except Exception as e:
        logger.error(f"Error logging data at step {step}: {e}")
        import traceback

        traceback.print_exc()


def log_reward_data(env, rewards, reward_components, step, env_idx=0):
    """Log reward data to Rerun for visualization."""
    if not RERUN_AVAILABLE:
        return

    try:
        # Initialize cumulative reward tracking
        if not hasattr(log_reward_data, "cumulative_reward"):
            log_reward_data.cumulative_reward = 0.0

        # Get total reward for this environment
        total_reward = rewards[env_idx].item() if rewards is not None else 0.0
        log_reward_data.cumulative_reward += total_reward

        # Plot 17: Common reward components (raw values)
        if reward_components:
            # Common rewards
            for name in [
                "alive",
                "height_safety",
                "finger_velocity",
                "hand_velocity",
                "hand_angular_velocity",
                "joint_limit",
                "finger_acceleration",
                "hand_acceleration",
                "hand_angular_acceleration",
                "contact_stability",
            ]:
                if name in reward_components:
                    value = reward_components[name][env_idx].item()
                    rr.log(f"Plot_17_rewards_common/{name}", rr.Scalar(float(value)))

        # Plot 18: Weighted reward contributions
        if reward_components:
            for name in [
                "alive",
                "height_safety",
                "finger_velocity",
                "hand_velocity",
                "hand_angular_velocity",
                "joint_limit",
                "finger_acceleration",
                "hand_acceleration",
                "hand_angular_acceleration",
                "contact_stability",
            ]:
                weighted_name = f"{name}_weighted"
                if weighted_name in reward_components:
                    value = reward_components[weighted_name][env_idx].item()
                    rr.log(f"Plot_18_rewards_weighted/{name}", rr.Scalar(float(value)))

        # Plot 19: Total reward over time
        rr.log("Plot_19_rewards_total/total", rr.Scalar(float(total_reward)))
        rr.log(
            "Plot_19_rewards_total/cumulative",
            rr.Scalar(float(log_reward_data.cumulative_reward)),
        )

        # Plot 20: Alive and height safety rewards (detailed)
        if reward_components:
            if "alive" in reward_components:
                rr.log(
                    "Plot_20_reward_alive/alive",
                    rr.Scalar(float(reward_components["alive"][env_idx].item())),
                )
            if "height_safety" in reward_components:
                rr.log(
                    "Plot_20_reward_alive/height_safety",
                    rr.Scalar(
                        float(reward_components["height_safety"][env_idx].item())
                    ),
                )

        # Plot 21: Velocity penalty rewards
        if reward_components:
            for name in ["finger_velocity", "hand_velocity", "hand_angular_velocity"]:
                if name in reward_components:
                    value = reward_components[name][env_idx].item()
                    rr.log(f"Plot_21_reward_velocities/{name}", rr.Scalar(float(value)))

        # Plot 22: Joint limit penalty
        if reward_components and "joint_limit" in reward_components:
            rr.log(
                "Plot_22_reward_limits/joint_limit",
                rr.Scalar(float(reward_components["joint_limit"][env_idx].item())),
            )

        # Plot 23: Acceleration penalties
        if reward_components:
            for name in [
                "finger_acceleration",
                "hand_acceleration",
                "hand_angular_acceleration",
            ]:
                if name in reward_components:
                    value = reward_components[name][env_idx].item()
                    rr.log(
                        f"Plot_23_reward_accelerations/{name}", rr.Scalar(float(value))
                    )

        # Plot 24: Contact stability
        if reward_components and "contact_stability" in reward_components:
            rr.log(
                "Plot_24_reward_contact/stability",
                rr.Scalar(
                    float(reward_components["contact_stability"][env_idx].item())
                ),
            )

        # Plot 25: Reward breakdown by category
        if reward_components:
            # Sum up different categories
            safety_rewards = 0.0
            velocity_rewards = 0.0
            acceleration_rewards = 0.0
            contact_rewards = 0.0

            # Safety (alive + height_safety + joint_limit)
            for name in [
                "alive_weighted",
                "height_safety_weighted",
                "joint_limit_weighted",
            ]:
                if name in reward_components:
                    safety_rewards += reward_components[name][env_idx].item()

            # Velocity penalties
            for name in [
                "finger_velocity_weighted",
                "hand_velocity_weighted",
                "hand_angular_velocity_weighted",
            ]:
                if name in reward_components:
                    velocity_rewards += reward_components[name][env_idx].item()

            # Acceleration penalties
            for name in [
                "finger_acceleration_weighted",
                "hand_acceleration_weighted",
                "hand_angular_acceleration_weighted",
            ]:
                if name in reward_components:
                    acceleration_rewards += reward_components[name][env_idx].item()

            # Contact
            if "contact_stability_weighted" in reward_components:
                contact_rewards = reward_components["contact_stability_weighted"][
                    env_idx
                ].item()

            rr.log("Plot_25_reward_breakdown/safety", rr.Scalar(float(safety_rewards)))
            rr.log(
                "Plot_25_reward_breakdown/velocity", rr.Scalar(float(velocity_rewards))
            )
            rr.log(
                "Plot_25_reward_breakdown/acceleration",
                rr.Scalar(float(acceleration_rewards)),
            )
            rr.log(
                "Plot_25_reward_breakdown/contact", rr.Scalar(float(contact_rewards))
            )

        # Plot 26: Cumulative rewards
        rr.log(
            "Plot_26_reward_cumulative/total",
            rr.Scalar(float(log_reward_data.cumulative_reward)),
        )

        # Reset cumulative on environment reset
        if (
            hasattr(env, "episode_step_count")
            and env.episode_step_count[env_idx].item() == 0
            and step > 0
        ):
            log_reward_data.cumulative_reward = total_reward  # Reset to current reward

    except Exception as e:
        logger.error(f"Error logging reward data at step {step}: {e}")
        import traceback

        traceback.print_exc()


def create_contact_test_box(gym, sim, env_ptr, env_id):
    """Create a box for contact testing."""
    # Create box asset
    box_size = 0.1  # 10cm cube
    box_x = 0.3  # Position beneath middle finger
    box_y = 0.0
    box_z = box_size / 2.0  # Half the box height to place bottom on ground
    box_position = gymapi.Vec3(box_x, box_y, box_z)

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True  # Make the box static
    asset_options.thickness = 0.001
    asset_options.density = 1000.0

    box_asset = gym.create_box(
        sim, box_size, box_size, box_size, asset_options  # width  # height  # depth
    )

    if box_asset is None:
        logger.warning(f"Failed to create box asset for environment {env_id}")
        return

    # Create initial pose for the box
    box_pose = gymapi.Transform()
    box_pose.p = box_position
    box_pose.r = gymapi.Quat(0, 0, 0, 1)  # No rotation

    # Create the box actor
    box_actor = gym.create_actor(
        env_ptr,
        box_asset,
        box_pose,
        f"contact_test_box_{env_id}",
        env_id,  # collision group
        1,  # collision filter (1 = collide with everything)
    )

    if box_actor is None:
        logger.warning(f"Failed to create box actor in environment {env_id}")
    else:
        # Set box color to red for visibility
        gym.set_rigid_body_color(
            env_ptr,
            box_actor,
            0,  # rigid body index (box only has one)
            gymapi.MESH_VISUAL,
            gymapi.Vec3(1.0, 0.0, 0.0),  # Red color
        )
        logger.info(
            f"Added contact test box to environment {env_id}: size={box_size}m, center at ({box_position.x}, {box_position.y}, {box_position.z})"
        )


class ContactTestTask(BaseTask):
    """Test task that adds a contact test box for testing contact forces."""

    def create_task_objects(self, gym, sim, env_ptr, env_id: int):
        """Add a contact test box to the environment."""
        # Call parent implementation first (though BaseTask doesn't add anything)
        super().create_task_objects(gym, sim, env_ptr, env_id)
        # Add our contact test box
        create_contact_test_box(gym, sim, env_ptr, env_id)


# For now, monkey patch BaseTask to add contact test box
# This is done properly to ensure actors are created in the correct order
def _patched_create_task_objects(self, gym, sim, env_ptr, env_id):
    """Patched version that adds contact test box."""
    # Call original implementation (which does nothing in BaseTask)
    # Note: We don't call super() here since we're patching the class method
    # BaseTask.create_task_objects doesn't do anything, so we can skip it
    create_contact_test_box(gym, sim, env_ptr, env_id)


BaseTask.create_task_objects = _patched_create_task_objects


@hydra.main(
    config_path="../dexhand_env/cfg", config_name="test_render", version_base=None
)
def main(cfg: DictConfig):
    """Main function to test the DexHand environment using BaseTask only.

    This script is hardcoded to BaseTask and provides:
    - Position and position_delta control mode testing
    - Policy control configuration verification (base/fingers)
    - Observation system validation with Rerun plotting
    - Contact force testing with simple box obstacles

    Usage examples:
      python dexhand_test.py                                     # Basic BaseTask test
      python dexhand_test.py env.viewer=false steps=500        # No viewer test for 500 steps
      python dexhand_test.py task.controlMode=position          # Test position control mode (IMPORTANT: use task.controlMode)
      python dexhand_test.py task.controlMode=position_delta    # Test position_delta control mode
      python dexhand_test.py task.policy_controls_hand_base=false   # Test finger-only control
      python dexhand_test.py enablePlotting=true env.videoRecord=true  # Test with visualization and recording
      python dexhand_test.py env.numEnvs=4 device=cuda:1        # Use 4 environments on GPU 1
    """

    # Set up logging
    log_level = cfg.get("log_level", "info")
    setup_logging(log_level)

    logger.info("Starting DexHand test script...")

    # Initialize plotting if requested
    plotting_enabled = False
    enable_plotting = cfg.get("enablePlotting", False)
    if enable_plotting:
        plotting_enabled = setup_rerun_logging()
        if plotting_enabled:
            plot_env_idx = cfg.get("plotEnvIdx", 0)
            logger.info(
                f"Real-time plotting enabled for environment index {plot_env_idx}"
            )
        else:
            logger.warning("Plotting requested but Rerun not available")

    # Validate argument combinations
    if enable_plotting and cfg.get("plotEnvIdx", 0) >= cfg.env.get("numEnvs", 1):
        raise ValueError(
            f"plotEnvIdx ({cfg.get('plotEnvIdx', 0)}) must be less than numEnvs ({cfg.env.get('numEnvs', 1)})"
        )

    # Make config flexible for setting new keys
    OmegaConf.set_struct(cfg, False)

    # Override initial hand position for contact testing
    if "env" not in cfg:
        cfg["env"] = {}
    cfg["env"]["initialHandPos"] = [0.0, 0.0, 0.15]  # Lower position for easier contact

    debug = cfg.get("debug", False)
    if debug:
        logger.debug("Configuration loaded:")
        logger.debug(OmegaConf.to_yaml(cfg))

    # Create the environment
    logger.info("Creating environment...")

    # Set simulation device based on configuration
    sim_device = cfg.get("device", "cuda:0")
    rl_device = cfg.get("device", "cuda:0")

    # Use -1 for graphics_device_id only if no viewer AND not recording video
    # Camera recording requires graphics device even without viewer
    viewer_enabled = cfg.env.get("viewer", False)
    video_record = cfg.env.get("videoRecord", False)
    has_video = video_record
    graphics_device_id = -1 if (not viewer_enabled and not has_video) else 0

    # Add debug information if requested
    if debug:
        logger.debug(f"Simulation device: {sim_device}")
        logger.debug(f"RL device: {rl_device}")
        logger.debug(f"Graphics device ID: {graphics_device_id}")
        # GPU pipeline is automatically determined from sim_device

        # Set physics debugging options
        if "physx" not in cfg["sim"]:
            cfg["sim"]["physx"] = {}

    # Create environment
    try:
        # Configure video recording if requested
        video_config = None
        if video_record:
            video_config = {
                "enabled": True,
                "output_dir": "/tmp/dexhand_videos",
                "resolution": [1024, 768],
                "fps": 30,
                "codec": "mp4v",
            }

        env = create_dex_env(
            task_name="BaseTask",
            cfg=cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            force_render=not viewer_enabled or has_video,
            virtual_screen_capture=has_video and not viewer_enabled,
            video_config=video_config,
        )
        logger.info("Environment created successfully!")
    except Exception as e:
        logger.error(f"Error creating environment: {e}")
        import traceback

        traceback.print_exc()
        return

    logger.info(f"Environment created with {env.num_envs} environments")
    logger.info(f"Observation space: {env.num_observations}")
    logger.info(f"Action space: {env.num_actions}")
    logger.info(f"Control mode: {env.action_control_mode}")
    logger.info(f"Policy controls hand base: {env.policy_controls_hand_base}")
    logger.info(f"Policy controls fingers: {env.policy_controls_fingers}")

    # Display keyboard shortcuts if viewer is enabled
    if viewer_enabled:
        logger.info("\nKeyboard shortcuts:")
        logger.info("  SPACE - Toggle random actions mode")
        logger.info("  E     - Reset current environment")
        logger.info("  G     - Toggle between single robot and global view")
        logger.info("  UP/DOWN - Navigate between robots")
        logger.info("  ENTER - Toggle camera view mode")
        logger.info("  C     - Toggle contact force visualization")

    # Calculate expected action space size
    expected_actions = 0
    if env.policy_controls_hand_base:
        expected_actions += 6  # base DOFs
    if env.policy_controls_fingers:
        expected_actions += 12  # finger controls

    if env.num_actions != expected_actions:
        logger.error(f"Expected {expected_actions} actions, got {env.num_actions}")
        return

    # Validate control mode - accept both position and position_delta as valid
    valid_control_modes = ["position", "position_delta"]
    if env.action_control_mode not in valid_control_modes:
        logger.error(
            f"Invalid control mode: {env.action_control_mode}. Supported modes: {valid_control_modes}"
        )
        return

    logger.info(f"Using control mode: {env.action_control_mode}")

    # First, set up the action rule based on control mode
    # This must be done before any actions are processed
    if env.action_control_mode == "position":

        def position_action_rule(
            active_prev_targets, active_rule_targets, actions, config
        ):
            """Position mode action rule that respects rule targets for uncontrolled DOFs."""
            # Start with rule targets - this preserves rule-based control for uncontrolled DOFs
            targets = active_rule_targets.clone()

            # Get action processor for scaling logic
            ap = env.action_processor

            # Apply position control logic only to policy-controlled DOFs
            if config["policy_controls_base"] and config["policy_controls_fingers"]:
                # Both base and fingers controlled by policy
                # Scale actions to DOF limits
                scaled_actions = torch.zeros_like(targets)
                scaled_actions[
                    :, ap.active_target_mask
                ] = ap.action_scaling.scale_to_limits(
                    actions,
                    ap.active_lower_limits[ap.active_target_mask],
                    ap.active_upper_limits[ap.active_target_mask],
                )

                # Compute diff and apply velocity clamping
                diff = scaled_actions - active_prev_targets
                clamped_diff = torch.clamp(diff, -ap.max_deltas, ap.max_deltas)
                targets = active_prev_targets + clamped_diff

            elif config["policy_controls_base"]:
                # Only base controlled by policy
                # Scale base actions
                base_lower = ap.active_lower_limits[:6]
                base_upper = ap.active_upper_limits[:6]
                scaled_base = ap.action_scaling.scale_to_limits(
                    actions, base_lower, base_upper
                )

                # Apply velocity clamping to base
                base_diff = scaled_base - active_prev_targets[:, :6]
                clamped_base_diff = torch.clamp(
                    base_diff, -ap.max_deltas[:6], ap.max_deltas[:6]
                )
                targets[:, :6] = active_prev_targets[:, :6] + clamped_base_diff

            elif config["policy_controls_fingers"]:
                # Only fingers controlled by policy
                # Scale finger actions
                finger_lower = ap.active_lower_limits[6:]
                finger_upper = ap.active_upper_limits[6:]
                scaled_fingers = ap.action_scaling.scale_to_limits(
                    actions, finger_lower, finger_upper
                )

                # Apply velocity clamping to fingers
                finger_diff = scaled_fingers - active_prev_targets[:, 6:]
                clamped_finger_diff = torch.clamp(
                    finger_diff, -ap.max_deltas[6:], ap.max_deltas[6:]
                )
                targets[:, 6:] = active_prev_targets[:, 6:] + clamped_finger_diff

            return targets

        env.action_processor.set_action_rule(position_action_rule)

    else:  # position_delta mode

        def position_delta_action_rule(
            active_prev_targets, active_rule_targets, actions, config
        ):
            """Position delta mode action rule that respects rule targets for uncontrolled DOFs."""
            # Start with rule targets - this preserves rule-based control for uncontrolled DOFs
            targets = active_rule_targets.clone()

            # Get action processor for scaling logic
            ap = env.action_processor

            # Apply position delta logic only to policy-controlled DOFs
            if config["policy_controls_base"] and config["policy_controls_fingers"]:
                # Both base and fingers controlled by policy
                scaled_deltas = actions * ap.max_deltas[ap.active_target_mask]
                targets[:, ap.active_target_mask] = (
                    active_prev_targets[:, ap.active_target_mask] + scaled_deltas
                )

            elif config["policy_controls_base"]:
                # Only base controlled by policy
                scaled_base_deltas = actions * ap.max_deltas[:6]
                targets[:, :6] = active_prev_targets[:, :6] + scaled_base_deltas

            elif config["policy_controls_fingers"]:
                # Only fingers controlled by policy
                scaled_finger_deltas = actions * ap.max_deltas[6:]
                targets[:, 6:] = active_prev_targets[:, 6:] + scaled_finger_deltas

            # Clamp to DOF limits
            targets = torch.clamp(
                targets, ap.active_lower_limits, ap.active_upper_limits
            )

            return targets

        env.action_processor.set_action_rule(position_delta_action_rule)

    # Now set up rule-based controllers for uncontrolled DOFs
    base_controller = (
        None if env.policy_controls_hand_base else create_rule_based_base_controller()
    )
    finger_controller = (
        None
        if env.policy_controls_fingers
        else create_rule_based_finger_controller(debug)
    )

    # Set up pre-action rule if any controllers are needed
    if base_controller or finger_controller:
        logger.info("Setting up pre-action rule for uncontrolled DOFs:")
        if base_controller:
            logger.info("- Base controller: Active (circular motion)")
        if finger_controller:
            logger.info("- Finger controller: Active (adaptive grasping with 5x speed)")

        # Define pre-action rule that applies rule-based control
        def rule_based_pre_action(active_prev_targets, state):
            """Pre-action rule with rule-based controllers."""
            env = state["env"]

            active_targets = active_prev_targets.clone()

            # Apply base controller if not policy-controlled
            if base_controller and not env.policy_controls_hand_base:
                base_targets = base_controller(env)
                active_targets[:, :6] = base_targets

            # Apply finger controller if not policy-controlled
            if finger_controller and not env.policy_controls_fingers:
                finger_targets = finger_controller(env)
                active_targets[:, 6:] = finger_targets

            return active_targets

        # Register the pre-action rule
        env.action_processor.set_pre_action_rule(rule_based_pre_action)

    # Initialize actions tensor
    actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)

    # Define expected action mapping for base controls
    base_action_map = [
        ("ARTx", "Base Translation X"),
        ("ARTy", "Base Translation Y"),
        ("ARTz", "Base Translation Z"),
        ("ARRx", "Base Rotation X"),
        ("ARRy", "Base Rotation Y"),
        ("ARRz", "Base Rotation Z"),
    ]

    # Define expected action mapping for 12 finger controls with coupling
    finger_action_map = [
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

    logger.info("\n===== ACTION MODE VERIFICATION =====")
    logger.info(f"Control Mode: {env.action_control_mode}")
    logger.info(f"Policy controls base: {env.policy_controls_hand_base}")
    logger.info(f"Policy controls fingers: {env.policy_controls_fingers}")
    logger.info(f"Action space size: {env.num_actions}")

    if not env.policy_controls_hand_base:
        logger.info("- Hand base will use RULE-BASED control (circular motion)")
    if not env.policy_controls_fingers:
        logger.info("- Fingers will use RULE-BASED control (grasping motion)")

    if env.policy_controls_hand_base:
        logger.info(f"Testing {len(base_action_map)} base actions:")
        for i, (dof_name, description) in enumerate(base_action_map):
            logger.info(f"  Base Action {i:2d}: {dof_name:<15} - {description}")

    if env.policy_controls_fingers:
        logger.info(f"Testing {len(finger_action_map)} finger actions:")
        for i, (dof_name, description) in enumerate(finger_action_map):
            logger.info(f"  Finger Action {i:2d}: {dof_name:<15} - {description}")

    if env.policy_controls_hand_base and env.policy_controls_fingers:
        logger.info("\nNOTE: Base and finger actions will be tested CONCURRENTLY")

    logger.info("=" * 50)

    # Get initial DOF positions for reference
    obs = env.reset()
    initial_dof_pos = env.dof_pos.clone() if hasattr(env, "dof_pos") else None

    # Log control_dt information
    logger.info("\nPhysics timing:")
    logger.info(f"  physics_dt: {env.physics_manager.physics_dt}")
    logger.info(f"  control_dt: {env.physics_manager.control_dt}")
    logger.info(
        f"  physics_steps_per_control: {env.physics_manager.physics_steps_per_control_step}"
    )

    logger.info("\nStarting action-to-DOF verification test...")
    logger.info("Movement pattern: -1 → 1 → -1 over 100 steps")
    logger.info("Steps per action: 100")
    logger.info(f"Total test duration: {12 * 100} steps")
    logger.info("=" * 50)

    # Action testing: each action goes from -1 → 1 → -1 over 100 steps
    if env.policy_controls_hand_base and env.policy_controls_fingers:
        logger.info("\n>>> Testing Concurrent Base and Finger Movement")
        logger.info("Base and finger actions will move together")
    else:
        logger.info("\n>>> Testing Sequential Action Movement")
    logger.info("Each action will move from -1 → 1 → -1 over 100 steps")
    logger.info(
        "Starting with all actions at -1, then testing each action individually"
    )

    # Reset to initial state
    env.reset()

    # Get initial DOF positions for reference
    if hasattr(env, "dof_pos"):
        initial_dof_pos = env.dof_pos[0].clone()
        logger.debug("Initial finger DOF positions:")
        finger_dofs = [
            "r_f_joint1_1",
            "r_f_joint1_2",
            "r_f_joint1_3",
            "r_f_joint1_4",
            "r_f_joint2_1",
            "r_f_joint2_2",
            "r_f_joint2_3",
            "r_f_joint2_4",
            "r_f_joint3_1",
            "r_f_joint3_2",
            "r_f_joint3_3",
            "r_f_joint3_4",
            "r_f_joint4_1",
            "r_f_joint4_2",
            "r_f_joint4_3",
            "r_f_joint4_4",
            "r_f_joint5_1",
            "r_f_joint5_2",
            "r_f_joint5_3",
            "r_f_joint5_4",
        ]
        for i, name in enumerate(finger_dofs):
            dof_idx = 6 + i  # finger DOFs start at index 6
            logger.debug(f"  {name}: {initial_dof_pos[dof_idx].item():.6f}")

    # Calculate total steps based on control mode
    steps_per_action = 100
    if env.policy_controls_hand_base and env.policy_controls_fingers:
        # Concurrent testing: max of base (6) and finger (12) actions
        total_actions = max(
            env.action_processor.NUM_BASE_DOFS,
            env.action_processor.NUM_ACTIVE_FINGER_DOFS,
        )
    elif env.policy_controls_hand_base:
        total_actions = env.action_processor.NUM_BASE_DOFS
    else:
        total_actions = env.action_processor.NUM_ACTIVE_FINGER_DOFS

    total_steps = total_actions * steps_per_action
    steps = cfg.get("steps", 1200)
    action_cycles = (steps + total_steps - 1) // total_steps  # Ceiling division

    logger.info("\nStarting sequential action test:")
    logger.info(f"User requested steps: {steps}")
    logger.info(f"Actions to test: {total_actions}")
    logger.info(f"Steps per action: {steps_per_action}")
    logger.info(f"One complete action cycle: {total_steps} steps")
    logger.info(f"Action cycles to complete: {action_cycles}")
    logger.info("=" * 60)

    for step in range(steps):
        # Initialize policy actions with proper defaults:
        # - Base DOFs: 0.0 (middle of range, neutral position)
        # - Finger DOFs: -1.0 (minimum of range, closed/contracted position)
        actions[:] = 0.0  # Initialize all to 0 first

        # Set finger actions to -1.0 (closed position) as default
        if env.policy_controls_fingers:
            # Find where finger actions start in the action space
            finger_start_idx = (
                env.action_processor.NUM_BASE_DOFS
                if env.policy_controls_hand_base
                else 0
            )
            finger_end_idx = (
                finger_start_idx + env.action_processor.NUM_ACTIVE_FINGER_DOFS
            )
            actions[:, finger_start_idx:finger_end_idx] = -1.0

        # Initialize action tracking variables (needed for progress display)
        finger_actions_count = env.action_processor.NUM_ACTIVE_FINGER_DOFS
        base_actions_count = env.action_processor.NUM_BASE_DOFS

        # Use global step count to determine action (cycles through all actions)
        current_action_idx = step // steps_per_action
        step_in_action = step % steps_per_action
        action_value = 0.0  # Default for rule-based control

        # Also get episode step for reset detection
        episode_step = env.episode_step_count[0].item()

        # Handle concurrent vs sequential testing
        if env.policy_controls_hand_base and env.policy_controls_fingers:
            # CONCURRENT TESTING: Test base and fingers together
            base_action_idx = current_action_idx % base_actions_count
            finger_action_idx = current_action_idx % finger_actions_count

            # Base action pattern: 0 → -1 → 1 → 0 over 100 steps
            if step_in_action < 25:
                base_progress = step_in_action / 24.0
                base_action_value = 0.0 - base_progress
            elif step_in_action < 75:
                base_progress = (step_in_action - 25) / 49.0
                base_action_value = -1.0 + 2.0 * base_progress
            else:
                base_progress = (step_in_action - 75) / 24.0
                base_action_value = 1.0 - base_progress

            # Finger action pattern: -1 → 1 → -1 over 100 steps
            if step_in_action < 50:
                finger_progress = step_in_action / 49.0
                finger_action_value = -1.0 + 2.0 * finger_progress
            else:
                finger_progress = (step_in_action - 50) / 49.0
                finger_action_value = 1.0 - 2.0 * finger_progress

            # Apply base action to all environments
            actions[:, base_action_idx] = base_action_value * 0.5  # Scale base actions

            # Apply finger action to all environments
            actions[:, base_actions_count + finger_action_idx] = finger_action_value

            # Debug: CPU/GPU multi-environment issue (see roadmap.md #5)
            if step % 200 == 50 and step > 0 and env.num_envs > 1:
                logger.debug(f"Step {step}: Multi-env debug - Device: {env.device}")
                logger.debug(
                    f"  Action shape: {actions.shape}, identical values: {torch.allclose(actions[0], actions[1])}"
                )
                if hasattr(env, "dof_pos"):
                    for env_idx in range(min(env.num_envs, 2)):
                        pos = env.dof_pos[env_idx, :3]  # First 3 DOFs
                        logger.debug(
                            f"  Env {env_idx} DOF pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]"
                        )

        else:
            # SEQUENTIAL TESTING: Original logic
            if (
                env.policy_controls_hand_base
                and current_action_idx < base_actions_count
            ):
                # Base joints: 0 → -1 → 1 → 0 pattern
                if step_in_action < 25:
                    progress = step_in_action / 24.0
                    action_value = 0.0 - progress
                elif step_in_action < 75:
                    progress = (step_in_action - 25) / 49.0
                    action_value = -1.0 + 2.0 * progress
                else:
                    progress = (step_in_action - 75) / 24.0
                    action_value = 1.0 - progress

                # Apply base action to all environments
                actions[:, current_action_idx] = action_value * 0.5

            elif env.policy_controls_fingers:
                # Finger joints: -1 → 1 → -1 pattern
                if step_in_action < 50:
                    progress = step_in_action / 49.0
                    action_value = -1.0 + 2.0 * progress
                else:
                    progress = (step_in_action - 50) / 49.0
                    action_value = 1.0 - 2.0 * progress

                # Apply finger action
                finger_action_idx = (
                    current_action_idx
                    if not env.policy_controls_hand_base
                    else current_action_idx - base_actions_count
                )
                if 0 <= finger_action_idx < finger_actions_count:
                    action_start_idx = (
                        base_actions_count if env.policy_controls_hand_base else 0
                    )
                    actions[:, action_start_idx + finger_action_idx] = action_value

        # Step the simulation (rule-based control is applied automatically in pre_physics_step)
        if step < 5:
            logger.debug(
                f"[Test] Before step {step}: episode_step_count = {env.episode_step_count[0].item()}, reset_buf = {env.reset_buf[0].item()}"
            )
        obs, rewards, dones, info = env.step(actions)
        if step < 5:
            logger.debug(
                f"[Test] After step {step}: episode_step_count = {env.episode_step_count[0].item()}, reset_buf = {env.reset_buf[0].item()}, done = {dones[0].item()}"
            )

        # Check if environment was reset
        if episode_step == 0 and step > 0:
            logger.info(f"Environment reset detected at step {step}")
            # Check actual DOF positions after reset
            if hasattr(env, "dof_pos"):
                current_base_dofs = env.dof_pos[0, :6]
                logger.info(
                    f"Base DOF positions after reset: {current_base_dofs.tolist()}"
                )

        env.render()

        # Log observation data for plotting
        if plotting_enabled:
            rr.set_time_sequence("step", step)
            plot_env_idx = cfg.get("plotEnvIdx", 0)
            log_observation_data(env, step, cfg, plot_env_idx, debug)

            # Log reward data
            # Get reward components from info if available
            reward_components = info.get("reward_components", {}) if info else {}
            log_reward_data(env, rewards, reward_components, step, plot_env_idx)

        # Print progress every 50 steps and at key transitions
        if step_in_action % 50 == 0 or step_in_action == 49 or step_in_action == 99:
            # Add random actions indicator to progress output
            random_mode_indicator = " [RANDOM]" if env.random_actions_enabled else ""

            if env.policy_controls_hand_base and env.policy_controls_fingers:
                # CONCURRENT MODE: Show both base and finger actions
                base_idx = current_action_idx % base_actions_count
                finger_idx = current_action_idx % finger_actions_count
                base_name = base_action_map[base_idx][0]
                finger_name = finger_action_map[finger_idx][0]
                base_val = actions[0, base_idx].item()
                finger_val = actions[0, base_actions_count + finger_idx].item()

                logger.info(
                    f"  Step {step+1:4d}: Action {current_action_idx} - Base: {base_name:>6}={base_val:+5.2f}, Finger: {finger_name:>13}={finger_val:+5.2f} (substep {step_in_action+1:2d}/100){random_mode_indicator}"
                )
            elif (
                env.policy_controls_hand_base
                and current_action_idx < base_actions_count
            ):
                # BASE ONLY MODE
                action_name = base_action_map[current_action_idx][0]
                action_sent = actions[0, current_action_idx].item()
                logger.info(
                    f"  Step {step+1:4d}: Base Action {current_action_idx} ({action_name:>13}) = {action_sent:+6.3f} (substep {step_in_action+1:2d}/100){random_mode_indicator}"
                )
            elif env.policy_controls_fingers:
                # FINGER ONLY MODE
                finger_idx = (
                    current_action_idx
                    if not env.policy_controls_hand_base
                    else current_action_idx - base_actions_count
                )
                if 0 <= finger_idx < finger_actions_count:
                    action_name = finger_action_map[finger_idx][0]
                    action_sent = actions[
                        0,
                        (base_actions_count if env.policy_controls_hand_base else 0)
                        + finger_idx,
                    ].item()
                    logger.info(
                        f"  Step {step+1:4d}: Finger Action {finger_idx} ({action_name:>13}) = {action_sent:+6.3f} (substep {step_in_action+1:2d}/100){random_mode_indicator}"
                    )
            else:
                logger.info(
                    f"  Step {step+1:4d}: Test in progress (substep {step_in_action+1:2d}/100){random_mode_indicator}"
                )

            # At transitions, show DOF changes for the current action (debug level only)
            if (
                debug
                and hasattr(env, "dof_pos")
                and (
                    step_in_action == 0 or step_in_action == 49 or step_in_action == 99
                )
            ):
                current_dof_pos = env.dof_pos[0]
                dof_change = current_dof_pos - initial_dof_pos

                # Show changes for the joints controlled by this action
                if env.policy_controls_hand_base and env.policy_controls_fingers:
                    # Concurrent mode - show both base and finger changes
                    base_idx = current_action_idx % base_actions_count
                    finger_idx = current_action_idx % finger_actions_count
                    base_name, base_desc = base_action_map[base_idx]
                    finger_name, finger_desc = finger_action_map[finger_idx]
                    logger.debug(f"    Base: {base_desc}, Finger: {finger_desc}")
                elif env.policy_controls_fingers and current_action_idx < len(
                    finger_action_map
                ):
                    action_name, description = finger_action_map[current_action_idx]
                    logger.debug(f"    {description}")

                    # Show finger DOF changes
                    finger_changes = dof_change[6:]  # Skip base DOFs
                    max_finger_change = torch.max(torch.abs(finger_changes)).item()

                    # Find the most changed finger DOF
                    if max_finger_change > 1e-6:
                        max_finger_idx = torch.argmax(torch.abs(finger_changes)).item()
                        finger_dof_name = finger_dofs[max_finger_idx]
                        finger_change_value = finger_changes[max_finger_idx].item()
                        logger.debug(
                            f"    Max finger DOF change: {finger_dof_name} = {finger_change_value:+.6f}"
                        )

                    # Show coupling verification for specific actions
                    if (
                        current_action_idx == 2
                    ):  # thumb_dip (should affect joints 1_3 and 1_4)
                        joint1_3_change = finger_changes[
                            2
                        ].item()  # r_f_joint1_3 (index 8-6=2)
                        joint1_4_change = finger_changes[
                            3
                        ].item()  # r_f_joint1_4 (index 9-6=3)
                        logger.debug(
                            f"    Coupling check: r_f_joint1_3={joint1_3_change:+.6f}, r_f_joint1_4={joint1_4_change:+.6f}"
                        )
                        if abs(joint1_3_change - joint1_4_change) < 1e-5:
                            logger.debug(
                                "    ✓ Coupling verified: both joints move together"
                            )
                        else:
                            logger.warning(
                                "    ⚠ Coupling issue: joints should move together"
                            )

                    elif (
                        current_action_idx == 3
                    ):  # finger_spread (should affect 2_1, 4_1, 5_1 with 5_1=2x)
                        joint2_1_change = finger_changes[
                            4
                        ].item()  # r_f_joint2_1 (index 10-6=4)
                        joint4_1_change = finger_changes[
                            12
                        ].item()  # r_f_joint4_1 (index 18-6=12)
                        joint5_1_change = finger_changes[
                            16
                        ].item()  # r_f_joint5_1 (index 22-6=16)
                        joint3_1_change = finger_changes[
                            8
                        ].item()  # r_f_joint3_1 (index 14-6=8, should stay 0)
                        logger.debug(
                            f"    Spread coupling: 2_1={joint2_1_change:+.6f}, 4_1={joint4_1_change:+.6f}, 5_1={joint5_1_change:+.6f}"
                        )
                        logger.debug(
                            f"    Fixed joint 3_1: {joint3_1_change:+.6f} (should be 0)"
                        )
                        if (
                            abs(joint5_1_change - 2.0 * joint2_1_change) < 1e-5
                            and abs(joint4_1_change - joint2_1_change) < 1e-5
                        ):
                            logger.debug(
                                "    ✓ Spread coupling verified: 5_1 ≈ 2×(2_1,4_1)"
                            )
                        else:
                            logger.warning("    ⚠ Spread coupling issue")

        sleep_time = cfg.get("sleep", 0.01)
        time.sleep(sleep_time)

    logger.info("\nSequential action test completed!")
    logger.info("All 12 actions have been tested with the -1 → 1 → -1 pattern.")

    # Final DOF state summary
    if hasattr(env, "dof_pos"):
        final_dof_pos = env.dof_pos[0]
        final_dof_change = final_dof_pos - initial_dof_pos
        final_finger_changes = final_dof_change[6:]
        max_final_change = torch.max(torch.abs(final_finger_changes)).item()

        logger.info("\nFinal finger DOF changes from initial state:")
        for i, name in enumerate(finger_dofs):
            change = final_finger_changes[i].item()
            logger.info(f"  {name}: {change:+.6f}")

        logger.info(f"Maximum final change: {max_final_change:.6f}")
        if max_final_change < 0.1:
            logger.info("✓ Hand returned close to initial position (good)")
        else:
            logger.warning("⚠ Hand position significantly changed from initial")

    logger.info("Test completed")


if __name__ == "__main__":
    main()
