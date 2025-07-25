"""
Reward calculator component for DexHand environment.

This module provides functionality to calculate rewards for dexterous
manipulation tasks.
"""

# Import PyTorch
import torch

# Import constants
from dexhand_env.constants import NUM_BASE_DOFS


class RewardCalculator:
    """
    Calculates rewards for dexterous manipulation tasks.

    This component provides functionality to:
    - Compute common reward components
    - Apply weights to reward components
    - Combine task-specific and common rewards
    """

    def __init__(self, parent, task_cfg):
        """
        Initialize the reward calculator.

        Args:
            parent: Parent DexHandBase instance
            task_cfg: Task-specific configuration dictionary
        """
        self.parent = parent
        self.task_cfg = task_cfg

        # Initialize reward weights from config
        self.reward_weights = task_cfg.get("rewardWeights", {})

        # Import loguru for debug logging
        from loguru import logger

        logger.info(
            f"RewardCalculator initialized with {len(self.reward_weights)} reward weights"
        )
        if len(self.reward_weights) == 0:
            logger.warning(
                "No reward weights loaded! This will result in zero rewards."
            )

        # State tracking for acceleration and contact stability rewards
        self.prev_finger_dof_vel = None
        self.prev_hand_vel = None
        self.prev_hand_ang_vel = None
        self.prev_contacts = None

    @property
    def num_envs(self):
        """Access num_envs from parent (single source of truth)."""
        return self.parent.num_envs

    @property
    def device(self):
        """Access device from parent (single source of truth)."""
        return self.parent.device

    def compute_common_reward_terms(self, obs_dict, parent_env):
        """
        Compute common reward components available to all tasks.

        This includes:
        1. alive - Small constant reward for each timestep (staying alive)
        2. height_safety - Penalizes when fingertips get too close to the ground
        3. Velocity penalties (separated: finger vs hand)
        4. Joint limit penalties (finger joints only)
        5. Acceleration penalties (separated: finger vs hand)
        6. Contact stability

        Args:
            obs_dict: Dictionary of observations
            parent_env: Parent DexHandBase instance for accessing simulation state

        Returns:
            Dictionary of common reward components
        """
        rewards = {}

        # Get current simulation state from parent environment
        hand_vel = parent_env.rigid_body_states[
            :, parent_env.hand_local_rigid_body_index, 7:10
        ]
        hand_ang_vel = parent_env.rigid_body_states[
            :, parent_env.hand_local_rigid_body_index, 10:13
        ]
        dof_vel = parent_env.dof_vel
        dof_pos = parent_env.dof_pos

        # Separate finger DOFs from base DOFs
        finger_dof_vel = dof_vel[:, NUM_BASE_DOFS:]  # Finger DOFs only (indices 6+)
        finger_dof_pos = dof_pos[:, NUM_BASE_DOFS:]  # Finger DOFs only

        # Get DOF limits for finger joints only (base DOFs don't have meaningful limits)
        finger_dof_lower_limits = parent_env.tensor_manager.dof_props[NUM_BASE_DOFS:, 4]
        finger_dof_upper_limits = parent_env.tensor_manager.dof_props[NUM_BASE_DOFS:, 5]

        # Add alive bonus - small reward for each step the agent survives
        rewards["alive"] = torch.ones(self.num_envs, device=self.device)

        # 1. Height safety reward: penalize when fingertips get too close to the ground
        min_height = 0.02  # Minimum safe height above ground

        # Extract fingertip positions from fingertip_poses_world (FAIL FAST - no fallbacks)
        # fingertip_poses_world is (num_envs, 35) where each fingertip has 7 values (x,y,z,qx,qy,qz,qw)
        # 5 fingertips * 7 values = 35
        fingertip_poses = obs_dict["fingertip_poses_world"]  # Fail if missing
        # Reshape to (num_envs, 5, 7) and extract z coordinates
        fingertip_poses_reshaped = fingertip_poses.view(self.num_envs, 5, 7)
        fingertip_heights = fingertip_poses_reshaped[:, :, 2]  # Z coordinates
        min_fingertip_height = torch.min(fingertip_heights, dim=1)[0]
        height_safety = torch.clamp(
            1.0 - torch.exp(-(min_fingertip_height - min_height) * 20), 0.0, 1.0
        )
        rewards["height_safety"] = height_safety

        # 2. Velocity penalties: penalize high velocities (separated finger vs hand)
        # 2a. Finger velocity penalty: penalize high finger joint velocities ONLY
        finger_vel_norm = torch.norm(finger_dof_vel, dim=1)
        finger_vel_penalty = torch.exp(-0.1 * finger_vel_norm)
        rewards["finger_velocity"] = finger_vel_penalty

        # 2b. Hand velocity penalty: penalize high hand movement speed
        hand_vel_norm = torch.norm(hand_vel, dim=1)
        hand_vel_penalty = torch.exp(-0.2 * hand_vel_norm)
        rewards["hand_velocity"] = hand_vel_penalty

        # 2c. Hand angular velocity penalty: penalize high rotational speed
        hand_ang_vel_norm = torch.norm(hand_ang_vel, dim=1)
        hand_ang_vel_penalty = torch.exp(-0.2 * hand_ang_vel_norm)
        rewards["hand_angular_velocity"] = hand_ang_vel_penalty

        # 3. Joint limit penalty: penalize when FINGER joints are close to their limits
        # (Base DOFs don't have meaningful limits, so exclude them)
        finger_joint_ranges = finger_dof_upper_limits - finger_dof_lower_limits
        normalized_finger_joints = (
            2.0 * (finger_dof_pos - finger_dof_lower_limits) / finger_joint_ranges - 1.0
        )
        # Penalize when |normalized_joints| > 0.8 (i.e., within 10% of limits)
        joint_limit_margin = 0.8
        joint_limit_penalties = torch.clamp(
            torch.abs(normalized_finger_joints) - joint_limit_margin, 0.0, 1.0
        )
        joint_limit_penalty = (
            torch.sum(joint_limit_penalties, dim=1) / finger_dof_pos.shape[1]
        )
        rewards["joint_limit"] = 1.0 - joint_limit_penalty

        # 4. Acceleration penalties: penalize rapid changes in velocities
        # Initialize previous states on first call
        if self.prev_finger_dof_vel is None:
            self.prev_finger_dof_vel = finger_dof_vel.clone()
            self.prev_hand_vel = hand_vel.clone()
            self.prev_hand_ang_vel = hand_ang_vel.clone()

        # 4a. Finger acceleration penalty: penalize rapid changes in FINGER joint velocities ONLY
        finger_acc = finger_dof_vel - self.prev_finger_dof_vel
        finger_acc_norm = torch.norm(finger_acc, dim=1)
        finger_acc_penalty = torch.exp(-2.0 * finger_acc_norm)
        rewards["finger_acceleration"] = finger_acc_penalty

        # 4b. Hand acceleration penalty: penalize rapid changes in hand velocity
        hand_accel = torch.norm(hand_vel - self.prev_hand_vel, dim=1)
        hand_acc_penalty = torch.exp(-0.5 * hand_accel)
        rewards["hand_acceleration"] = hand_acc_penalty

        # 4c. Hand angular acceleration penalty: penalize rapid changes in hand angular velocity
        hand_ang_accel = torch.norm(hand_ang_vel - self.prev_hand_ang_vel, dim=1)
        hand_ang_acc_penalty = torch.exp(-0.5 * hand_ang_accel)
        rewards["hand_angular_acceleration"] = hand_ang_acc_penalty

        # 5. Contact stability: reward consistent contacts (FAIL FAST - no fallbacks)
        # Calculate contact state (boolean tensor: is each fingertip touching something?)
        # contact_forces is flattened to (num_envs, num_bodies * 3)
        contact_forces_flat = obs_dict["contact_forces"]  # Fail if missing
        num_bodies = contact_forces_flat.shape[1] // 3
        contact_forces_3d = contact_forces_flat.view(self.num_envs, num_bodies, 3)
        contact_force_norm = torch.norm(contact_forces_3d, dim=2)
        contacts = contact_force_norm > 0.1

        # Initialize previous contacts on first call
        if self.prev_contacts is None:
            self.prev_contacts = contacts.clone()

        # Compute changes in contact state
        contact_changes = torch.sum(
            torch.logical_xor(contacts, self.prev_contacts).float(), dim=1
        )
        contact_stability = torch.exp(-contact_changes)
        rewards["contact_stability"] = contact_stability

        # Update previous states for next iteration
        # Use copy_() for in-place copy to avoid creating new tensors
        if self.prev_finger_dof_vel.shape == finger_dof_vel.shape:
            self.prev_finger_dof_vel.copy_(finger_dof_vel)
        else:
            self.prev_finger_dof_vel = finger_dof_vel.clone()

        if self.prev_hand_vel.shape == hand_vel.shape:
            self.prev_hand_vel.copy_(hand_vel)
        else:
            self.prev_hand_vel = hand_vel.clone()

        if self.prev_hand_ang_vel.shape == hand_ang_vel.shape:
            self.prev_hand_ang_vel.copy_(hand_ang_vel)
        else:
            self.prev_hand_ang_vel = hand_ang_vel.clone()

        if self.prev_contacts.shape == contacts.shape:
            self.prev_contacts.copy_(contacts)
        else:
            self.prev_contacts = contacts.clone()

        return rewards

    def compute_total_reward(
        self, common_rewards, task_rewards, success_failure_rewards=None
    ):
        """
        Compute total reward as weighted sum of reward components.

        This method combines common reward components with task-specific ones,
        applies configured weights, and returns both the total reward and
        individual components for logging.

        Args:
            common_rewards: Dictionary of common reward components
            task_rewards: Dictionary of task-specific reward components
            success_failure_rewards: Optional dictionary of success/failure rewards

        Returns:
            Tuple containing:
                rewards: Tensor of total rewards for each environment
                reward_components: Dictionary of reward components for logging
        """
        # Combine reward dictionaries
        all_rewards = {**common_rewards, **task_rewards}

        # Add success/failure rewards if provided
        if success_failure_rewards is not None:
            all_rewards.update(success_failure_rewards)

        # Initialize total rewards and components dictionary
        total_rewards = torch.zeros(self.num_envs, device=self.device)
        reward_components = {}

        # Apply weights to each reward component
        for name, reward in all_rewards.items():
            # Get weight from config (default to 0 if not specified)
            weight = self.reward_weights.get(name, 0.0)

            if weight != 0:
                # Apply weight
                weighted_reward = reward * weight

                # Add to total reward
                total_rewards += weighted_reward

                # Store for logging
                reward_components[
                    name
                ] = reward  # Store unweighted rewards for interpretability
                reward_components[f"{name}_weighted"] = weighted_reward

        # Add total to components for logging
        reward_components["total"] = total_rewards

        return total_rewards, reward_components
