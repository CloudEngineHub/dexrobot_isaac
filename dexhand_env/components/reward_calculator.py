"""
Reward calculator component for DexHand environment.

This module provides functionality to calculate rewards for dexterous
manipulation tasks.
"""

# Import PyTorch
import torch


class RewardCalculator:
    """
    Calculates rewards for dexterous manipulation tasks.

    This component provides functionality to:
    - Compute common reward components
    - Apply weights to reward components
    - Combine task-specific and common rewards
    """

    def __init__(self, parent, cfg):
        """
        Initialize the reward calculator.

        Args:
            parent: Parent DexHandBase instance
            cfg: Configuration dictionary
        """
        self.parent = parent

        # Initialize reward weights from config
        self.reward_weights = cfg["env"].get("rewardWeights", {})

    @property
    def num_envs(self):
        """Access num_envs from parent (single source of truth)."""
        return self.parent.num_envs

    @property
    def device(self):
        """Access device from parent (single source of truth)."""
        return self.parent.device

    def compute_common_reward_terms(
        self,
        obs_dict,
        hand_vel,
        hand_ang_vel,
        dof_vel,
        dof_pos,
        dof_lower_limits,
        dof_upper_limits,
        prev_dof_vel,
        prev_hand_vel,
        prev_hand_ang_vel,
        prev_contacts,
    ):
        """
        Compute common reward components available to all tasks.

        This includes:
        1. alive - Small constant reward for each timestep (staying alive)
        2. height_safety - Penalizes when fingertips get too close to the ground
        3. Velocity penalties
        4. Joint limit penalties
        5. Acceleration penalties
        6. Contact stability

        Args:
            obs_dict: Dictionary of observations
            hand_vel: Hand linear velocity
            hand_ang_vel: Hand angular velocity
            dof_vel: DOF velocities
            dof_pos: DOF positions
            dof_lower_limits: Lower joint limits
            dof_upper_limits: Upper joint limits
            prev_dof_vel: Previous DOF velocities
            prev_hand_vel: Previous hand linear velocity
            prev_hand_ang_vel: Previous hand angular velocity
            prev_contacts: Previous contact state

        Returns:
            Dictionary of common reward components
        """
        rewards = {}

        # Add alive bonus - small reward for each step the agent survives
        rewards["alive"] = torch.ones(self.num_envs, device=self.device)

        # 1. Height safety reward: penalize when fingertips get too close to the ground
        min_height = 0.02  # Minimum safe height above ground

        # Extract fingertip positions from fingertip_poses_world
        # fingertip_poses_world is (num_envs, 35) where each fingertip has 7 values (x,y,z,qx,qy,qz,qw)
        # 5 fingertips * 7 values = 35
        fingertip_poses = obs_dict.get("fingertip_poses_world", None)
        if fingertip_poses is not None:
            # Reshape to (num_envs, 5, 7) and extract z coordinates
            fingertip_poses_reshaped = fingertip_poses.view(self.num_envs, 5, 7)
            fingertip_heights = fingertip_poses_reshaped[:, :, 2]  # Z coordinates
            min_fingertip_height = torch.min(fingertip_heights, dim=1)[0]
        else:
            # Fallback if not available
            min_fingertip_height = torch.ones(self.num_envs, device=self.device) * 0.1
        height_safety = torch.clamp(
            1.0 - torch.exp(-(min_fingertip_height - min_height) * 20), 0.0, 1.0
        )
        rewards["height_safety"] = height_safety

        # 2. Velocity penalties: penalize high velocities
        # 2a. Finger velocity penalty: penalize high finger joint velocities
        finger_vel_norm = torch.norm(dof_vel, dim=1)
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

        # 3. Joint limit penalty: penalize when joints are close to their limits
        # Normalize joint positions to [-1, 1] range based on limits
        joint_ranges = dof_upper_limits - dof_lower_limits
        normalized_joints = 2.0 * (dof_pos - dof_lower_limits) / joint_ranges - 1.0
        # Penalize when |normalized_joints| > 0.8 (i.e., within 10% of limits)
        joint_limit_margin = 0.8
        joint_limit_penalties = torch.clamp(
            torch.abs(normalized_joints) - joint_limit_margin, 0.0, 1.0
        )
        joint_limit_penalty = torch.sum(joint_limit_penalties, dim=1) / dof_pos.shape[1]
        rewards["joint_limit"] = 1.0 - joint_limit_penalty

        # 4. Acceleration penalties: penalize rapid changes in velocities
        # 4a. Finger acceleration penalty: penalize rapid changes in finger joint velocities
        finger_acc = dof_vel - prev_dof_vel
        finger_acc_norm = torch.norm(finger_acc, dim=1)
        finger_acc_penalty = torch.exp(-2.0 * finger_acc_norm)
        rewards["finger_acceleration"] = finger_acc_penalty

        # 4b. Hand acceleration penalty: penalize rapid changes in hand velocity
        hand_accel = torch.norm(hand_vel - prev_hand_vel, dim=1)
        hand_acc_penalty = torch.exp(-0.5 * hand_accel)
        rewards["hand_acceleration"] = hand_acc_penalty

        # 4c. Hand angular acceleration penalty: penalize rapid changes in hand angular velocity
        hand_ang_accel = torch.norm(hand_ang_vel - prev_hand_ang_vel, dim=1)
        hand_ang_acc_penalty = torch.exp(-0.5 * hand_ang_accel)
        rewards["hand_angular_acceleration"] = hand_ang_acc_penalty

        # 5. Contact stability: reward consistent contacts
        if "contact_forces" in obs_dict and prev_contacts is not None:
            # Calculate contact state (boolean tensor: is each fingertip touching something?)
            # contact_forces is flattened to (num_envs, num_bodies * 3)
            # Reshape it back to (num_envs, num_bodies, 3) for norm calculation
            contact_forces_flat = obs_dict["contact_forces"]
            num_bodies = contact_forces_flat.shape[1] // 3
            contact_forces_3d = contact_forces_flat.view(self.num_envs, num_bodies, 3)
            contact_force_norm = torch.norm(contact_forces_3d, dim=2)
            contacts = contact_force_norm > 0.1

            # Compute changes in contact state
            contact_changes = torch.sum(
                torch.logical_xor(contacts, prev_contacts).float(), dim=1
            )
            contact_stability = torch.exp(-contact_changes)
            rewards["contact_stability"] = contact_stability
        else:
            # No contact info available yet, assume stable contacts
            rewards["contact_stability"] = torch.ones(self.num_envs, device=self.device)

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
