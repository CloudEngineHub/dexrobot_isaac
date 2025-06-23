"""
Grasping task for DexHand.

This module provides a task implementation for grasping objects with the DexHand.
"""

import numpy as np
from typing import Dict, Optional

# Import IsaacGym first
from isaacgym import gymapi

# Then import PyTorch
import torch

from dexhand_env.tasks.base_task import BaseTask
from dexhand_env.utils.coordinate_transforms import point_in_hand_frame


class DexGraspTask(BaseTask):
    """
    Grasping task for DexHand.

    The task is to grasp an object and lift it to a target position.
    Success is measured by stably lifting the object to the target position.
    """

    def __init__(self, sim, gym, device, num_envs, cfg):
        """
        Initialize the grasping task.

        Args:
            sim: Simulation instance
            gym: Gym instance
            device: PyTorch device
            num_envs: Number of environments
            cfg: Configuration dictionary
        """
        super().__init__(sim, gym, device, num_envs, cfg)

        # Target parameters
        self.target_pos = torch.tensor(
            self.cfg["task"].get("targetPos", [0.0, 0.0, 0.3]), device=self.device
        )
        self.target_radius = self.cfg["task"].get("targetRadius", 0.05)

        # Object parameters
        self.object_type = self.cfg["task"].get("objectType", "cube")
        self.object_size = self.cfg["task"].get("objectSize", 0.05)
        self.object_mass = self.cfg["task"].get("objectMass", 0.1)

        # Initial object position range
        self.object_pos_init_x = self.cfg["task"].get("objectPosInitX", [0.0, 0.0])
        self.object_pos_init_y = self.cfg["task"].get("objectPosInitY", [0.0, 0.0])
        self.object_pos_init_z = self.cfg["task"].get("objectPosInitZ", [0.05, 0.05])

        # Success criteria
        self.lift_height_threshold = self.cfg["task"].get("liftHeightThreshold", 0.2)
        self.stable_grasp_threshold = self.cfg["task"].get("stableGraspThreshold", 2.0)
        self.grasp_stability_window = self.cfg["task"].get("graspStabilityWindow", 50)

        # Reward scales
        self.reach_object_scale = self.cfg["task"].get("reachObjectScale", 5.0)
        self.grasp_reward_scale = self.cfg["task"].get("graspRewardScale", 10.0)
        self.lift_reward_scale = self.cfg["task"].get("liftRewardScale", 15.0)
        self.target_reward_scale = self.cfg["task"].get("targetRewardScale", 20.0)

        # Counters for success criteria
        self.stable_grasp_counts = torch.zeros(self.num_envs, device=self.device)

        # Asset handles and actor handles
        self.object_asset = None
        self.object_handles = []

        # Initialize object state
        self.object_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.object_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_angvel = torch.zeros((self.num_envs, 3), device=self.device)

        # Initial object positions for resets
        self.initial_object_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.initial_object_rot = torch.zeros((self.num_envs, 4), device=self.device)

        # Configure target visualization
        self.target_visual_size = 0.02  # Size of target visualization sphere
        self.target_handles = []

        # Track number of actors per environment
        self.num_task_actors = 2  # Object + target visualization

        # Reference to root state tensor (will be set by environment)
        self.root_state_tensor = None

    def load_task_assets(self):
        """
        Load task-specific assets.

        Loads the object asset based on the configured object type.
        """
        # Create object asset options
        asset_options = gymapi.AssetOptions()
        asset_options.density = 1000  # kg/m^3
        asset_options.fix_base_link = False
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.disable_gravity = False

        # Create object asset based on object type
        if self.object_type == "cube":
            # Create cube asset
            self.object_asset = self.gym.create_box(
                self.sim,
                self.object_size,
                self.object_size,
                self.object_size,
                asset_options,
            )
        elif self.object_type == "sphere":
            # Create sphere asset
            self.object_asset = self.gym.create_sphere(
                self.sim, self.object_size / 2, asset_options  # Radius
            )
        else:
            raise ValueError(f"Unknown object type: {self.object_type}")

    def create_task_objects(self, gym, sim, env_ptr, env_id: int):
        """
        Add task-specific objects to the environment.

        Adds the object and target visualization to the environment.
        Note: This is called AFTER hands are created, so hands are always actor 0.

        Args:
            gym: Gym instance
            sim: Simulation instance
            env_ptr: Pointer to the environment to add objects to
            env_id: Index of the environment being created
        """
        # Set initial object pose with randomization
        object_pos_x = np.random.uniform(
            self.object_pos_init_x[0], self.object_pos_init_x[1]
        )
        object_pos_y = np.random.uniform(
            self.object_pos_init_y[0], self.object_pos_init_y[1]
        )
        object_pos_z = np.random.uniform(
            self.object_pos_init_z[0], self.object_pos_init_z[1]
        )

        # Store initial object positions for reset
        self.initial_object_pos[env_id] = torch.tensor(
            [object_pos_x, object_pos_y, object_pos_z], device=self.device
        )
        self.initial_object_rot[env_id] = torch.tensor([0, 0, 0, 1], device=self.device)

        # Create object
        object_pose = gymapi.Transform()
        object_pose.p = gymapi.Vec3(object_pos_x, object_pos_y, object_pos_z)
        object_pose.r = gymapi.Quat(0, 0, 0, 1)

        # Add object to environment
        object_handle = self.gym.create_actor(
            env_ptr, self.object_asset, object_pose, f"object_{env_id}", env_id, 0
        )

        # Set object properties
        props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
        props[0].mass = self.object_mass
        self.gym.set_actor_rigid_body_properties(
            env_ptr, object_handle, props, relink=True
        )

        # Set collision group to enable object-hand interaction
        self.gym.set_rigid_body_segmentation_id(env_ptr, object_handle, 0, 1)

        # Store object handle
        self.object_handles.append(object_handle)

        # Create target visualization
        target_pose = gymapi.Transform()
        target_pose.p = gymapi.Vec3(
            self.target_pos[0], self.target_pos[1], self.target_pos[2]
        )

        target_asset = self.gym.create_sphere(
            self.sim, self.target_visual_size, gymapi.AssetOptions()  # Radius
        )

        target_handle = self.gym.create_actor(
            env_ptr,
            target_asset,
            target_pose,
            f"target_{env_id}",
            env_id,
            1,  # Set to different collision group
        )

        # Make target visualization invisible to collisions
        self.gym.set_rigid_body_collision_filter(env_ptr, target_handle, 0, 0)

        # Set target color (green)
        self.gym.set_rigid_body_color(
            env_ptr, target_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.0, 1.0, 0.0)
        )

        # Store target handle
        self.target_handles.append(target_handle)

    def get_task_observations(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get task-specific observations.

        Returns:
            Dictionary of task-specific observations
        """
        # Get object state in hand frame
        object_pos_hand_frame = point_in_hand_frame(
            self.object_pos, self.hand_pos, self.hand_rot
        )

        # Compute relative position to target
        object_to_target = self.target_pos - self.object_pos

        # Return task-specific observations
        return {
            "object_pos": self.object_pos,
            "object_rot": self.object_rot,
            "object_vel": self.object_vel,
            "object_angvel": self.object_angvel,
            "object_pos_hand_frame": object_pos_hand_frame,
            "object_to_target": object_to_target,
            "target_pos": self.target_pos.repeat(self.num_envs, 1),
        }

    def reset_task_state(self, env_ids: torch.Tensor):
        """
        Reset task-specific state for the specified environments.

        Args:
            env_ids: Environment indices to reset
        """
        # Reset object pose to initial values with randomization
        if len(env_ids) == 0:
            return

        # Randomize initial object positions
        rand_pos_x = (
            torch.rand(len(env_ids), device=self.device)
            * (self.object_pos_init_x[1] - self.object_pos_init_x[0])
            + self.object_pos_init_x[0]
        )
        rand_pos_y = (
            torch.rand(len(env_ids), device=self.device)
            * (self.object_pos_init_y[1] - self.object_pos_init_y[0])
            + self.object_pos_init_y[0]
        )
        rand_pos_z = (
            torch.rand(len(env_ids), device=self.device)
            * (self.object_pos_init_z[1] - self.object_pos_init_z[0])
            + self.object_pos_init_z[0]
        )

        # Create randomized positions
        self.initial_object_pos[env_ids, 0] = rand_pos_x
        self.initial_object_pos[env_ids, 1] = rand_pos_y
        self.initial_object_pos[env_ids, 2] = rand_pos_z

        # Reset object state in tensor
        # Since hands are created first (actor 0), object is actor 1
        object_actor_ids = 1  # Object is the second actor (hand is 0)
        self.root_state_tensor[
            env_ids, object_actor_ids, 0:3
        ] = self.initial_object_pos[env_ids]
        self.root_state_tensor[
            env_ids, object_actor_ids, 3:7
        ] = self.initial_object_rot[env_ids]
        self.root_state_tensor[env_ids, object_actor_ids, 7:10] = torch.zeros_like(
            self.root_state_tensor[env_ids, object_actor_ids, 7:10]
        )
        self.root_state_tensor[env_ids, object_actor_ids, 10:13] = torch.zeros_like(
            self.root_state_tensor[env_ids, object_actor_ids, 10:13]
        )

        # Reset counters
        self.stable_grasp_counts[env_ids] = 0

    def compute_task_reward_terms(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute task-specific reward terms.

        Rewards are given for:
        1. Reaching the object
        2. Grasping the object
        3. Lifting the object
        4. Moving the object to the target position

        Args:
            obs_dict: Dictionary of observations

        Returns:
            Dictionary of task-specific reward terms
        """
        # Extract relevant observations
        object_pos = obs_dict["object_pos"]
        fingertip_pos = obs_dict["fingertip_pos"]

        # Update object state for tracking
        self.object_pos = object_pos
        self.object_rot = obs_dict["object_rot"]
        self.object_vel = obs_dict["object_vel"]
        self.object_angvel = obs_dict["object_angvel"]

        # Store hand position and rotation for coordinate transforms
        self.hand_pos = obs_dict["hand_pos"]
        self.hand_rot = obs_dict["hand_rot"]

        # Compute reaching reward - distance from fingertips to object
        fingertip_to_object = fingertip_pos.view(
            self.num_envs, 5, 3
        ) - object_pos.unsqueeze(1)
        fingertip_dist = torch.norm(fingertip_to_object, dim=2)
        closest_fingertip_dist = torch.min(fingertip_dist, dim=1)[0]
        reach_reward = torch.exp(-closest_fingertip_dist * 10.0)

        # Compute grasping reward - based on contact forces (assumed to be in obs_dict)
        contact_forces = obs_dict["contact_forces"]
        contact_force_mag = torch.sum(torch.norm(contact_forces, dim=2), dim=1)
        grasp_reward = torch.tanh(contact_force_mag / 10.0)

        # Check if object is lifted above threshold
        is_lifted = object_pos[:, 2] > self.lift_height_threshold
        lift_reward = torch.where(
            is_lifted,
            torch.ones_like(is_lifted, dtype=torch.float) * self.lift_reward_scale,
            torch.zeros_like(is_lifted, dtype=torch.float),
        )

        # Compute distance to target
        object_to_target = self.target_pos - object_pos
        target_dist = torch.norm(object_to_target, dim=1)
        target_reward = torch.exp(-target_dist * 10.0)

        # Compute combined reward terms
        rewards = {
            "reach_reward": reach_reward * self.reach_object_scale,
            "grasp_reward": grasp_reward * self.grasp_reward_scale,
            "lift_reward": lift_reward,
            "target_reward": target_reward * self.target_reward_scale,
        }

        return rewards

    def check_task_success_criteria(self) -> Dict[str, torch.Tensor]:
        """
        Check task-specific success criteria.

        Success is defined as the object being stably lifted near the target position.

        Returns:
            Dictionary of task-specific success criteria
        """
        # Compute distance to target
        object_to_target = self.target_pos - self.object_pos
        target_dist = torch.norm(object_to_target, dim=1)

        # Check if object is at target position
        at_target = target_dist < self.target_radius

        # Check if object is lifted above threshold
        is_lifted = self.object_pos[:, 2] > self.lift_height_threshold

        # Check if object is stably grasped (low velocity)
        object_vel_norm = torch.norm(self.object_vel, dim=1)
        object_angvel_norm = torch.norm(self.object_angvel, dim=1)
        is_stable = (object_vel_norm < 0.1) & (object_angvel_norm < 0.5)

        # Update stability counter
        self.stable_grasp_counts = torch.where(
            is_stable & is_lifted,
            self.stable_grasp_counts + 1,
            torch.zeros_like(self.stable_grasp_counts),
        )

        # Success criteria: lifted to target position and stably grasped for enough steps
        stable_success = self.stable_grasp_counts > self.grasp_stability_window
        task_success = at_target & is_lifted & stable_success

        return {"task_success": task_success}

    def check_task_failure_criteria(self) -> Dict[str, torch.Tensor]:
        """
        Check task-specific failure criteria.

        For this task, there are no specific failure conditions besides timeout,
        which is handled by the base environment.

        Returns:
            Empty dictionary of task-specific failure criteria
        """
        return {}
