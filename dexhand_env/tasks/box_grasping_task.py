"""
Box grasping task for DexHand.

This module implements a blind grasping task where the policy learns to grasp
a 5cm box using only tactile feedback (binary contacts and duration).
"""

from typing import Dict, Optional, Tuple

# Import PyTorch
import torch

# Import IsaacGym
from isaacgym import gymapi

# Import loguru for logging
from loguru import logger

# Import task interface
from dexhand_env.tasks.task_interface import DexTask

# Import constants


class BoxGraspingTask(DexTask):
    """
    Box grasping task implementation.

    The policy is "blind" - it only receives tactile feedback (binary contacts
    and duration), not object pose/velocity. This encourages learning robust
    grasping strategies based on touch rather than vision.

    Success criteria:
    - Object lifted to height > 20cm
    - At least 2 fingers in contact for at least 2 seconds

    Failure criteria:
    - Episode timeout (10 seconds)
    - Any fingertip/pad or hand base touches ground (handled by base)
    """

    def __init__(self, sim, gym, device, num_envs, cfg):
        """
        Initialize the box grasping task.

        Args:
            sim: Simulation instance
            gym: Gym instance
            device: PyTorch device
            num_envs: Number of environments
            cfg: Configuration dictionary
        """
        self.sim = sim
        self.gym = gym
        self.device = device
        self.num_envs = num_envs
        self.cfg = cfg

        # Reference to parent environment (set by DexHandBase)
        self.parent_env = None

        # Box configuration
        self.box_size = cfg["env"]["box"]["size"]
        self.box_mass = cfg["env"]["box"]["mass"]
        self.box_friction = cfg["env"]["box"]["friction"]
        self.box_restitution = cfg["env"]["box"]["restitution"]
        self.box_xy_range = cfg["env"]["box"]["initial_position"]["xy_range"]
        self.box_z = cfg["env"]["box"]["initial_position"]["z"]

        # Reward configuration
        self.object_height_weight = cfg["env"]["rewards"]["object_height"]["weight"]
        self.grasp_approach_weight = cfg["env"]["rewards"]["grasp_approach"]["weight"]
        self.finger_to_object_weight = cfg["env"]["rewards"]["finger_to_object"][
            "weight"
        ]
        self.hand_to_object_weight = cfg["env"]["rewards"]["hand_to_object"]["weight"]

        # Task-specific parameters from config
        task_params = cfg["env"]["task_params"]
        self.height_threshold = task_params["success_height_threshold"]
        self.contact_duration_threshold_seconds = task_params[
            "contact_duration_threshold"
        ]
        self.min_fingers_for_grasp = task_params["min_fingers_for_grasp"]
        self.max_box_distance = task_params["max_box_distance"]
        self.density_conversion_factor = task_params["density_conversion_factor"]
        self.grasp_not_started_value = task_params["grasp_not_started_value"]
        self.box_actor_index = task_params["box_actor_index"]

        # Contact duration will be converted to steps after physics manager is initialized
        self.contact_duration_threshold_steps = None

        # Asset and actor tracking
        self.box_asset = None
        self.box_actor_handles = []  # Store handles during creation
        self.box_actor_indices = None  # Will be set in set_tensor_references

        # Internal state for rewards/termination (NOT exposed to policy)
        self.box_states = None
        self.box_positions = None
        self.box_velocities = None
        self.initial_box_positions = None

    def finalize_setup(self):
        """
        Complete setup after physics manager is available.

        This is called after the physics manager has been created,
        allowing us to access control_dt.
        """
        # Convert contact duration from seconds to steps
        if self.parent_env is None:
            raise RuntimeError("parent_env is None - initialization failed")

        control_dt = self.parent_env.physics_manager.control_dt
        self.contact_duration_threshold_steps = int(
            self.contact_duration_threshold_seconds / control_dt
        )
        logger.info(
            f"Contact duration threshold: {self.contact_duration_threshold_seconds}s = "
            f"{self.contact_duration_threshold_steps} steps at {1/control_dt}Hz"
        )

        # Register grasp timing state with observation encoder (now available)
        if self.parent_env.observation_encoder is None:
            raise RuntimeError("observation_encoder is None - initialization failed")

        self.parent_env.observation_encoder.register_task_state(
            "grasp_start_steps", (self.num_envs,), dtype=torch.long
        )

    def load_task_assets(self):
        """Load box asset for the grasping task."""
        logger.info("Loading box asset...")

        # Create box asset
        asset_options = gymapi.AssetOptions()
        asset_options.density = (
            self.density_conversion_factor * self.box_mass / (self.box_size**3)
        )  # Compute density from mass
        asset_options.fix_base_link = False

        # Create box geometry
        self.box_asset = self.gym.create_box(
            self.sim, self.box_size, self.box_size, self.box_size, asset_options
        )

        # Initial positions for each environment
        self.initial_box_positions = torch.zeros((self.num_envs, 3), device=self.device)

        logger.info(f"Box asset loaded: {self.box_size}m cube, mass={self.box_mass}kg")

    def create_task_objects(self, gym, sim, env_ptr, env_id: int):
        """
        Create box actor for the environment.

        Args:
            gym: Gym instance
            sim: Simulation instance
            env_ptr: Environment pointer
            env_id: Environment index
        """
        # Create box actor
        box_pose = gymapi.Transform()

        # Randomize initial X,Y position
        x_offset = (torch.rand(1, device=self.device) * 2 - 1) * self.box_xy_range
        y_offset = (torch.rand(1, device=self.device) * 2 - 1) * self.box_xy_range

        box_pose.p = gymapi.Vec3(
            x_offset.item(), y_offset.item(), self.box_z  # Fixed Z on table surface
        )
        box_pose.r = gymapi.Quat(0, 0, 0, 1)

        # Store initial position
        self.initial_box_positions[env_id, 0] = x_offset.item()
        self.initial_box_positions[env_id, 1] = y_offset.item()
        self.initial_box_positions[env_id, 2] = self.box_z

        # Create actor
        box_actor = gym.create_actor(
            env_ptr,
            self.box_asset,
            box_pose,
            f"box_{env_id}",
            env_id,  # collision group
            0,  # collision filter - collides with everything
        )

        # Set box properties
        box_props = gym.get_actor_rigid_shape_properties(env_ptr, box_actor)
        for prop in box_props:
            prop.friction = self.box_friction
            prop.restitution = self.box_restitution
        gym.set_actor_rigid_shape_properties(env_ptr, box_actor, box_props)

        # Store actor handle (will be converted to index in set_tensor_references)
        self.box_actor_handles.append(box_actor)

    def set_tensor_references(self, root_state_tensor: torch.Tensor):
        """
        Set references to simulation tensors and convert actor indices.

        Args:
            root_state_tensor: Root state tensor for all actors
        """
        self.root_state_tensor = root_state_tensor

        # Convert box actor handles to global indices
        global_indices = []
        for env_id in range(self.num_envs):
            env_ptr = self.gym.get_env(self.sim, env_id)
            global_index = self.gym.get_actor_index(
                env_ptr, self.box_actor_handles[env_id], gymapi.DOMAIN_SIM
            )
            global_indices.append(global_index)

        self.box_actor_indices = torch.tensor(global_indices, device=self.device)

        # Extract box states
        # For GPU pipeline, we need to index differently
        # root_state_tensor shape: (num_envs, num_actors_per_env, 13)
        self.box_states = self.root_state_tensor[
            :, self.box_actor_index, :
        ]  # Shape: (num_envs, 13)
        self.box_positions = self.box_states[:, :3]
        self.box_velocities = self.box_states[:, 7:10]

        # Registration of task state will happen later in finalize_setup
        # when observation encoder is available

    def reset_task_state(self, env_ids: torch.Tensor):
        """
        Reset task-specific state for specified environments.

        Args:
            env_ids: Environment indices to reset
        """
        if len(env_ids) == 0:
            return

        # Skip if tensors not set up yet (called during initialization)
        if self.root_state_tensor is None:
            return
        if self.box_actor_indices is None:
            return

        # Reset box positions with randomization - vectorized
        num_resets = len(env_ids)

        # Generate random positions for all environments at once
        x_offsets = (
            torch.rand(num_resets, device=self.device) * 2 - 1
        ) * self.box_xy_range
        y_offsets = (
            torch.rand(num_resets, device=self.device) * 2 - 1
        ) * self.box_xy_range

        # Update initial positions
        self.initial_box_positions[env_ids, 0] = x_offsets
        self.initial_box_positions[env_ids, 1] = y_offsets
        self.initial_box_positions[env_ids, 2] = self.box_z

        # Set in root state tensor - vectorized
        self.root_state_tensor[env_ids, self.box_actor_index, 0] = x_offsets
        self.root_state_tensor[env_ids, self.box_actor_index, 1] = y_offsets
        self.root_state_tensor[env_ids, self.box_actor_index, 2] = self.box_z

        # Zero velocities
        self.root_state_tensor[env_ids, self.box_actor_index, 7:13] = 0

        # Clear grasp timing via observer state (only if already registered)
        if "grasp_start_steps" in self.parent_env.observation_encoder.task_states:
            grasp_start = self.parent_env.observation_encoder.get_task_state(
                "grasp_start_steps"
            )
            grasp_start[env_ids] = self.grasp_not_started_value

    def get_task_observations(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get task-specific observations.

        These observations are computed for reward calculation and analysis,
        but are NOT exposed to the policy (blind policy design).

        Args:
            obs_dict: Dictionary of current observations

        Returns:
            Dictionary of task observations (for rewards/analysis only)
        """
        task_obs = {}

        # Object pose (position + orientation)
        if self.box_positions is not None:
            task_obs["object_pos"] = self.box_positions
            task_obs["object_vel"] = self.box_velocities

        # Compute finger-to-object distances for reward calculation
        if "fingertip_poses_world" in obs_dict and self.box_positions is not None:
            fingertip_poses = obs_dict["fingertip_poses_world"]  # (num_envs, 35)
            fingertip_poses_reshaped = fingertip_poses.view(self.num_envs, 5, 7)
            fingertip_positions = fingertip_poses_reshaped[:, :, :3]  # (num_envs, 5, 3)

            # Compute distances from each fingertip to object center
            object_pos_expanded = self.box_positions.unsqueeze(1).expand(
                -1, 5, -1
            )  # (num_envs, 5, 3)
            finger_to_object_distances = torch.norm(
                fingertip_positions - object_pos_expanded, dim=2
            )  # (num_envs, 5)

            # Minimum distance from any fingertip to object
            min_finger_to_object_distance = torch.min(
                finger_to_object_distances, dim=1
            )[0]
            task_obs["finger_to_object_distances"] = finger_to_object_distances
            task_obs["min_finger_to_object_distance"] = min_finger_to_object_distance

        # Compute hand-to-object distance
        if "hand_pose" in obs_dict and self.box_positions is not None:
            hand_poses = obs_dict["hand_pose"]  # (num_envs, 7)
            hand_positions = hand_poses[:, :3]  # (num_envs, 3)
            hand_to_object_distance = torch.norm(
                hand_positions - self.box_positions, dim=1
            )  # (num_envs,)
            task_obs["hand_to_object_distance"] = hand_to_object_distance

        return task_obs

    def compute_task_reward_terms(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute task-specific reward components.

        Args:
            obs_dict: Dictionary of observations

        Returns:
            Dictionary of task-specific reward components
        """
        rewards = {}

        # Object height reward - encourage lifting
        if self.box_positions is not None:
            height_above_table = self.box_positions[:, 2] - self.box_z
            height_reward = torch.clamp(
                height_above_table / (self.height_threshold - self.box_z), 0, 1
            )
            rewards["object_height"] = height_reward * self.object_height_weight

        # Grasp approach reward - encourage any contact
        if "contact_binary" in obs_dict:
            any_contact = obs_dict["contact_binary"].any(dim=1).float()
            rewards["grasp_approach"] = any_contact * self.grasp_approach_weight

        # Finger-to-object distance reward - encourage getting fingers close to object
        if "min_finger_to_object_distance" in obs_dict:
            # Exponential reward: max reward when distance is 0, decays as distance increases
            min_distance = obs_dict["min_finger_to_object_distance"]
            finger_distance_reward = torch.exp(
                -2.0 * min_distance
            )  # Decays quickly with distance
            rewards["finger_to_object"] = (
                finger_distance_reward * self.finger_to_object_weight
            )

        # Hand-to-object distance reward - encourage getting hand close to object
        if "hand_to_object_distance" in obs_dict:
            # Similar exponential reward for hand approach
            hand_distance = obs_dict["hand_to_object_distance"]
            hand_distance_reward = torch.exp(
                -1.0 * hand_distance
            )  # Slower decay than finger reward
            rewards["hand_to_object"] = (
                hand_distance_reward * self.hand_to_object_weight
            )

        return rewards

    def check_task_success_criteria(self) -> Dict[str, torch.Tensor]:
        """
        Check task-specific success criteria.

        Success requires:
        1. Object height > 20cm
        2. At least 2 fingers in contact for at least 2 seconds

        Returns:
            Dictionary with "grasp_lift_success" boolean tensor
        """
        # Check height criteria
        height_success = self.box_positions[:, 2] > self.height_threshold

        # Check contact duration criteria
        # Access contact duration directly from observation encoder
        if self.contact_duration_threshold_steps is None:
            raise RuntimeError(
                "contact_duration_threshold_steps not set - finalize_setup not called"
            )

        # Get contact durations in steps
        contact_durations = self.parent_env.observation_encoder.contact_duration_steps

        # Count fingers with sufficient contact
        sufficient_contact = (
            contact_durations >= self.contact_duration_threshold_steps
        ).sum(dim=1) >= self.min_fingers_for_grasp

        # Both criteria must be met
        success = height_success & sufficient_contact

        return {"grasp_lift_success": success}

    def check_task_failure_criteria(self) -> Dict[str, torch.Tensor]:
        """
        Check task-specific failure criteria.

        No additional failure criteria beyond base (ground collisions).
        Episode timeout is handled by base environment.

        Returns:
            Empty dictionary
        """
        return {}

    def compute_task_rewards(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute task rewards using the reward calculator.

        Args:
            obs_dict: Dictionary of observations

        Returns:
            Tuple of (reward tensor, reward terms dictionary)
        """
        # Check parent environment access
        if self.parent_env is None:
            raise RuntimeError("parent_env is None - initialization failed")

        # Compute common reward terms
        common_rewards = self.parent_env.reward_calculator.compute_common_reward_terms(
            obs_dict, self.parent_env
        )

        # Compute task-specific rewards
        task_rewards = self.compute_task_reward_terms(obs_dict)

        # Compute total reward
        (
            total_rewards,
            reward_components,
        ) = self.parent_env.reward_calculator.compute_total_reward(
            common_rewards=common_rewards,
            task_rewards=task_rewards,
        )

        return total_rewards, reward_components

    def check_task_reset(self) -> torch.Tensor:
        """
        Check if task-specific reset conditions are met.

        Reset if box falls off table or gets too far from origin.

        Returns:
            Boolean tensor indicating which environments should reset
        """
        # Reset if box falls below table
        box_fell = self.box_positions[:, 2] < 0.0

        # Reset if box gets too far from origin
        box_distance = torch.norm(self.box_positions[:, :2], dim=1)
        box_too_far = box_distance > self.max_box_distance

        return box_fell | box_too_far
