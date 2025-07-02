"""
Box grasping task for DexHand.

This module implements a blind grasping task where the policy learns to grasp
a 5cm box using only tactile feedback (binary contacts and duration).
"""

from typing import Dict, Optional, Tuple
import math

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

        # Task configuration parameters (not weights)

        # Task-specific parameters from config
        task_params = cfg["env"]["task_params"]
        self.height_threshold = task_params["success_height_threshold"]
        self.contact_duration_threshold_seconds = task_params[
            "contact_duration_threshold"
        ]
        self.min_fingers_for_grasp = task_params["min_fingers_for_grasp"]
        self.max_box_distance = task_params["max_box_distance"]

        # Contact duration will be converted to steps after physics manager is initialized
        self.contact_duration_threshold_steps = None

        # Asset and actor tracking
        self.box_asset = None
        self.box_actor_handles = []  # Store handles during creation
        self.box_actor_indices = None  # Will be set in set_tensor_references
        self.box_local_actor_index = None  # Local actor index within each environment
        self.box_local_rigid_body_index = (
            None  # Local rigid body index for contact detection
        )

        # Internal state for rewards/termination (NOT exposed to policy)
        self.box_states = None
        self.box_positions = None
        self.box_velocities = None
        self.initial_box_positions = None

    def initialize_task_states(self):
        """
        Initialize task states that need to be registered with observation encoder.

        This is called early in initialization, before observation encoder setup,
        to ensure states are available when computing observation dimensions.
        """
        if self.parent_env is None:
            raise RuntimeError("parent_env is None - initialization failed")

        # Register success tracking states
        self.parent_env.observation_encoder.register_task_state(
            "success_duration_steps", (self.num_envs,), dtype=torch.long
        )
        self.parent_env.observation_encoder.register_task_state(
            "success_conditions_met", (self.num_envs,), dtype=torch.bool
        )

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

        # Task states are now registered in initialize_task_states()

    def load_task_assets(self):
        """Load box asset for the grasping task."""
        logger.info("Loading box asset...")

        # Create box asset
        asset_options = gymapi.AssetOptions()
        # Compute density directly from mass and volume
        box_volume = self.box_size**3  # Volume of cube
        asset_options.density = self.box_mass / box_volume
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

        # Get the local actor index within each environment
        # Query the local index of the box actor in the first environment
        env0_ptr = self.gym.get_env(self.sim, 0)
        num_actors_env0 = self.gym.get_actor_count(env0_ptr)

        # Find which local index corresponds to our box
        for local_idx in range(num_actors_env0):
            actor_handle = self.gym.get_actor_handle(env0_ptr, local_idx)
            if actor_handle == self.box_actor_handles[0]:
                self.box_local_actor_index = local_idx
                break

        if self.box_local_actor_index is None:
            raise RuntimeError("Failed to find box actor's local index")

        logger.info(f"Box actor local index: {self.box_local_actor_index}")

        # Get box rigid body index for contact detection
        # The box actor has only one rigid body (the box itself)
        box_handle = self.box_actor_handles[0]
        self.box_local_rigid_body_index = self.gym.find_actor_rigid_body_index(
            env0_ptr, box_handle, "box", gymapi.DOMAIN_SIM
        )

        # Convert from global to local index
        # Get the first rigid body global index in env0
        env0_first_body_global_idx = self.gym.get_actor_rigid_body_index(
            env0_ptr, self.gym.get_actor_handle(env0_ptr, 0), 0, gymapi.DOMAIN_SIM
        )
        self.box_local_rigid_body_index -= env0_first_body_global_idx

        logger.info(f"Box rigid body local index: {self.box_local_rigid_body_index}")

        # Extract box states
        # For GPU pipeline, we need to index differently
        # root_state_tensor shape: (num_envs, num_actors_per_env, 13)
        self.box_states = self.root_state_tensor[
            :, self.box_local_actor_index, :
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
        self.root_state_tensor[env_ids, self.box_local_actor_index, 0] = x_offsets
        self.root_state_tensor[env_ids, self.box_local_actor_index, 1] = y_offsets
        self.root_state_tensor[env_ids, self.box_local_actor_index, 2] = self.box_z

        # Zero velocities
        self.root_state_tensor[env_ids, self.box_local_actor_index, 7:13] = 0

        # Reset success tracking states
        success_duration_steps = self.parent_env.observation_encoder.get_task_state(
            "success_duration_steps"
        )
        success_conditions_met = self.parent_env.observation_encoder.get_task_state(
            "success_conditions_met"
        )
        success_duration_steps[env_ids] = 0
        success_conditions_met[env_ids] = False

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

        # Compute fingerpad pairwise distances (for policy observation)
        if "fingerpad_poses_world" in obs_dict:
            fingerpad_poses = obs_dict["fingerpad_poses_world"]  # (num_envs, 35)
            fingerpad_poses_reshaped = fingerpad_poses.view(self.num_envs, 5, 7)
            fingerpad_positions = fingerpad_poses_reshaped[:, :, :3]  # (num_envs, 5, 3)

            # Compute pairwise distances between all fingerpads
            # This gives us 5×4/2 = 10 unique distances
            fingerpad_distances = self._compute_fingerpad_pairwise_distances(
                fingerpad_positions
            )
            task_obs["fingerpad_distances"] = fingerpad_distances  # (num_envs, 10)

        # Compute success duration (privileged information for grasp duration)
        # Get registered task states - fail fast if not registered
        success_duration_steps = self.parent_env.observation_encoder.get_task_state(
            "success_duration_steps"
        )
        success_conditions_met = self.parent_env.observation_encoder.get_task_state(
            "success_conditions_met"
        )

        # Check if we have the necessary information to compute success conditions
        if (
            self.parent_env.physics_manager is not None
            and self.parent_env.physics_manager.control_dt is not None
            and self.box_local_rigid_body_index is not None
        ):
            # Detect finger-box contact using our heuristic method
            finger_box_contact = self._detect_finger_box_contacts(obs_dict)
            num_fingers_on_box = finger_box_contact.sum(dim=1)

            # Check height condition
            height_above_threshold = self.box_positions[:, 2] > self.height_threshold

            # Success conditions: height > threshold AND at least min_fingers on box
            current_success_conditions = height_above_threshold & (
                num_fingers_on_box >= self.min_fingers_for_grasp
            )

            # Detect when success conditions start being met
            success_started = current_success_conditions & ~success_conditions_met

            # Update duration counter
            success_duration_steps[:] = torch.where(
                success_started,
                torch.ones_like(success_duration_steps),
                torch.where(
                    current_success_conditions,
                    success_duration_steps + 1,
                    torch.zeros_like(success_duration_steps),
                ),
            )

            # Update state
            success_conditions_met[:] = current_success_conditions

            # Convert to seconds for observation (this is the "grasp duration")
            control_dt = self.parent_env.physics_manager.control_dt
            grasp_duration_seconds = success_duration_steps.float() * control_dt
            task_obs["grasp_duration"] = grasp_duration_seconds.unsqueeze(1)
        else:
            # During initialization, return zeros with correct shape
            task_obs["grasp_duration"] = torch.zeros(
                (self.num_envs, 1), device=self.device
            )

        return task_obs

    def _compute_fingerpad_pairwise_distances(self, fingerpad_positions):
        """
        Compute pairwise distances between all fingerpads.

        Args:
            fingerpad_positions: Tensor of shape (num_envs, 5, 3)

        Returns:
            Tensor of shape (num_envs, 10) with pairwise distances
        """
        # Indices for upper triangular (excluding diagonal)
        # This gives us: (0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
        indices = torch.triu_indices(5, 5, offset=1, device=self.device)

        # Extract positions for each pair
        pos1 = fingerpad_positions[:, indices[0]]  # (num_envs, 10, 3)
        pos2 = fingerpad_positions[:, indices[1]]  # (num_envs, 10, 3)

        # Compute distances
        distances = torch.norm(pos1 - pos2, dim=2)  # (num_envs, 10)

        return distances

    def _detect_finger_box_contacts(self, obs_dict: Dict[str, torch.Tensor]):
        """
        Detect which fingers are in contact with the box using heuristic criteria.

        This uses a combination of three conditions to infer finger-box contact:
        1. Finger contact: The finger must be experiencing contact forces (from any source)
        2. Box contact: The box must be experiencing contact forces (from any source)
        3. Proximity: The fingerpad must be within sqrt(3) * box_size/2 of the box center

        This heuristic approach avoids the need to iterate through contact pairs while
        providing reasonable accuracy. The proximity threshold ensures we only consider
        contacts that could plausibly be finger-box interactions (the threshold is the
        diagonal distance from box center to corner, ensuring coverage of the entire box).

        Returns:
            torch.Tensor: Boolean tensor of shape (num_envs, num_fingers) indicating
                         which fingers are likely in contact with the box
        """
        # Get contact forces for box from the full contact forces tensor
        contact_forces_all = self.parent_env.tensor_manager.contact_forces_all
        box_forces = contact_forces_all[:, self.box_local_rigid_body_index, :]
        box_force_magnitude = torch.norm(box_forces, dim=1)

        # Box has contact if force magnitude exceeds threshold
        contact_threshold = self.parent_env.cfg["env"].get(
            "contactBinaryThreshold", 1.0
        )
        box_has_contact = box_force_magnitude > contact_threshold

        # Get fingerpad positions from rigid body states
        rigid_body_states = self.parent_env.tensor_manager.rigid_body_states
        fingerpad_indices = self.parent_env.hand_initializer.fingerpad_local_indices
        fingerpad_positions = rigid_body_states[
            :, fingerpad_indices, :3
        ]  # (num_envs, 5, 3)

        # Get box positions (already extracted)
        box_positions_expanded = self.box_positions.unsqueeze(1)  # (num_envs, 1, 3)

        # Compute distances from each fingerpad to box center
        distances = torch.norm(
            fingerpad_positions - box_positions_expanded, dim=2
        )  # (num_envs, 5)

        # Proximity check: distance < sqrt(3) * box_size/2
        # This is the diagonal distance from center to corner of the box
        proximity_threshold = math.sqrt(3) * self.box_size / 2
        fingerpad_near_box = distances < proximity_threshold

        # Get finger contact from observation (already computed by observation encoder)
        # Convert from float to bool (contact_binary is stored as float for compatibility)
        finger_has_contact = obs_dict["contact_binary"].bool()

        # Combine all conditions: finger contact AND box contact AND proximity
        finger_box_contact = (
            finger_has_contact
            & box_has_contact.unsqueeze(1)  # Finger sensing contact
            & fingerpad_near_box  # Box experiencing force  # Fingerpad close to box
        )

        return finger_box_contact

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
            rewards["object_height"] = height_reward

        # Grasp approach reward - encourage finger-box contact specifically
        # Use our heuristic detection to identify true finger-box contact
        if self.box_local_rigid_body_index is not None:
            finger_box_contact = self._detect_finger_box_contacts(obs_dict)
            any_box_contact = finger_box_contact.any(dim=1).float()
            rewards["grasp_approach"] = any_box_contact
        else:
            # During initialization, no reward
            rewards["grasp_approach"] = torch.zeros(self.num_envs, device=self.device)

        # Finger-to-object distance reward - encourage getting fingers close to object
        # Exponential reward: max reward when distance is 0, decays as distance increases
        # Tuned for: ~0.8 at 3cm distance, ~0.1 at 50cm distance
        min_distance = obs_dict["min_finger_to_object_distance"]
        finger_distance_reward = torch.exp(
            -5.0 * min_distance
        )  # Quick decay to focus on close-range approach
        rewards["finger_to_object"] = finger_distance_reward

        # Hand-to-object distance reward - encourage getting hand close to object
        # Similar exponential reward for hand approach
        hand_distance = obs_dict["hand_to_object_distance"]
        hand_distance_reward = torch.exp(
            -1.0 * hand_distance
        )  # Slower decay than finger reward
        rewards["hand_to_object"] = hand_distance_reward

        # Grasp duration reward - reward for maintaining grasp
        # Grasp duration is computed in observation encoder as privileged info
        if "grasp_duration" in obs_dict:
            grasp_duration = obs_dict["grasp_duration"].squeeze(-1)  # In seconds
            # Normalize to [0, 1] based on contact duration threshold
            grasp_duration_reward = torch.clamp(
                grasp_duration / self.contact_duration_threshold_seconds, 0, 1
            )
            rewards["grasp_duration"] = grasp_duration_reward

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
        # Check success duration criteria
        # We track success duration based on height AND finger-box contact
        if self.contact_duration_threshold_steps is None:
            raise RuntimeError(
                "contact_duration_threshold_steps not set - finalize_setup not called"
            )

        # Get success duration that we've been tracking
        success_duration_steps = self.parent_env.observation_encoder.get_task_state(
            "success_duration_steps"
        )

        # Success requires maintained conditions for threshold duration
        success = success_duration_steps >= self.contact_duration_threshold_steps

        return {"grasp_lift_success": success}

    def check_task_failure_criteria(self) -> Dict[str, torch.Tensor]:
        """
        Check task-specific failure criteria.

        Failures:
        1. Box fell below table
        2. Hand too far from box (> max_box_distance)

        Returns:
            Dictionary with failure criteria
        """
        failures = {}

        # Box fell below table
        failures["box_fell"] = self.box_positions[:, 2] < 0.0

        # Hand too far from box (use configured threshold)
        if "hand_to_object_distance" in self.parent_env.obs_dict:
            hand_to_box_distance = self.parent_env.obs_dict["hand_to_object_distance"]
            failures["box_too_far"] = hand_to_box_distance > self.max_box_distance

        return failures

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

        Task-specific resets are now handled through failure criteria.
        This method returns all False to let failure criteria handle resets.

        Returns:
            Boolean tensor of all False
        """
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
