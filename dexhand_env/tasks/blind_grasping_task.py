"""
Blind grasping task for DexHand.

This module implements a blind grasping task where the policy learns to grasp
a 5cm box using only tactile feedback (binary contacts and duration).
"""

from typing import Dict, Optional
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
from dexhand_env.constants import FINGER_JOINT_NAMES


class BlindGraspingTask(DexTask):
    """
    Blind grasping task implementing curriculum learning through structured stages.

    ## Design Philosophy: Blind Tactile-Only Grasping

    This task embodies a key robotics research principle: robust manipulation should not
    depend on perfect visual information. Real-world grasping often occurs with occluded
    views, poor lighting, or when manipulating objects behind obstacles. By forcing the
    policy to learn using only tactile feedback (contact binary/duration), we encourage
    the development of generalizable grasping strategies that rely on fundamental physical
    principles rather than visual pattern matching.

    The "blind" constraint serves two research purposes:
    1. **Generalization**: Policies trained without visual dependencies transfer better
       to novel objects and environments
    2. **Robustness**: Tactile-based grasping is inherently more robust to environmental
       variations (lighting, visual noise, partial occlusion)

    ## Design Philosophy: Three-Stage Curriculum Learning

    The finite state machine architecture addresses a fundamental challenge in
    manipulation learning: the enormous action space and sparse reward problem.
    Without structure, policies struggle to discover the sequential nature of
    successful grasping (approach → contact → lift).

    **Why Three Stages?**
    - **Stage 1 (Pre-grasp)**: Provides dense rewards for hand positioning before contact,
      solving the exploration problem of "where to place the hand"
    - **Stage 2 (Grasp)**: Focuses reward on contact establishment and grasp formation,
      teaching the policy to recognize and achieve stable grasps
    - **Stage 3 (Lift)**: Rewards maintenance of established grasps during manipulation,
      encouraging policies to preserve rather than recklessly modify successful grasps

    This staged approach implements curriculum learning: each stage builds necessary
    skills for the next, dramatically reducing training time compared to end-to-end
    learning with only terminal rewards.

    ## Design Philosophy: Policy-Observable vs Privileged Information Split

    The observation design reflects a core tension in RL: training signal quality vs.
    deployment robustness. We resolve this through careful information segregation:

    **Policy Observations (Deployment Reality):**
    Only information realistically available to deployed robots: proprioception,
    tactile feedback, and hand state. No object-specific information that would
    require perfect perception systems.

    **Privileged Information (Training Signal Only):**
    Ground-truth object state used exclusively for reward computation and failure
    detection. This enables precise training signals without creating deployment
    dependencies on perfect perception.

    This design ensures policies learn from high-quality reward signals during training
    but remain deployable with realistic sensor suites.

    ## Design Philosophy: Fail-Fast Stage Evaluation

    Each stage has explicit quality criteria and failure modes because manipulation
    research requires precise performance attribution. Rather than hoping policies
    will eventually discover good strategies, we explicitly evaluate and terminate
    episodes that demonstrate poor technique.

    **Why Explicit Failure Detection?**
    - **Training Efficiency**: Avoids wasting training time on episodes that started poorly
    - **Analysis Capability**: Enables researchers to identify specific failure modes
    - **Reward Signal Clarity**: Provides clear feedback about what constitutes progress

    Stage-specific failures (pregrasp_failed, contact_failed, grasp_lost) allow detailed
    analysis of which aspects of the manipulation pipeline need improvement.

    ## Design Philosophy: Exponential Reward Shaping

    Reward functions use exponential decay (e^(-k*distance)) rather than linear rewards
    because exponential functions provide:
    - **Smooth gradients**: Essential for gradient-based policy optimization
    - **Distance-appropriate scaling**: Large penalties for gross failures, fine-tuned
      rewards near success conditions
    - **Bounded values**: Prevents reward components from overwhelming each other

    The decay constants are tuned to provide meaningful gradients across the expected
    range of behaviors while maintaining numerical stability.

    All specific parameters, thresholds, and weights are configurable in BoxGrasping.yaml
    to support research experimentation without code modification.
    """

    def __init__(self, sim, gym, device, num_envs, full_config):
        """
        Initialize the blind grasping task.

        Args:
            sim: Simulation instance
            gym: Gym instance
            device: PyTorch device
            num_envs: Number of environments
            full_config: Complete configuration dictionary
        """
        self.sim = sim
        self.gym = gym
        self.device = device
        self.num_envs = num_envs
        self.cfg = full_config

        # Reference to parent environment (set by DexHandBase)
        self.parent_env = None

        # Clean config section access
        env_config = full_config["env"]
        task_config = full_config["task"]

        # Box configuration
        self.box_size = env_config["box"]["size"]
        self.box_mass = env_config["box"]["mass"]
        self.box_friction = env_config["box"]["friction"]
        self.box_restitution = env_config["box"]["restitution"]
        self.box_xy_range = env_config["box"]["initial_position"]["xy_range"]
        self.box_z = env_config["box"]["initial_position"]["z"]

        # Task configuration parameters (not weights)
        self.height_threshold = task_config["success_height_threshold"]
        self.contact_duration_threshold_seconds = task_config[
            "contact_duration_threshold"
        ]
        self.min_fingers_for_grasp = task_config["min_fingers_for_grasp"]
        self.max_box_distance = task_config["max_box_distance"]

        # Stage timing parameters
        self.stage1_duration = task_config["stage1_duration"]
        self.stage2_duration = task_config["stage2_duration"]

        # Hand randomization parameters
        self.hand_translation_range = task_config["hand_translation_range"]
        self.hand_rotation_range = task_config["hand_rotation_range"]

        # Finger randomization parameters
        finger_rand = task_config["finger_randomization"]
        self.thumb_rotation_range = finger_rand["thumb_rotation_range"]
        self.other_finger_range = finger_rand["other_finger_range"]

        # Stage evaluation parameters
        stage_eval = task_config["stage_evaluation"]
        self.stage2_contact_success_threshold = stage_eval[
            "stage2_contact_success_threshold"
        ]

        # Reward calculation parameters
        reward_calc = task_config["reward_calculation"]
        self.height_alignment_decay = reward_calc["height_alignment_decay"]
        self.centroid_positioning_decay = reward_calc["centroid_positioning_decay"]
        self.object_stability_decay = reward_calc["object_stability_decay"]
        self.first_three_height_consistency_decay = reward_calc[
            "first_three_height_consistency_decay"
        ]
        self.fingerpad_proximity_decay = reward_calc["fingerpad_proximity_decay"]
        self.base_stability_decay = reward_calc["base_stability_decay"]

        # Load penetration prevention parameters
        penetration_cfg = task_config["penetrationPrevention"]
        self.geometric_penetration_factor = penetration_cfg[
            "geometricPenetrationFactor"
        ]
        self.proximity_min_distance_factor = penetration_cfg[
            "proximityMinDistanceFactor"
        ]
        self.penetration_depth_scale = penetration_cfg["penetrationDepthScale"]

        # Quality evaluation thresholds
        quality_thresh = task_config["quality_thresholds"]
        self.height_tolerance = quality_thresh["height_tolerance"]
        self.centroid_tolerance = quality_thresh["centroid_tolerance"]
        self.position_drift_tolerance = quality_thresh["position_drift_tolerance"]
        self.velocity_tolerance = quality_thresh["velocity_tolerance"]

        # Visual settings
        visual = task_config["visualization"]
        self.box_color = visual["box_color"]

        # Contact duration will be converted to steps after physics manager is initialized
        self.contact_duration_threshold_steps = None

        # Preallocate failure tensors for efficiency (avoid allocation on every call)
        self._failure_tensors = None

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
        self.register_task_state(
            "success_duration_steps", (self.num_envs,), dtype=torch.long
        )
        self.register_task_state(
            "success_conditions_met", (self.num_envs,), dtype=torch.bool
        )

        # Register stage tracking states
        self.register_task_state("current_stage", (self.num_envs,), dtype=torch.long)
        self.register_task_state("time_in_stage", (self.num_envs,), dtype=torch.float32)
        self.register_task_state(
            "stage_contact_duration", (self.num_envs,), dtype=torch.float32
        )

        # Register transition tracking states for exact-step detection
        self.register_task_state(
            "just_transitioned_to_stage2", (self.num_envs,), dtype=torch.bool
        )
        self.register_task_state(
            "just_transitioned_to_stage3", (self.num_envs,), dtype=torch.bool
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

        # CRITICAL: Initialize all task states with proper values
        # This ensures current_stage starts at 1, time_in_stage at 0, etc.
        # Without this, task states remain at default values (0) and stage transitions never work
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self.reset_task_state(all_env_ids)
        logger.info(
            f"Contact duration threshold: {self.contact_duration_threshold_seconds}s = "
            f"{self.contact_duration_threshold_steps} steps at {1/control_dt}Hz"
        )

        # Initialize preallocated failure tensors for efficiency
        self._failure_tensors = {
            "stage1_pregrasp_failed": torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            ),
            "stage2_contact_failed": torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            ),
            "stage3_grasp_lost": torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            ),
        }

        # Task states are now registered in initialize_task_states()

    def load_task_assets(self):
        """Load box asset for the blind grasping task."""
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
        # Create box actor with default pose (randomization handled in reset_task_state)
        box_pose = gymapi.Transform()
        box_pose.p = gymapi.Vec3(
            0.0, 0.0, self.box_z
        )  # Center position, fixed Z on table
        box_pose.r = gymapi.Quat(0, 0, 0, 1)  # Default orientation

        # Store default initial position (will be updated during first reset)
        self.initial_box_positions[env_id, 0] = 0.0
        self.initial_box_positions[env_id, 1] = 0.0
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

        # Set box color from configuration
        gym.set_rigid_body_color(
            env_ptr, box_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*self.box_color)
        )

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

        # Fail fast if required dependencies are missing
        if self.root_state_tensor is None:
            raise RuntimeError("root_state_tensor is None - initialization failed")
        if self.box_actor_indices is None:
            raise RuntimeError("box_actor_indices is None - initialization failed")

        # Reset box positions with randomization - vectorized
        num_resets = len(env_ids)

        # Generate random positions for all environments at once
        x_offsets = (
            torch.rand(num_resets, device=self.device) * 2 - 1
        ) * self.box_xy_range
        y_offsets = (
            torch.rand(num_resets, device=self.device) * 2 - 1
        ) * self.box_xy_range

        # Generate random Z-axis rotations (around vertical axis)
        rotation_angles = (
            torch.rand(num_resets, device=self.device) * 2 - 1
        ) * math.pi  # ±180°

        # Convert to quaternions: quat = [x, y, z, w] = [0, 0, sin(θ/2), cos(θ/2)]
        quat_z = torch.sin(rotation_angles / 2)
        quat_w = torch.cos(rotation_angles / 2)

        # Update initial positions
        self.initial_box_positions[env_ids, 0] = x_offsets
        self.initial_box_positions[env_ids, 1] = y_offsets
        self.initial_box_positions[env_ids, 2] = self.box_z

        # Set position and rotation in root state tensor - vectorized
        self.root_state_tensor[env_ids, self.box_local_actor_index, 0] = x_offsets
        self.root_state_tensor[env_ids, self.box_local_actor_index, 1] = y_offsets
        self.root_state_tensor[env_ids, self.box_local_actor_index, 2] = self.box_z

        # Set rotation quaternion [x, y, z, w]
        self.root_state_tensor[env_ids, self.box_local_actor_index, 3] = 0  # quat_x
        self.root_state_tensor[env_ids, self.box_local_actor_index, 4] = 0  # quat_y
        self.root_state_tensor[
            env_ids, self.box_local_actor_index, 5
        ] = quat_z  # quat_z
        self.root_state_tensor[
            env_ids, self.box_local_actor_index, 6
        ] = quat_w  # quat_w

        # Zero velocities (both linear and angular)
        self.root_state_tensor[env_ids, self.box_local_actor_index, 7:13] = 0

        # Reset success tracking states
        success_duration_steps = self.task_states["success_duration_steps"]
        success_conditions_met = self.task_states["success_conditions_met"]
        success_duration_steps[env_ids] = 0
        success_conditions_met[env_ids] = False

        # Reset stage tracking states
        current_stage = self.task_states["current_stage"]
        time_in_stage = self.task_states["time_in_stage"]
        stage_contact_duration = self.task_states["stage_contact_duration"]
        just_transitioned_to_stage2 = self.task_states["just_transitioned_to_stage2"]
        just_transitioned_to_stage3 = self.task_states["just_transitioned_to_stage3"]

        current_stage[env_ids] = 1  # Start in Stage 1
        time_in_stage[env_ids] = 0.0
        stage_contact_duration[env_ids] = 0.0
        just_transitioned_to_stage2[env_ids] = False
        just_transitioned_to_stage3[env_ids] = False

        # Hand DOF randomization for diversity in training

        # Access DOF state tensor from parent environment
        dof_state = self.parent_env.dof_state
        num_resets = len(env_ids)

        # Translation randomization for ARTx, ARTy, ARTz (DOFs 0,1,2)
        translation_noise = (
            torch.rand(num_resets, 3, device=self.device) * 2 - 1
        ) * self.hand_translation_range
        dof_state[env_ids, 0:3, 0] = translation_noise

        # Rotation randomization for ARRx, ARRy, ARRz (DOFs 3,4,5)
        rotation_noise = (
            torch.rand(num_resets, 3, device=self.device) * 2 - 1
        ) * self.hand_rotation_range
        dof_state[env_ids, 3:6, 0] = rotation_noise

        # Vectorized finger DOF randomization
        finger_dof_count = len(FINGER_JOINT_NAMES)  # 20 finger DOFs
        finger_random_values = torch.rand(
            num_resets, finger_dof_count, device=self.device
        )

        # Create range mask: thumb rotation gets special range, others get standard range
        finger_ranges = torch.full(
            (finger_dof_count,), self.other_finger_range, device=self.device
        )
        finger_ranges[
            0
        ] = self.thumb_rotation_range  # r_f_joint1_1 is first in FINGER_JOINT_NAMES

        # Apply ranges and set all finger DOFs vectorized
        finger_random_values *= finger_ranges.unsqueeze(
            0
        )  # Broadcast ranges across environments
        dof_state[env_ids, 6 : 6 + finger_dof_count, 0] = finger_random_values

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

        # Object pose (position + orientation) - fail fast if not initialized
        if self.box_positions is None:
            raise RuntimeError("box_positions is None - initialization failed")
        task_obs["object_pos"] = self.box_positions
        task_obs["object_vel"] = self.box_velocities

        # Extract common fingerpad positions once
        fingerpad_positions = self._extract_fingerpad_positions(obs_dict)

        # Compute spatial relationships
        task_obs.update(
            self._compute_spatial_relationships(fingerpad_positions, obs_dict)
        )

        # Compute fingerpad geometry
        task_obs.update(self._compute_fingerpad_geometry(fingerpad_positions))

        # Compute contact state for reuse throughout the system
        (
            finger_box_contact,
            thumb_contact,
            other_fingers_contact,
            grasp_state,
        ) = self._compute_finger_contact_state(obs_dict)
        task_obs["thumb_contact"] = thumb_contact.unsqueeze(1).float()
        task_obs["other_fingers_contact"] = other_fingers_contact.unsqueeze(1).float()
        task_obs["grasp_state"] = grasp_state.unsqueeze(1).float()

        # Compute success duration (privileged information for grasp duration)
        # Get registered task states - fail fast if not registered
        success_duration_steps = self.task_states["success_duration_steps"]
        success_conditions_met = self.task_states["success_conditions_met"]

        # Check if we have the necessary information to compute success conditions
        # Some dependencies may be None during initialization - handle gracefully
        if self.parent_env.physics_manager is None:
            raise RuntimeError("physics_manager is None - initialization failed")
        if self.box_local_rigid_body_index is None:
            raise RuntimeError(
                "box_local_rigid_body_index is None - initialization failed"
            )

        # During initialization, control_dt may not be set yet
        if self.parent_env.physics_manager.control_dt is None:
            # During initialization, provide default grasp duration
            task_obs["grasp_duration"] = torch.zeros(
                (self.num_envs, 1), device=self.device
            )
            # During initialization, provide default stage observations
            task_obs["current_stage"] = torch.ones(
                (self.num_envs, 1), device=self.device, dtype=torch.float32
            )  # Start at stage 1
            task_obs["time_in_stage"] = torch.zeros(
                (self.num_envs, 1), device=self.device
            )  # No time elapsed yet
            task_obs["stage_progress"] = torch.zeros(
                (self.num_envs, 1), device=self.device
            )  # No progress yet
        else:
            # Normal runtime computation of grasp duration
            # Detect finger-box contact using our heuristic method
            finger_box_contact = self._detect_finger_box_contacts(obs_dict)
            num_fingers_on_box = finger_box_contact.sum(dim=1)

            # Check height condition
            height_above_threshold = self.box_positions[:, 2] > self.height_threshold

            # Success conditions: height > threshold AND at least min_fingers on box
            current_success_conditions = height_above_threshold & (
                num_fingers_on_box >= self.min_fingers_for_grasp
            )

            # Simple counter logic: increment when condition met, reset when not met
            success_duration_steps[current_success_conditions] += 1
            success_duration_steps[~current_success_conditions] = 0

            # Update state
            success_conditions_met[:] = current_success_conditions

            # Convert to seconds for observation (this is the "grasp duration")
            control_dt = self.parent_env.physics_manager.control_dt
            grasp_duration_seconds = success_duration_steps.float() * control_dt
            task_obs["grasp_duration"] = grasp_duration_seconds.unsqueeze(1)

            # Stage management logic
            self._update_stage_state(obs_dict, task_obs, control_dt)

        return task_obs

    def _extract_fingerpad_positions(self, obs_dict):
        """
        Extract and reshape fingerpad positions from observations.

        Args:
            obs_dict: Dictionary of observations

        Returns:
            Tensor of shape (num_envs, 5, 3) with fingerpad positions
        """
        fingerpad_poses = obs_dict["fingerpad_poses_world"]  # (num_envs, 35)
        return fingerpad_poses.view(self.num_envs, 5, 7)[:, :, :3]  # (num_envs, 5, 3)

    def _compute_finger_contact_state(self, obs_dict):
        """
        Single source of truth for finger contact computations.

        Args:
            obs_dict: Dictionary of observations

        Returns:
            Tuple of (finger_box_contact, thumb_contact, other_fingers_contact, grasp_state)
        """
        finger_box_contact = self._detect_finger_box_contacts(obs_dict)
        thumb_contact = finger_box_contact[:, 0]
        other_fingers_contact = finger_box_contact[:, 1:].any(dim=1)
        grasp_state = thumb_contact & other_fingers_contact
        return finger_box_contact, thumb_contact, other_fingers_contact, grasp_state

    def _compute_spatial_relationships(self, fingerpad_positions, obs_dict):
        """
        Compute spatial relationships between fingers, hand, and object.

        Args:
            fingerpad_positions: Tensor of shape (num_envs, 5, 3)
            obs_dict: Dictionary of observations

        Returns:
            Dictionary of spatial relationship observations
        """
        spatial_obs = {}

        # Finger-to-object distances
        object_pos_expanded = self.box_positions.unsqueeze(1).expand(-1, 5, -1)
        finger_to_object_distances = torch.norm(
            fingerpad_positions - object_pos_expanded, dim=2
        )
        spatial_obs["finger_to_object_distances"] = finger_to_object_distances
        spatial_obs["avg_finger_to_object_distance"] = torch.mean(
            finger_to_object_distances, dim=1
        )

        # Finger-to-object height differences
        fingerpad_heights = fingerpad_positions[:, :, 2]
        object_height = self.box_positions[:, 2]
        finger_to_object_height_diff = torch.abs(
            fingerpad_heights - object_height.unsqueeze(1)
        )
        spatial_obs["finger_to_object_height_diff"] = finger_to_object_height_diff
        spatial_obs["avg_finger_to_object_height_diff"] = torch.mean(
            finger_to_object_height_diff, dim=1
        )

        # Hand-to-object distance
        hand_positions = obs_dict["hand_pose"][:, :3]
        spatial_obs["hand_to_object_distance"] = torch.norm(
            hand_positions - self.box_positions, dim=1
        )

        return spatial_obs

    def _compute_fingerpad_geometry(self, fingerpad_positions):
        """
        Compute geometric relationships between fingerpads.

        Args:
            fingerpad_positions: Tensor of shape (num_envs, 5, 3)

        Returns:
            Dictionary of geometric observations
        """
        geometric_obs = {}

        # Fingerpad pairwise distances (geometric measurement)
        geometric_obs[
            "fingerpad_distances"
        ] = self._compute_fingerpad_pairwise_distances(fingerpad_positions)

        # First three fingerpad centroid (geometric measurement)
        geometric_obs["first_three_fingerpad_centroid"] = torch.mean(
            fingerpad_positions[:, :3, :], dim=1
        )

        return geometric_obs

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
        contact_threshold = self.parent_env.task_cfg["contactBinaryThreshold"]
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
        proximity_threshold = (
            math.sqrt(3) * self.box_size / 2 * 1.2
        )  # 20% margin for noise
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

    def _update_stage_state(self, obs_dict, task_obs, control_dt):
        """
        Update stage state machine and manage transitions.

        This method handles the core state machine logic:
        1. Updates temporal state (time_in_stage counters)
        2. Tracks contact duration for Stage 2 transition evaluation
        3. Evaluates transition conditions and performs stage changes
        4. Exposes stage information to policy observations

        Time Complexity: O(num_envs) - vectorized operations on environment tensors
        """
        # Get stage states (shared across all environments)
        current_stage = self.task_states["current_stage"]  # (num_envs,) tensor
        time_in_stage = self.task_states["time_in_stage"]  # (num_envs,) tensor
        stage_contact_duration = self.task_states[
            "stage_contact_duration"
        ]  # (num_envs,) tensor
        just_transitioned_to_stage2 = self.task_states[
            "just_transitioned_to_stage2"
        ]  # (num_envs,) tensor
        just_transitioned_to_stage3 = self.task_states[
            "just_transitioned_to_stage3"
        ]  # (num_envs,) tensor

        # Update timers - incremental time tracking per environment
        time_in_stage += control_dt

        # Track contact duration for Stage 2 using policy-observable contact only
        # CRITICAL: Uses policy-observable sensors, not privileged finger-box collision detection
        # This ensures policy can learn the transition conditions it will experience during inference
        finger_has_contact = obs_dict[
            "contact_binary"
        ].bool()  # (num_envs, 5) - Policy observable only
        thumb_and_other_contact = finger_has_contact[:, 0] & finger_has_contact[
            :, 1:
        ].any(
            dim=1
        )  # (num_envs,)

        # Update Stage 2 contact duration timer (only for environments in Stage 2)
        stage2_mask = current_stage == 2  # (num_envs,) boolean mask
        stage_contact_duration[stage2_mask] = torch.where(
            thumb_and_other_contact[stage2_mask],
            stage_contact_duration[stage2_mask] + control_dt,  # Accumulate contact time
            torch.zeros_like(
                stage_contact_duration[stage2_mask]
            ),  # Reset if contact lost
        )

        # Check stage transitions - modifies current_stage and time_in_stage in-place
        self._check_stage_transitions(
            current_stage,
            time_in_stage,
            stage_contact_duration,
            just_transitioned_to_stage2,
            just_transitioned_to_stage3,
        )

        # Expose stage information to policy (policy-observable state machine info)
        task_obs["current_stage"] = current_stage.unsqueeze(1).float()  # (num_envs, 1)
        task_obs["time_in_stage"] = time_in_stage.unsqueeze(1)  # (num_envs, 1)
        task_obs["stage_progress"] = self._compute_stage_progress(
            current_stage, time_in_stage
        )  # (num_envs, 1)

    def _check_stage_transitions(
        self,
        current_stage,
        time_in_stage,
        stage_contact_duration,
        just_transitioned_to_stage2,
        just_transitioned_to_stage3,
    ):
        """
        Evaluate and perform stage transitions based on time and success conditions.

        Transition Logic:
        - Stage 1→2: Time-based (allows 4s for positioning)
        - Stage 2→3: Success-based (0.5s contact) OR timeout-based (3s maximum)
        - No reverse transitions (progressive state machine)

        Modifies tensors in-place for efficiency (no tensor allocation).
        Sets exact-step transition flags for precise reward timing.
        Time Complexity: O(num_envs) - vectorized boolean operations
        """
        # Reset transition flags from previous step
        just_transitioned_to_stage2.fill_(False)
        just_transitioned_to_stage3.fill_(False)

        # Stage 1 → Stage 2: Time-based transition after positioning phase
        # Gives policy sufficient time to position hand optimally above object
        stage1_complete = (current_stage == 1) & (
            time_in_stage >= self.stage1_duration
        )  # (num_envs,)

        # Stage 2 → Stage 3: Dual condition transition (success OR timeout)
        # Success path: Policy achieved sustained contact (thumb + other finger)
        stage2_contact_success = (current_stage == 2) & (
            stage_contact_duration >= self.stage2_contact_success_threshold
        )  # (num_envs,)

        # Timeout path: Policy failed to establish contact within time limit
        stage2_timeout = (current_stage == 2) & (
            time_in_stage >= self.stage2_duration
        )  # (num_envs,)

        # Combine both transition conditions (allows reward for success, progression for timeout)
        stage2_complete = stage2_contact_success | stage2_timeout  # (num_envs,)

        # Set exact transition flags for reward/failure detection
        just_transitioned_to_stage2[stage1_complete] = True
        just_transitioned_to_stage3[stage2_complete] = True

        # Apply transitions in-place (modifies shared state tensors)
        # Stage progression: 1→2, 2→3 (no reverse transitions)
        current_stage[stage1_complete] = 2
        current_stage[stage2_complete] = 3

        # Reset timers for newly transitioned environments
        time_in_stage[stage1_complete | stage2_complete] = 0.0
        stage_contact_duration[stage1_complete | stage2_complete] = 0.0

    def _compute_stage_progress(self, current_stage, time_in_stage):
        """Compute normalized stage progress [0,1]."""
        progress = torch.zeros_like(time_in_stage)

        # Stage 1: configured duration
        stage1_mask = current_stage == 1
        progress[stage1_mask] = torch.clamp(
            time_in_stage[stage1_mask] / self.stage1_duration, 0, 1
        )

        # Stage 2: configured duration
        stage2_mask = current_stage == 2
        progress[stage2_mask] = torch.clamp(
            time_in_stage[stage2_mask] / self.stage2_duration, 0, 1
        )

        # Stage 3: No fixed duration
        stage3_mask = current_stage == 3
        progress[stage3_mask] = 1.0  # Always "complete" in final stage

        return progress.unsqueeze(1)

    def compute_task_reward_terms(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute task-specific reward components with unified stage-based logic.

        This method implements a sophisticated stage-based reward system where different
        reward components are active based on the current stage of the state machine.
        Each stage focuses on specific objectives that guide policy learning.

        Stage-based Reward Philosophy:
        - Stage 1: Positioning rewards (height alignment, centroid positioning, stability)
        - Stage 2: Contact rewards (thumb contact, other finger contact, grasp formation)
        - Stage 3: Lifting rewards (object height, grasp maintenance, duration)

        Performance: O(num_envs) with early termination - only active stages compute rewards.
        Memory: Reuses contact computations from obs_dict to prevent duplicate calculations.

        Args:
            obs_dict: Dictionary of observations (includes pre-computed contact states)

        Returns:
            Dictionary of task-specific reward components with stage-based activation
        """
        rewards = {}

        # Get current stage state (shared across all environments)
        current_stage = self.task_states["current_stage"]  # (num_envs,)

        # Create stage-specific boolean masks for reward activation
        stage1_mask = current_stage == 1  # Pre-grasp environments
        stage2_mask = current_stage == 2  # Grasp formation environments
        stage3_mask = current_stage == 3  # Lifting environments

        # Stage-specific reward computation (early termination optimization)
        # Only compute rewards for stages that have active environments
        if stage1_mask.any():
            rewards.update(self._compute_stage1_rewards(obs_dict, stage1_mask))

        if stage2_mask.any():
            rewards.update(self._compute_stage2_rewards(obs_dict, stage2_mask))

        if stage3_mask.any():
            rewards.update(self._compute_stage3_rewards(obs_dict, stage3_mask))

        # Stage completion bonuses: One-time rewards for successful transitions only
        # Uses exact-step detection and validates that transition was successful (not failed)
        just_transitioned_to_stage2 = self.task_states[
            "just_transitioned_to_stage2"
        ]  # (num_envs,)
        just_transitioned_to_stage3 = self.task_states[
            "just_transitioned_to_stage3"
        ]  # (num_envs,)

        # Stage 1→2 completion bonus (successful positioning phase)
        # Only awarded if transition occurred AND pregrasp quality check passed
        stage1_transition_successful = (
            just_transitioned_to_stage2
            & ~self._failure_tensors["stage1_pregrasp_failed"]
        )
        rewards["s1_completion"] = stage1_transition_successful.float()

        # Stage 2→3 completion bonus (successful contact establishment)
        # Only awarded if transition occurred AND real contact was established
        stage2_transition_successful = (
            just_transitioned_to_stage3
            & ~self._failure_tensors["stage2_contact_failed"]
        )
        rewards["s2_completion"] = stage2_transition_successful.float()

        # Penetration prevention penalty (applies to all stages)
        rewards["penetration_penalty"] = self._compute_penetration_penalty(obs_dict)

        return rewards

    def _compute_stage1_rewards(self, obs_dict, stage1_mask):
        """Compute Stage 1 (pre-grasp) rewards."""
        rewards = {}

        # Pre-grasp height alignment reward
        avg_height_diff = obs_dict["avg_finger_to_object_height_diff"]
        height_alignment_reward = torch.exp(
            -self.height_alignment_decay * avg_height_diff
        )
        rewards["s1_height_alignment"] = height_alignment_reward * stage1_mask.float()

        # Pre-grasp centroid positioning reward
        fingerpad_centroid = obs_dict["first_three_fingerpad_centroid"]
        centroid_distance = torch.norm(fingerpad_centroid - self.box_positions, dim=1)
        centroid_positioning_reward = torch.exp(
            -self.centroid_positioning_decay * centroid_distance
        )
        rewards["s1_centroid_positioning"] = (
            centroid_positioning_reward * stage1_mask.float()
        )

        # Object stability reward
        position_drift = torch.norm(
            self.box_positions - self.initial_box_positions[:, :3], dim=1
        )
        velocity_magnitude = torch.norm(self.box_velocities, dim=1)
        stability_reward = torch.exp(
            -self.object_stability_decay * (position_drift + velocity_magnitude)
        )
        rewards["s1_object_stability"] = stability_reward * stage1_mask.float()

        # First three fingerpad height consistency reward
        fingerpad_positions = self._extract_fingerpad_positions(obs_dict)
        first_three_heights = fingerpad_positions[
            :, :3, 2
        ]  # Z coordinates of first 3 fingerpads (thumb, index, middle)
        height_variance = torch.var(
            first_three_heights, dim=1
        )  # Variance across the 3 fingerpads
        height_consistency_reward = torch.exp(
            -self.first_three_height_consistency_decay * height_variance
        )
        rewards["s1_finger_height_consistency"] = (
            height_consistency_reward * stage1_mask.float()
        )

        # Thumb rotation encouragement - reward for r_f_joint1_1 near 90°
        target_thumb_rotation = math.pi / 2  # 90 degrees
        current_thumb_rotation = self.observation_encoder.get_raw_finger_dof(
            "r_f_joint1_1", "pos", obs_dict
        )
        thumb_rotation_error = torch.abs(current_thumb_rotation - target_thumb_rotation)
        thumb_rotation_reward = torch.exp(
            -5.0 * thumb_rotation_error
        )  # Exponential decay for precise targeting
        rewards["s1_thumb_rotation"] = thumb_rotation_reward * stage1_mask.float()

        return rewards

    def _compute_stage2_rewards(self, obs_dict, stage2_mask):
        """Compute Stage 2 (grasp) rewards using contact state from obs_dict."""
        rewards = {}

        # Extract contact state from obs_dict (computed once in get_task_observations)
        thumb_contact = obs_dict["thumb_contact"].squeeze(-1).bool()
        other_fingers_contact = obs_dict["other_fingers_contact"].squeeze(-1).bool()
        grasp_state = obs_dict["grasp_state"].squeeze(-1).bool()

        # Binary contact rewards (existing)
        rewards["s2_thumb_contact"] = thumb_contact.float() * stage2_mask.float()
        rewards["s2_other_fingers_contact"] = (
            other_fingers_contact.float() * stage2_mask.float()
        )
        rewards["s2_grasp_achievement"] = grasp_state.float() * stage2_mask.float()

        # Fingerpad proximity reward with penetration protection
        _, min_distances = self._detect_geometric_penetration(obs_dict)
        min_box_half_size = self.box_size / 2.0

        # Clamp minimum distance to prevent penetration exploitation
        min_reward_distance = min_box_half_size * self.proximity_min_distance_factor
        safe_distances = torch.clamp(min_distances, min=min_reward_distance)
        proximity_reward = torch.exp(-self.fingerpad_proximity_decay * safe_distances)
        rewards["s2_fingerpad_proximity"] = proximity_reward * stage2_mask.float()

        # Hand base stability reward - encourage smooth base motion during contact establishment
        base_dof_velocities = obs_dict["base_dof_vel"]  # Shape: (num_envs, 6)
        base_velocity_magnitude = torch.norm(
            base_dof_velocities, dim=1
        )  # L2 norm across 6 DOFs
        base_stability_reward = torch.exp(
            -self.base_stability_decay * base_velocity_magnitude
        )
        rewards["s2_base_stability"] = base_stability_reward * stage2_mask.float()

        return rewards

    def _compute_stage3_rewards(self, obs_dict, stage3_mask):
        """Compute Stage 3 (lift) rewards using contact state from obs_dict."""
        rewards = {}

        # Extract grasp state from obs_dict (computed once in get_task_observations)
        grasp_state = obs_dict["grasp_state"].squeeze(-1).bool()

        # Object height reward - encourage lifting
        height_above_table = self.box_positions[:, 2] - self.box_z
        height_reward = torch.clamp(
            height_above_table / (self.height_threshold - self.box_z), 0, 1
        )
        rewards["s3_object_height"] = height_reward * stage3_mask.float()

        # Grasp maintenance reward using contact state from obs_dict
        rewards["s3_grasp_maintenance"] = grasp_state.float() * stage3_mask.float()

        # Grasp duration reward
        grasp_duration = obs_dict["grasp_duration"].squeeze(-1)
        grasp_duration_reward = torch.clamp(
            grasp_duration / self.contact_duration_threshold_seconds, 0, 1
        )
        rewards["s3_grasp_duration"] = grasp_duration_reward * stage3_mask.float()

        return rewards

    def _detect_geometric_penetration(self, obs_dict):
        """Detect fingertip penetration inside box volume."""
        fingertip_positions = obs_dict["fingertip_poses_world"].view(-1, 5, 7)[:, :, :3]
        box_positions_expanded = self.box_positions.unsqueeze(1)  # (num_envs, 1, 3)

        # Calculate distance from each fingertip to box center
        distances = torch.norm(fingertip_positions - box_positions_expanded, dim=-1)
        min_distances = distances.min(dim=1)[0]  # Closest fingertip per env

        # Configurable geometric penetration threshold
        penetration_threshold = (
            self.box_size / 2.0
        ) * self.geometric_penetration_factor
        penetration_detected = min_distances < penetration_threshold
        return penetration_detected, min_distances

    def _compute_penetration_penalty(self, obs_dict):
        """Continuous penalty proportional to penetration depth."""
        geometric_penetration, min_distances = self._detect_geometric_penetration(
            obs_dict
        )

        # Calculate penetration depth (how far inside the box)
        penetration_threshold = (
            self.box_size / 2.0
        ) * self.geometric_penetration_factor
        penetration_depth = torch.clamp(penetration_threshold - min_distances, min=0.0)

        # Continuous penalty: larger penetration = larger penalty
        penalty = penetration_depth * self.penetration_depth_scale
        return penalty

    def check_task_success_criteria(
        self, obs_dict: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
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
        success_duration_steps = self.task_states["success_duration_steps"]

        # Success requires maintained conditions for threshold duration
        success = success_duration_steps >= self.contact_duration_threshold_steps

        return {"grasp_lift_success": success}

    def check_task_failure_criteria(
        self, obs_dict: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Check task-specific failure criteria with comprehensive stage-based validation.

        This method implements a multi-layered failure detection system where ALL failure
        criteria are explicitly declared in activeFailureCriteria configuration. Uses both
        policy-observable information and privileged information to provide accurate
        failure detection while maintaining the blind policy design.

        Failure Detection Philosophy:
        - Global failures: Basic physics violations (hand too far, hitting ground)
        - Stage-specific failures: Quality validation using privileged information
        - Transition-based evaluation: Check failures only when transitioning stages
        - Complete transparency: All failures declared in configuration

        Key Design: Policy sees stage transitions but not the privileged failure reasons.
        This maintains the blind design while providing accurate training signals.

        Args:
            obs_dict: Dictionary of observations with pre-computed contact states
                     (Always provided by step_processor - no fallback needed)

        Returns:
            Dictionary with failure criteria:
            - Global: box_too_far (physics-based), hitting_ground (handled by base class)
            - Stage 1: stage1_pregrasp_failed (quality assessment)
            - Stage 2: stage2_contact_failed (privileged contact validation)
            - Stage 3: stage3_grasp_lost (privileged grasp validation)
        """
        failures = {}

        # Global failure detection using physics simulation state
        hand_to_box_distance = obs_dict[
            "hand_to_object_distance"
        ]  # Pre-computed distance
        failures["box_too_far"] = (
            hand_to_box_distance > self.max_box_distance
        )  # Hand moved beyond reach

        # Stage-based failure detection using privileged information
        # CRITICAL: These failures use privileged info for accurate assessment
        # but are triggered by exact stage transitions that policy can observe
        current_stage = self.task_states["current_stage"]  # (num_envs,)
        just_transitioned_to_stage2 = self.task_states[
            "just_transitioned_to_stage2"
        ]  # (num_envs,)
        just_transitioned_to_stage3 = self.task_states[
            "just_transitioned_to_stage3"
        ]  # (num_envs,)

        # Reset preallocated failure tensors for memory efficiency
        # Prevents repeated tensor allocations during failure detection
        for tensor in self._failure_tensors.values():
            tensor.fill_(False)

        # Stage 1 Failure: Evaluate pregrasp quality when transitioning to Stage 2
        # Uses privileged information (box pose, fingerpad positions) to assess positioning quality
        if just_transitioned_to_stage2.any():
            # Privileged evaluation: height alignment, centroid positioning, object stability
            pregrasp_quality = self._evaluate_pregrasp_quality(
                obs_dict
            )  # Uses box pose (privileged)
            self._failure_tensors["stage1_pregrasp_failed"][:] = (
                just_transitioned_to_stage2 & ~pregrasp_quality
            )

        # Stage 2 Failure: Validate contact establishment when transitioning to Stage 3 via timeout
        # Distinguishes between successful contact (early transition) vs timeout (potential failure)
        if just_transitioned_to_stage3.any():
            # Privileged validation: actual finger-box collision vs policy-observable contact sensors
            grasp_state = (
                obs_dict["grasp_state"].squeeze(-1).bool()
            )  # True finger-box contact (privileged)
            had_real_contact = grasp_state  # Prevents false failures from sensor noise
            self._failure_tensors["stage2_contact_failed"][:] = (
                just_transitioned_to_stage3 & ~had_real_contact
            )

        # Stage 3 Failure: Lost grasp state during lifting phase
        # Continuous monitoring using privileged finger-box contact validation
        stage3_mask = current_stage == 3
        if stage3_mask.any():
            # Privileged monitoring: true grasp state vs policy-observable sensors
            grasp_state = (
                obs_dict["grasp_state"].squeeze(-1).bool()
            )  # Accurate grasp detection
            self._failure_tensors["stage3_grasp_lost"][:] = stage3_mask & ~grasp_state

        # Combine preallocated failure tensors with global failures
        # Memory efficient: reuses allocated tensors rather than creating new ones
        failures.update(self._failure_tensors)

        return failures

    def _evaluate_pregrasp_quality(self, obs_dict):
        """Evaluate pregrasp position quality using privileged information and reusing computed data."""
        # Height alignment check (first 3 fingerpads) - reuse computed positions
        fingerpad_positions = self._extract_fingerpad_positions(obs_dict)
        fingerpad_heights = fingerpad_positions[:, :3, 2]  # First 3 fingerpads only

        object_height = self.box_positions[:, 2]
        height_diff = torch.abs(fingerpad_heights - object_height.unsqueeze(1))
        height_check = torch.all(
            height_diff <= self.height_tolerance, dim=1
        )  # All within tolerance

        # Centroid positioning check - reuse computed centroid from obs_dict
        first_three_centroid = obs_dict[
            "first_three_fingerpad_centroid"
        ]  # Already computed in get_task_observations
        centroid_distance = torch.norm(first_three_centroid - self.box_positions, dim=1)
        centroid_check = (
            centroid_distance <= self.centroid_tolerance
        )  # Within tolerance

        # Object stability check
        position_drift = torch.norm(
            self.box_positions - self.initial_box_positions[:, :3], dim=1
        )
        velocity_magnitude = torch.norm(self.box_velocities, dim=1)
        stability_check = (position_drift <= self.position_drift_tolerance) & (
            velocity_magnitude <= self.velocity_tolerance
        )

        return height_check & centroid_check & stability_check
