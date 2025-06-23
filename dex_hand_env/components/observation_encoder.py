"""
Observation encoder component for DexHand environment.

This module provides observation encoding functionality for the DexHand environment,
including proprioceptive states, sensor readings, and task-specific observations.
"""

# Import standard libraries
import torch
from typing import Dict, List, Tuple, Union
from loguru import logger

# Import gym for spaces
import gym

# Import IsaacGym
from isaacgym.torch_utils import quat_mul, quat_conjugate

# Import utilities
from dex_hand_env.utils.coordinate_transforms import point_in_hand_frame

# Import constants
from dex_hand_env.constants import NUM_BASE_DOFS, NUM_ACTIVE_FINGER_DOFS


class ObservationEncoder:
    """
    Encodes observations for the DexHand environment.

    This component provides functionality to:
    - Generate proprioceptive observations (joint positions, velocities)
    - Process sensor readings (contact forces, tactile)
    - Create task-specific observations
    - Normalize and format observations for the policy

    The new design follows a cleaner architecture:
    1. Compute a dictionary of "default" observations
    2. Call a function to compute task-specific observations
    3. Merge the two dictionaries
    4. Concat tensors with selected keys into final observation buffer
    """

    def __init__(self, parent):
        """
        Initialize the observation encoder.

        Args:
            parent: Parent DexHandBase instance
        """
        self.parent = parent
        self.gym = parent.gym
        self.sim = parent.sim

        # Constants for observation dimensions (imported from constants.py)
        self.NUM_BASE_DOFS = NUM_BASE_DOFS
        self.NUM_ACTIVE_FINGER_DOFS = NUM_ACTIVE_FINGER_DOFS
        self.NUM_FINGERS = 5  # TODO: Import from constants.py

        # Configuration - will be set during initialize()
        self.observation_keys = []
        self.num_observations = 0

        # Initialize observation buffers
        self.obs_buf = None
        self.states_buf = None

        # Previous actions for observation (if enabled)
        self.prev_actions = None

        # Manual velocity computation (to replace unreliable Isaac Gym velocities)
        self.prev_dof_pos = None
        # control_dt is accessed via property decorator from physics_manager

    @property
    def num_envs(self):
        """Access num_envs from parent (single source of truth)."""
        return self.parent.num_envs

    @property
    def device(self):
        """Access device from parent (single source of truth)."""
        return self.parent.device

    @property
    def tensor_manager(self):
        """Access tensor_manager from parent (single source of truth)."""
        return self.parent.tensor_manager

    @property
    def hand_initializer(self):
        """Access hand_initializer from parent (single source of truth)."""
        return self.parent.hand_initializer

    @property
    def physics_manager(self):
        """Access physics_manager from parent (single source of truth)."""
        return self.parent.physics_manager

    @property
    def control_dt(self):
        """Access control_dt from physics manager (single source of truth)."""
        if self.physics_manager is None:
            raise RuntimeError("physics_manager not set. Cannot access control_dt.")
        return self.physics_manager.control_dt

    @property
    def dof_names(self):
        """Access DOF names from hand_initializer (single source of truth)."""
        return self.parent.hand_initializer.dof_names

    def initialize(
        self,
        observation_keys: List[str],
        joint_to_control: Dict[str, str],
        active_joint_names: List[str],
        num_actions: int = None,
        action_processor=None,
        index_mappings: Dict = None,
    ):
        """
        Initialize the observation encoder with configuration.

        Note: Rigid body indices (hand_indices, fingertip_indices, fingerpad_indices) are
        now accessed directly from hand_initializer to maintain single source of truth.

        Args:
            observation_keys: List of observation components to include
            joint_to_control: Mapping from joint names to control names
            active_joint_names: List of active joint names
            num_actions: Actual number of actions in the action space
            action_processor: Reference to action processor for accessing DOF targets
            index_mappings: Dictionary of index mappings for tensor access
        """
        self.observation_keys = observation_keys
        self.joint_to_control = joint_to_control
        self.active_joint_names = active_joint_names
        self.action_processor = action_processor

        # hand_initializer is now a required parameter

        # Pre-compute active finger DOF indices for efficient observation extraction
        self.active_finger_dof_indices = self._compute_active_finger_dof_indices()

        # Store index mappings from DexHandBase
        if index_mappings:
            if "base_joint_to_index" not in index_mappings:
                raise RuntimeError(
                    "base_joint_to_index mapping not found in index_mappings"
                )
            if "control_name_to_index" not in index_mappings:
                raise RuntimeError(
                    "control_name_to_index mapping not found in index_mappings"
                )
            if "raw_dof_name_to_index" not in index_mappings:
                raise RuntimeError(
                    "raw_dof_name_to_index mapping not found in index_mappings"
                )
            if "finger_body_to_index" not in index_mappings:
                raise RuntimeError(
                    "finger_body_to_index mapping not found in index_mappings"
                )

            self.base_joint_to_index = index_mappings["base_joint_to_index"]
            self.control_name_to_index = index_mappings["control_name_to_index"]
            self.raw_dof_name_to_index = index_mappings["raw_dof_name_to_index"]
            self.finger_body_to_index = index_mappings["finger_body_to_index"]
        else:
            raise RuntimeError(
                "index_mappings not provided to ObservationEncoder. These mappings are required for tensor indexing."
            )

        # Initialize previous actions tensor - always available for inspection
        if num_actions is None:
            raise RuntimeError(
                "num_actions not provided to ObservationEncoder initialization. Cannot determine action space size."
            )
        self.prev_actions = torch.zeros(
            (self.num_envs, num_actions), device=self.device
        )

        # Compute observation dimension dynamically by creating a test observation
        test_obs_dict = self._compute_default_observations()
        test_task_obs_dict = self._compute_task_observations(test_obs_dict)
        merged_obs_dict = {**test_obs_dict, **test_task_obs_dict}

        # Log dimensions of each observation component
        logger.debug("Observation component dimensions:")
        total_dim = 0
        for key in self.observation_keys:
            if key in merged_obs_dict:
                tensor = merged_obs_dict[key]
                if len(tensor.shape) > 2:
                    tensor = tensor.reshape(self.num_envs, -1)
                dim = tensor.shape[1]
                total_dim += dim
                logger.debug(f"  {key}: {dim}")
            else:
                logger.warning(f"  {key}: MISSING")

        test_obs_tensor = self._concat_selected_observations(merged_obs_dict)

        self.num_observations = test_obs_tensor.shape[1]
        logger.debug(f"Total observation dimension: {total_dim}")
        logger.info(
            f"ObservationEncoder initialized with dynamic observation dimension: {self.num_observations}"
        )

        # Initialize observation buffers
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_observations), device=self.device
        )
        self.states_buf = torch.zeros(
            (self.num_envs, self.num_observations), device=self.device
        )

        # Build contact force body name to index mapping
        self.contact_force_body_name_to_index = {}
        if hasattr(self.hand_initializer, "contact_force_body_names"):
            for i, body_name in enumerate(
                self.hand_initializer.contact_force_body_names
            ):
                self.contact_force_body_name_to_index[body_name] = i

    @property
    def hand_index(self):
        """Access hand rigid body index from parent (single source of truth)."""
        return self.parent.hand_rigid_body_index

    @property
    def fingertip_indices(self):
        """Access fingertip indices from hand_initializer (single source of truth)."""
        if self.hand_initializer is None:
            raise RuntimeError("hand_initializer not set in ObservationEncoder")
        # Get global indices from first environment
        if not self.hand_initializer.fingertip_indices:
            return []

        global_indices = self.hand_initializer.fingertip_indices[0]
        if not global_indices:
            return []

        # Convert global indices to local indices for tensor indexing
        # Same logic as fingerpad_indices
        rigid_body_states = self.tensor_manager.rigid_body_states
        if rigid_body_states is not None and len(rigid_body_states.shape) >= 2:
            num_bodies_per_env = rigid_body_states.shape[1]
            # Convert global indices to local by taking modulo
            local_indices = [idx % num_bodies_per_env for idx in global_indices]
            return local_indices
        else:
            # Fallback: assume indices are already local (for single environment)
            return global_indices

    @property
    def fingerpad_indices(self):
        """Access fingerpad indices from hand_initializer (single source of truth)."""
        if self.hand_initializer is None:
            raise RuntimeError("hand_initializer not set in ObservationEncoder")
        # Get global indices from first environment
        if not self.hand_initializer.fingerpad_indices:
            return []

        global_indices = self.hand_initializer.fingerpad_indices[0]
        if not global_indices:
            return []

        # Convert global indices to local indices for tensor indexing
        # All environments have the same local structure, so we can use env 0's indices
        # but we need to convert them to be relative to the environment
        # Since rigid_body_states has shape (num_envs, num_bodies_per_env, ...),
        # we need local indices within each environment

        # Get the number of rigid bodies per environment
        # This is determined by looking at the rigid body state tensor shape
        rigid_body_states = self.tensor_manager.rigid_body_states
        if rigid_body_states is not None and len(rigid_body_states.shape) >= 2:
            num_bodies_per_env = rigid_body_states.shape[1]
            # Convert global indices to local by taking modulo
            local_indices = [idx % num_bodies_per_env for idx in global_indices]
            return local_indices
        else:
            # Fallback: assume indices are already local (for single environment)
            return global_indices

    # set_control_dt method removed - control_dt now accessed via property decorator from physics_manager

    def update_prev_actions(self, actions: torch.Tensor):
        """
        Update previous actions for observation.

        Args:
            actions: Current action tensor
        """
        # prev_actions is initialized in initialize() and should never be None
        # actions parameter is required and should be a valid tensor
        self.prev_actions = actions.clone()

    def _compute_manual_velocities(self, current_dof_pos: torch.Tensor) -> torch.Tensor:
        """
        Compute DOF velocities manually using position differences.

        This replaces the unreliable Isaac Gym velocity readings with numerical differentiation.

        Args:
            current_dof_pos: Current DOF positions tensor (num_envs, num_dofs)

        Returns:
            Computed velocities tensor (num_envs, num_dofs)
        """
        if self.prev_dof_pos is None or self.control_dt is None:
            # First step - initialize with zeros
            self.prev_dof_pos = current_dof_pos.clone()
            return torch.zeros_like(current_dof_pos)

        # Compute velocity as (current_pos - prev_pos) / dt
        velocity = (current_dof_pos - self.prev_dof_pos) / self.control_dt

        # Update previous positions for next iteration
        self.prev_dof_pos = current_dof_pos.clone()

        return velocity

    def compute_observations(
        self, exclude_components: List[str] = None
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Compute the observation vector and dictionary.

        Args:
            exclude_components: List of component names to exclude from computation
                               If provided, returns only obs_dict without concatenation

        Returns:
            If exclude_components is provided: observation dictionary only
            Otherwise: Tuple of (observation tensor, observation dictionary)
        """
        # Step 1: Compute default observations
        default_obs_dict = self._compute_default_observations()

        # Step 2: Compute task-specific observations
        task_obs_dict = self._compute_task_observations(default_obs_dict)

        # Step 3: Merge dictionaries
        merged_obs_dict = {**default_obs_dict, **task_obs_dict}

        # If excluding components, return dict only (no concatenation)
        if exclude_components:
            # Remove excluded components from dict
            for component in exclude_components:
                merged_obs_dict.pop(component, None)
            return merged_obs_dict

        # Step 4: Concat selected observations into final observation buffer
        self.obs_buf = self._concat_selected_observations(merged_obs_dict)

        # Copy to states buffer (for asymmetric actor-critic)
        self.states_buf = self.obs_buf.clone()

        return self.obs_buf, merged_obs_dict

    def concatenate_observations(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Concatenate observation dictionary into tensor.

        Args:
            obs_dict: Pre-computed observation dictionary

        Returns:
            Concatenated observation tensor
        """
        self.obs_buf = self._concat_selected_observations(obs_dict)
        self.states_buf = self.obs_buf.clone()
        return self.obs_buf

    def _compute_active_finger_dof_indices(self) -> torch.Tensor:
        """
        Pre-compute indices mapping from full DOF tensor to active finger DOFs.

        For controls that map to multiple DOFs (like ff_spr), this uses the PRIMARY DOF
        (the first one in the joint list) to represent the control in observations.

        Returns:
            torch.Tensor of shape (num_active_finger_dofs,) with DOF indices
        """
        if not self.dof_names:
            raise RuntimeError(
                "DOF names not available from asset. Cannot compute active finger DOF indices."
            )

        if not self.active_joint_names:
            raise RuntimeError(
                "active_joint_names not provided. Cannot compute active finger DOF indices."
            )

        # Import HardwareMapping to get the authoritative control-to-joint mapping
        from dex_hand_env.components.hand_initializer import HardwareMapping

        # Pre-build mappings for vectorized lookup
        # Create dict of control_name -> HardwareMapping
        control_to_hardware = {}
        for member in HardwareMapping:
            control_to_hardware[member.control_name] = member

        # Create dict of DOF name -> index for fast lookup
        finger_dof_names = self.dof_names[self.NUM_BASE_DOFS :]
        dof_name_to_idx = {
            name: idx + self.NUM_BASE_DOFS for idx, name in enumerate(finger_dof_names)
        }

        # Create array to store DOF indices for each active control
        active_indices = torch.full(
            (len(self.active_joint_names),),
            -1,
            dtype=torch.long,
            device=self.tensor_manager.device,
        )

        # Vectorized mapping: build all indices at once
        for control_idx, control_name in enumerate(self.active_joint_names):
            # Get hardware mapping
            hardware_mapping = control_to_hardware.get(control_name)
            if hardware_mapping is None:
                raise RuntimeError(
                    f"Control '{control_name}' not found in HardwareMapping"
                )

            # Use the PRIMARY DOF (first joint) to represent this control
            primary_joint = hardware_mapping.joint_names[0]

            # Direct lookup instead of loop
            dof_idx = dof_name_to_idx.get(primary_joint)
            if dof_idx is None:
                raise RuntimeError(
                    f"Primary joint '{primary_joint}' for control '{control_name}' not found in DOF names"
                )

            active_indices[control_idx] = dof_idx
            logger.debug(
                f"Mapped control '{control_name}' -> primary joint '{primary_joint}' -> DOF index {dof_idx}"
            )

        # Check that all active controls have been mapped
        valid_mask = active_indices >= 0
        if not valid_mask.all():
            missing_indices = torch.nonzero(~valid_mask).flatten()
            missing_controls = [self.active_joint_names[idx] for idx in missing_indices]
            raise RuntimeError(
                f"Failed to map active finger controls to DOF indices: {missing_controls}"
            )

        return active_indices

    def _compute_default_observations(self) -> Dict[str, torch.Tensor]:
        """
        Compute default observations dictionary.

        Returns:
            Dictionary of default observations
        """
        obs_dict = {}

        # Access tensors directly from tensor manager (no caching needed)
        dof_pos = self.tensor_manager.dof_pos
        isaac_dof_vel = (
            self.tensor_manager.dof_vel
        )  # Original Isaac Gym velocities (unreliable)
        actor_root_state_tensor = self.tensor_manager.actor_root_state_tensor
        rigid_body_states = self.tensor_manager.rigid_body_states
        contact_forces = self.tensor_manager.contact_forces

        # Tensors should never be None during observation computation
        # If they are, that indicates a tensor initialization bug that must be fixed
        if dof_pos is None:
            raise RuntimeError(
                "dof_pos tensor is None. This indicates tensor_manager was not properly initialized."
            )
        if isaac_dof_vel is None:
            raise RuntimeError(
                "dof_vel tensor is None. This indicates tensor_manager was not properly initialized."
            )
        if actor_root_state_tensor is None:
            raise RuntimeError(
                "actor_root_state_tensor is None. This indicates tensor_manager was not properly initialized."
            )

        # Compute manual velocities to replace unreliable Isaac Gym velocities
        dof_vel = self._compute_manual_velocities(dof_pos)

        # Base DOF positions (6 DOFs: x, y, z, rx, ry, rz)
        obs_dict["base_dof_pos"] = dof_pos[:, : self.NUM_BASE_DOFS]

        # Base DOF velocities (6 DOFs: x, y, z, rx, ry, rz)
        obs_dict["base_dof_vel"] = dof_vel[:, : self.NUM_BASE_DOFS]

        # Active finger DOF positions (12 active finger controls) - for RL observation tensor
        obs_dict["active_finger_dof_pos"] = dof_pos[:, self.active_finger_dof_indices]

        # Active finger DOF velocities (12 active finger controls) - for RL observation tensor
        obs_dict["active_finger_dof_vel"] = dof_vel[:, self.active_finger_dof_indices]

        # All finger DOF positions (20 finger DOFs: excluding base 6 DOFs, r_f_joint3_1 is fixed)
        # Always available in obs_dict for semantic access, but not included in RL tensor by default
        all_finger_dof_indices = torch.arange(
            self.NUM_BASE_DOFS, dof_pos.shape[1], device=self.device
        )
        obs_dict["all_finger_dof_pos"] = dof_pos[:, all_finger_dof_indices]
        obs_dict["all_finger_dof_vel"] = dof_vel[:, all_finger_dof_indices]

        # Hand pose (position and orientation)
        # hand_indices and rigid_body_states should never be None/empty during observation computation
        # If they are, that indicates an initialization bug that must be fixed
        # Use constant index (same across all envs)
        hand_base_idx = self.hand_index

        if hand_base_idx < 0 or hand_base_idx >= rigid_body_states.shape[1]:
            raise RuntimeError(
                f"Invalid hand base rigid body index {hand_base_idx}. Rigid body states shape: {rigid_body_states.shape}"
            )

        # Vectorized extraction: get hand base pose for all envs at once
        # rigid_body_states shape: (num_envs, num_bodies_per_env, 13)
        hand_poses = rigid_body_states[
            :, hand_base_idx, :7
        ]  # Extract pos(3) + quat(4) for all envs
        obs_dict["hand_pose"] = hand_poses

        # ARR-aligned hand pose (orientation aligned with ARRx/ARRy/ARRz DOFs)
        # Due to the floating hand model design, the hand is mounted with a built-in 90° Y-axis rotation.
        # When ARRx=ARRy=ARRz=0, the hand base has quaternion [0, sqrt(0.5), 0, sqrt(0.5)] instead of
        # identity [0, 0, 0, 1]. This observation compensates for that built-in rotation to provide
        # orientation values that directly correspond to the ARRx, ARRy, ARRz DOF values.
        arr_aligned_poses = self._compute_arr_aligned_pose(hand_poses)
        obs_dict["hand_pose_arr_aligned"] = arr_aligned_poses

        # Contact forces (3D force for each finger)
        # contact_forces should never be None during observation computation
        # If it is None, that indicates a tensor initialization bug that must be fixed
        flat_contacts = contact_forces.reshape(self.num_envs, -1)
        obs_dict["contact_forces"] = flat_contacts

        # Previous actions
        if self.prev_actions is None:
            raise RuntimeError(
                "prev_actions tensor not initialized. This indicates a programming error in ObservationEncoder initialization."
            )
        obs_dict["prev_actions"] = self.prev_actions

        # Active previous targets (18D: 6 base + 12 finger)
        if (
            hasattr(self.action_processor, "active_prev_targets")
            and self.action_processor.active_prev_targets is not None
        ):
            obs_dict["active_prev_targets"] = self.action_processor.active_prev_targets

        # Active rule targets (18D: 6 base + 12 finger) - will be added by DexHandBase
        # This is computed after pre-action rule is applied, so not available here

        # Base DOF targets (6 DOFs: x, y, z, rx, ry, rz)
        # action_processor and its current_targets must be initialized
        # If they are not, that indicates an initialization bug that must be fixed
        if self.action_processor.full_dof_targets is None:
            raise RuntimeError(
                "ActionProcessor full_dof_targets is None. This indicates action_processor was not properly initialized."
            )

        obs_dict["base_dof_target"] = self.action_processor.full_dof_targets[
            :, : self.NUM_BASE_DOFS
        ]

        # Active finger DOF targets (12 active finger controls) - for RL observation tensor
        obs_dict["active_finger_dof_target"] = self.action_processor.full_dof_targets[
            :, self.active_finger_dof_indices
        ]

        # All finger DOF targets (20 finger DOFs) - always available for semantic access
        obs_dict["all_finger_dof_target"] = self.action_processor.full_dof_targets[
            :, all_finger_dof_indices
        ]

        # Contact force magnitude (magnitude for each finger)
        # contact_forces should never be None during observation computation (same tensor as above)
        contact_magnitudes = torch.norm(
            contact_forces, dim=2
        )  # Shape: (num_envs, num_fingers)
        obs_dict["contact_force_magnitude"] = contact_magnitudes

        # Fingertip poses in world frame (5 fingers × 7 pose dimensions = 35)
        fingertip_poses_world = self._extract_fingertip_poses_world()
        obs_dict["fingertip_poses_world"] = fingertip_poses_world

        # Fingertip poses in hand frame (5 fingers × 7 pose dimensions = 35)
        fingertip_poses_hand = self._extract_fingertip_poses_hand()
        obs_dict["fingertip_poses_hand"] = fingertip_poses_hand

        # Fingerpad poses in world frame (5 fingers × 7 pose dimensions = 35)
        fingerpad_poses_world = self._extract_fingerpad_poses_world()
        obs_dict["fingerpad_poses_world"] = fingerpad_poses_world

        # Fingerpad poses in hand frame (5 fingers × 7 pose dimensions = 35)
        fingerpad_poses_hand = self._extract_fingerpad_poses_hand()
        obs_dict["fingerpad_poses_hand"] = fingerpad_poses_hand

        return obs_dict

    def _compute_task_observations(
        self, default_obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute task-specific observations.

        This method can be overridden by specific tasks to add custom observations.
        The base implementation returns an empty dictionary.

        Args:
            default_obs_dict: Dictionary of default observations

        Returns:
            Dictionary of task-specific observations
        """
        # Base implementation returns empty dict
        # Specific tasks can override this method to add custom observations
        return {}

    def _concat_selected_observations(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Concatenate selected observations into final observation tensor.
        Records slice indices for each component during concatenation.

        Args:
            obs_dict: Dictionary of all available observations

        Returns:
            Concatenated observation tensor
        """
        obs_tensors = []

        # Initialize slice indices dictionary if not exists
        if not hasattr(self, "component_slice_indices"):
            self.component_slice_indices = {}

        current_idx = 0

        # Concat observations in the order specified by observation_keys
        for key in self.observation_keys:
            if key in obs_dict:
                tensor = obs_dict[key]
                # Ensure tensor is 2D (num_envs, obs_dim)
                if len(tensor.shape) > 2:
                    tensor = tensor.reshape(self.num_envs, -1)

                # Record slice indices for this component
                component_size = tensor.shape[1]
                self.component_slice_indices[key] = (
                    current_idx,
                    current_idx + component_size,
                )
                current_idx += component_size

                obs_tensors.append(tensor)
            else:
                logger.warning(
                    f"Observation key '{key}' not found in observation dictionary"
                )

        if obs_tensors:
            return torch.cat(obs_tensors, dim=1)
        else:
            return torch.zeros((self.num_envs, 0), device=self.device)

    def _extract_fingertip_poses_world(self) -> torch.Tensor:
        """
        Extract fingertip poses in world frame.

        Returns:
            torch.Tensor of shape (num_envs, 35) with 5 fingers × 7 pose dimensions
        """
        rigid_body_states = self.tensor_manager.rigid_body_states
        if rigid_body_states is None:
            raise RuntimeError(
                "Rigid body states tensor not initialized. Cannot compute fingertip poses."
            )
        if self.fingertip_indices is None:
            raise RuntimeError(
                "Fingertip indices not initialized. Cannot compute fingertip poses."
            )

        # Extract all fingertip poses at once - fingertip_indices should be shape (5,)
        fingertip_poses = rigid_body_states[
            :, self.fingertip_indices, :7
        ]  # (num_envs, 5, 7)
        poses = fingertip_poses.reshape(self.num_envs, 35)

        return poses

    def _extract_fingerpad_poses_world(self) -> torch.Tensor:
        """
        Extract fingerpad poses in world frame.

        Returns:
            torch.Tensor of shape (num_envs, 35) with 5 fingers × 7 pose dimensions
        """
        rigid_body_states = self.tensor_manager.rigid_body_states
        if rigid_body_states is None:
            raise RuntimeError(
                "Rigid body states tensor not initialized. Cannot compute fingerpad poses."
            )
        if self.fingerpad_indices is None:
            raise RuntimeError(
                "Fingerpad indices not initialized. Cannot compute fingerpad poses."
            )

        # Extract all fingerpad poses at once - fingerpad_indices should be shape (5,)
        fingerpad_poses = rigid_body_states[
            :, self.fingerpad_indices, :7
        ]  # (num_envs, 5, 7)
        poses = fingerpad_poses.reshape(self.num_envs, 35)

        return poses

    def _extract_fingertip_poses_hand(self) -> torch.Tensor:
        """
        Extract fingertip poses in hand reference frame.

        Returns:
            torch.Tensor of shape (num_envs, 35) with 5 fingers × 7 pose dimensions
        """
        # Get fingertip poses in world frame
        fingertip_poses_world = self._extract_fingertip_poses_world()

        # Transform to hand frame
        fingertip_poses_hand = self._transform_poses_to_hand_frame(
            fingertip_poses_world
        )

        return fingertip_poses_hand

    def _extract_fingerpad_poses_hand(self) -> torch.Tensor:
        """
        Extract fingerpad poses in hand reference frame.

        Returns:
            torch.Tensor of shape (num_envs, 35) with 5 fingers × 7 pose dimensions
        """
        # Get fingerpad poses in world frame
        fingerpad_poses_world = self._extract_fingerpad_poses_world()

        # Transform to hand frame
        fingerpad_poses_hand = self._transform_poses_to_hand_frame(
            fingerpad_poses_world
        )

        return fingerpad_poses_hand

    def _transform_poses_to_hand_frame(self, poses_world: torch.Tensor) -> torch.Tensor:
        """
        Transform poses from world frame to hand reference frame.

        Args:
            poses_world: tensor of shape (num_envs, 35) with poses in world frame

        Returns:
            torch.Tensor of shape (num_envs, 35) with poses in hand frame
        """
        rigid_body_states = self.tensor_manager.rigid_body_states
        # Fail fast - these should never be None during normal operation
        if rigid_body_states is None:
            raise RuntimeError(
                "rigid_body_states is None - this indicates tensor_manager initialization bug"
            )
        # Get hand base index - use constant index
        hand_base_idx = self.hand_index
        if hand_base_idx is None:
            raise RuntimeError(
                "hand_index not properly initialized - this indicates initialization bug"
            )

        if hand_base_idx < 0 or hand_base_idx >= rigid_body_states.shape[1]:
            raise RuntimeError(
                f"Invalid hand base rigid body index {hand_base_idx}. Rigid body states shape: {rigid_body_states.shape}"
            )

        # Get hand base pose for all envs at once
        hand_pos = rigid_body_states[:, hand_base_idx, :3]  # (num_envs, 3)
        hand_quat = rigid_body_states[:, hand_base_idx, 3:7]  # (num_envs, 4)

        # Reshape poses for vectorized transformation
        # poses_world shape: (num_envs, 35) = (num_envs, 5 fingers * 7)
        poses_world_reshaped = poses_world.view(self.num_envs, 5, 7)

        # Extract all finger positions and quaternions at once
        finger_pos_world = poses_world_reshaped[:, :, :3]  # (num_envs, 5, 3)
        finger_quat_world = poses_world_reshaped[:, :, 3:7]  # (num_envs, 5, 4)

        # Reshape for batch processing: flatten fingers into batch dimension
        finger_pos_world_flat = finger_pos_world.reshape(-1, 3)  # (num_envs * 5, 3)
        finger_quat_world_flat = finger_quat_world.reshape(-1, 4)  # (num_envs * 5, 4)

        # Expand hand pose for all fingers
        hand_pos_expanded = (
            hand_pos.unsqueeze(1).expand(-1, 5, -1).reshape(-1, 3)
        )  # (num_envs * 5, 3)
        hand_quat_expanded = (
            hand_quat.unsqueeze(1).expand(-1, 5, -1).reshape(-1, 4)
        )  # (num_envs * 5, 4)

        # Transform all positions and orientations at once
        finger_pos_hand_flat = point_in_hand_frame(
            finger_pos_world_flat, hand_pos_expanded, hand_quat_expanded
        )
        hand_quat_conj_expanded = quat_conjugate(hand_quat_expanded)
        finger_quat_hand_flat = quat_mul(
            hand_quat_conj_expanded, finger_quat_world_flat
        )

        # Reshape back to (num_envs, 5, 3) and (num_envs, 5, 4)
        finger_pos_hand = finger_pos_hand_flat.reshape(self.num_envs, 5, 3)
        finger_quat_hand = finger_quat_hand_flat.reshape(self.num_envs, 5, 4)

        # Combine into final poses tensor
        poses_hand = torch.cat(
            [finger_pos_hand, finger_quat_hand], dim=2
        )  # (num_envs, 5, 7)

        # Reshape back to (num_envs, 35)
        return poses_hand.view(self.num_envs, -1)

    def get_observation_space(self):
        """
        Get the observation space for the environment.

        Returns:
            gym.spaces.Box observation space
        """
        return gym.spaces.Box(
            low=-float("inf"), high=float("inf"), shape=(self.num_observations,)
        )

    def get_obs_index_for_base_joint(self, joint_name: str, obs_type: str = "pos"):
        """
        Get the index in the concatenated observation tensor for a base joint.

        Args:
            joint_name: Name of the base joint (e.g., "ARTx", "ARTy", etc.)
            obs_type: Type of observation ("pos", "vel", "target")

        Returns:
            Index in the concatenated observation tensor
        """
        if joint_name not in self.base_joint_to_index:
            raise ValueError(
                f"Unknown base joint name: {joint_name}. Available: {list(self.base_joint_to_index.keys())}"
            )

        joint_idx = self.base_joint_to_index[joint_name]

        # Determine which observation component to use
        if obs_type == "pos":
            component_key = "base_dof_pos"
        elif obs_type == "vel":
            component_key = "base_dof_vel"
        elif obs_type == "target":
            component_key = "base_dof_target"
        else:
            raise ValueError(
                f"Unknown obs_type: {obs_type}. Available: pos, vel, target"
            )

        if component_key not in self.component_slice_indices:
            raise ValueError(
                f"Observation component {component_key} not found in observation keys"
            )

        component_start, _ = self.component_slice_indices[component_key]
        return component_start + joint_idx

    def get_obs_index_for_finger_control(
        self, control_name: str, obs_type: str = "pos"
    ):
        """
        Get the index in the concatenated observation tensor for a finger control.

        Args:
            control_name: Name of the finger control (e.g., "th_dip", "if_pip", etc.)
            obs_type: Type of observation ("pos", "vel", "target")

        Returns:
            Index in the concatenated observation tensor
        """
        if control_name not in self.control_name_to_index:
            raise ValueError(
                f"Unknown control name: {control_name}. Available: {list(self.control_name_to_index.keys())}"
            )

        control_idx = self.control_name_to_index[control_name]

        # Determine which observation component to use
        if obs_type == "pos":
            component_key = "active_finger_dof_pos"
        elif obs_type == "vel":
            component_key = "active_finger_dof_vel"
        elif obs_type == "target":
            component_key = "active_finger_dof_target"
        else:
            raise ValueError(
                f"Unknown obs_type: {obs_type}. Available: pos, vel, target"
            )

        if component_key not in self.component_slice_indices:
            raise ValueError(
                f"Observation component {component_key} not found in observation keys"
            )

        component_start, _ = self.component_slice_indices[component_key]
        return component_start + control_idx

    def get_raw_finger_dof(
        self, dof_name: str, obs_type: str = "pos", obs_data=None, env_idx: int = None
    ):
        """
        Get finger DOF value by raw DOF name (including inactive DOFs like r_f_joint3_1).

        Args:
            dof_name: Name of the raw finger DOF (e.g., "r_f_joint1_1", "r_f_joint3_1", etc.)
            obs_type: Type of observation ("pos", "vel", "target")
            obs_data: Either observation tensor OR observation dictionary (not both)
            env_idx: Environment index (if None, returns data for all environments)

        Returns:
            DOF value(s) - scalar if env_idx specified, tensor if env_idx is None
        """
        if dof_name not in self.raw_dof_name_to_index:
            raise ValueError(
                f"Unknown DOF name: {dof_name}. Available: {list(self.raw_dof_name_to_index.keys())}"
            )

        if obs_data is None:
            raise ValueError("obs_data must be provided")

        # Determine which observation component to use
        if obs_type == "pos":
            component_key = "all_finger_dof_pos"
        elif obs_type == "vel":
            component_key = "all_finger_dof_vel"
        elif obs_type == "target":
            component_key = "all_finger_dof_target"
        else:
            raise ValueError(
                f"Unknown obs_type: {obs_type}. Available: pos, vel, target"
            )

        # Check if obs_data is a dictionary
        if isinstance(obs_data, dict):
            if component_key not in obs_data:
                raise ValueError(f"Component '{component_key}' not found in obs_dict")

            raw_dof_idx = self.raw_dof_name_to_index[dof_name]
            finger_dof_idx = raw_dof_idx - self.NUM_BASE_DOFS

            if finger_dof_idx < 0:
                raise ValueError(
                    f"DOF {dof_name} is not a finger DOF (index {raw_dof_idx} < {self.NUM_BASE_DOFS})"
                )

            data = obs_data[component_key][:, finger_dof_idx]
            return data[env_idx].item() if env_idx is not None else data

        # obs_data is a tensor - raw finger DOFs not available in concatenated tensor
        else:
            raise ValueError(
                f"Raw finger DOF '{dof_name}' access requires obs_dict. "
                f"obs_tensor only contains active finger DOFs, not all raw DOFs."
            )

    def get_obs_range_for_finger_pose(
        self, finger_body_name: str, frame: str = "world"
    ):
        """
        Get the index range in the concatenated observation tensor for a finger pose.

        Args:
            finger_body_name: Name of the finger body (e.g., "r_f_link1_tip", "r_f_link2_pad", etc.)
            frame: Reference frame ("world" or "hand")

        Returns:
            Tuple of (start_index, end_index) for the 7D pose in concatenated tensor
        """
        if finger_body_name not in self.finger_body_to_index:
            raise ValueError(
                f"Unknown finger body name: {finger_body_name}. Available: {list(self.finger_body_to_index.keys())}"
            )

        body_type, finger_idx = self.finger_body_to_index[finger_body_name]

        # Determine which observation component to use
        if body_type == "fingertip":
            component_key = f"fingertip_poses_{frame}"
        elif body_type == "fingerpad":
            component_key = f"fingerpad_poses_{frame}"
        else:
            raise ValueError(f"Unknown body type: {body_type}")

        if component_key not in self.component_slice_indices:
            raise ValueError(
                f"Observation component {component_key} not found in observation keys"
            )

        component_start, _ = self.component_slice_indices[component_key]
        pose_start = component_start + finger_idx * 7
        pose_end = pose_start + 7

        return (pose_start, pose_end)

    def get_obs_range_for_contact_force(self, finger_idx: int):
        """
        Get the index range in the concatenated observation tensor for contact force.

        Args:
            finger_idx: Finger index (0-4 for 5 fingers)

        Returns:
            Tuple of (start_index, end_index) for the 3D force in concatenated tensor
        """
        if finger_idx < 0 or finger_idx >= 5:
            raise ValueError(f"Finger index must be 0-4, got {finger_idx}")

        if "contact_forces" not in self.component_slice_indices:
            raise ValueError("Contact forces not found in observation keys")

        component_start, _ = self.component_slice_indices["contact_forces"]
        force_start = component_start + finger_idx * 3
        force_end = force_start + 3

        return (force_start, force_end)

    def get_base_dof_value(
        self, joint_name: str, obs_type: str = "pos", obs_data=None, env_idx: int = None
    ):
        """
        Get base DOF value by joint name.

        Args:
            joint_name: Name of the base joint (e.g., "ARTx", "ARTy", etc.)
            obs_type: Type of observation ("pos", "vel", "target")
            obs_data: Either observation tensor OR observation dictionary (not both)
            env_idx: Environment index (if None, returns data for all environments)

        Returns:
            DOF value(s) - scalar if env_idx specified, tensor if env_idx is None
        """
        if joint_name not in self.base_joint_to_index:
            raise ValueError(
                f"Unknown base joint name: {joint_name}. Available: {list(self.base_joint_to_index.keys())}"
            )

        if obs_data is None:
            raise ValueError("obs_data must be provided")

        joint_idx = self.base_joint_to_index[joint_name]

        # Determine which observation component to use
        if obs_type == "pos":
            component_key = "base_dof_pos"
        elif obs_type == "vel":
            component_key = "base_dof_vel"
        elif obs_type == "target":
            component_key = "base_dof_target"
        else:
            raise ValueError(
                f"Unknown obs_type: {obs_type}. Available: pos, vel, target"
            )

        # Check if obs_data is a dictionary
        if isinstance(obs_data, dict):
            if component_key not in obs_data:
                raise ValueError(f"Component '{component_key}' not found in obs_dict")

            data = obs_data[component_key][:, joint_idx]
            return data[env_idx].item() if env_idx is not None else data

        # obs_data is a tensor
        else:
            if component_key not in self.component_slice_indices:
                raise ValueError(
                    f"Component '{component_key}' not found in observation tensor"
                )

            component_start, _ = self.component_slice_indices[component_key]
            tensor_idx = component_start + joint_idx

            if env_idx is not None:
                return obs_data[env_idx, tensor_idx].item()
            else:
                return obs_data[:, tensor_idx]

    def get_active_finger_dof_value(
        self,
        control_name: str,
        obs_type: str = "pos",
        obs_data=None,
        env_idx: int = None,
    ):
        """
        Get active finger DOF value by control name.

        Args:
            control_name: Name of the finger control (e.g., "th_dip", "if_pip", etc.)
            obs_type: Type of observation ("pos", "vel", "target")
            obs_data: Either observation tensor OR observation dictionary (not both)
            env_idx: Environment index (if None, returns data for all environments)

        Returns:
            DOF value(s) - scalar if env_idx specified, tensor if env_idx is None
        """
        if control_name not in self.control_name_to_index:
            raise ValueError(
                f"Unknown control name: {control_name}. Available: {list(self.control_name_to_index.keys())}"
            )

        if obs_data is None:
            raise ValueError("obs_data must be provided")

        control_idx = self.control_name_to_index[control_name]

        # Determine which observation component to use
        if obs_type == "pos":
            component_key = "active_finger_dof_pos"
        elif obs_type == "vel":
            component_key = "active_finger_dof_vel"
        elif obs_type == "target":
            component_key = "active_finger_dof_target"
        else:
            raise ValueError(
                f"Unknown obs_type: {obs_type}. Available: pos, vel, target"
            )

        # Check if obs_data is a dictionary
        if isinstance(obs_data, dict):
            if component_key not in obs_data:
                raise ValueError(f"Component '{component_key}' not found in obs_dict")

            data = obs_data[component_key][:, control_idx]
            return data[env_idx].item() if env_idx is not None else data

        # obs_data is a tensor
        else:
            if component_key not in self.component_slice_indices:
                raise ValueError(
                    f"Component '{component_key}' not found in observation tensor"
                )

            component_start, _ = self.component_slice_indices[component_key]
            tensor_idx = component_start + control_idx

            if env_idx is not None:
                return obs_data[env_idx, tensor_idx].item()
            else:
                return obs_data[:, tensor_idx]

    def get_finger_pose_value(
        self,
        finger_body_name: str,
        frame: str = "world",
        obs_data=None,
        env_idx: int = None,
    ):
        """
        Get finger pose by body name.

        Args:
            finger_body_name: Name of the finger body (e.g., "r_f_link1_tip", "r_f_link2_pad", etc.)
            frame: Reference frame ("world" or "hand")
            obs_data: Either observation tensor OR observation dictionary (not both)
            env_idx: Environment index (if None, returns data for all environments)

        Returns:
            Pose data - dict with 'position' and 'orientation' (numpy arrays if env_idx specified, tensors if None)
        """
        if finger_body_name not in self.finger_body_to_index:
            raise ValueError(
                f"Unknown finger body name: {finger_body_name}. Available: {list(self.finger_body_to_index.keys())}"
            )

        if obs_data is None:
            raise ValueError("obs_data must be provided")

        body_type, finger_idx = self.finger_body_to_index[finger_body_name]

        # Determine which observation component to use
        if body_type == "fingertip":
            component_key = f"fingertip_poses_{frame}"
        elif body_type == "fingerpad":
            component_key = f"fingerpad_poses_{frame}"
        else:
            raise ValueError(f"Unknown body type: {body_type}")

        # Check if obs_data is a dictionary
        if isinstance(obs_data, dict):
            if component_key not in obs_data:
                raise ValueError(f"Component '{component_key}' not found in obs_dict")

            pose_start = finger_idx * 7
            pose_end = pose_start + 7
            pose_data = obs_data[component_key][:, pose_start:pose_end]

            if env_idx is not None:
                return {
                    "position": pose_data[env_idx, :3].cpu().numpy(),
                    "orientation": pose_data[env_idx, 3:7].cpu().numpy(),
                }
            else:
                return {"position": pose_data[:, :3], "orientation": pose_data[:, 3:7]}

        # obs_data is a tensor
        else:
            if component_key not in self.component_slice_indices:
                raise ValueError(
                    f"Component '{component_key}' not found in observation tensor"
                )

            component_start, _ = self.component_slice_indices[component_key]
            pose_start = component_start + finger_idx * 7
            pose_end = pose_start + 7

            if env_idx is not None:
                pose_data = obs_data[env_idx, pose_start:pose_end]
                return {
                    "position": pose_data[:3].cpu().numpy(),
                    "orientation": pose_data[3:7].cpu().numpy(),
                }
            else:
                pose_data = obs_data[:, pose_start:pose_end]
                return {"position": pose_data[:, :3], "orientation": pose_data[:, 3:7]}

    def get_contact_force_value(
        self, body_name: str, obs_data=None, env_idx: int = None
    ):
        """
        Get contact force by body name.

        Args:
            body_name: Name of the body to get contact force for (e.g., "r_f_link3_4")
            obs_data: Either observation tensor OR observation dictionary (not both)
            env_idx: Environment index (if None, returns data for all environments)

        Returns:
            Contact force - numpy array if env_idx specified, tensor if env_idx is None
        """
        if body_name not in self.contact_force_body_name_to_index:
            available_bodies = list(self.contact_force_body_name_to_index.keys())
            raise ValueError(
                f"Unknown contact force body: {body_name}. Available: {available_bodies}"
            )

        body_idx = self.contact_force_body_name_to_index[body_name]

        if obs_data is None:
            raise ValueError("obs_data must be provided")

        component_key = "contact_forces"

        # Check if obs_data is a dictionary
        if isinstance(obs_data, dict):
            if component_key not in obs_data:
                raise ValueError(f"Component '{component_key}' not found in obs_dict")

            force_start = body_idx * 3
            force_end = force_start + 3
            force_data = obs_data[component_key][:, force_start:force_end]

            if env_idx is not None:
                return force_data[env_idx].cpu().numpy()
            else:
                return force_data

        # obs_data is a tensor
        else:
            if component_key not in self.component_slice_indices:
                raise ValueError(
                    f"Component '{component_key}' not found in observation tensor"
                )

            component_start, _ = self.component_slice_indices[component_key]
            force_start = component_start + body_idx * 3
            force_end = force_start + 3

            if env_idx is not None:
                return obs_data[env_idx, force_start:force_end].cpu().numpy()
            else:
                return obs_data[:, force_start:force_end]

    def parse_obs_tensor(
        self,
        obs_tensor: torch.Tensor = None,
        obs_dict: Dict[str, torch.Tensor] = None,
        env_idx: int = 0,
    ):
        """
        Parse observation tensor and provide convenient access to components.

        Args:
            obs_tensor: Observation tensor to parse (if None, uses current observation)
            obs_dict: Observation dictionary for accessing all_finger_dof components (if None, uses limited access)
            env_idx: Environment index

        Returns:
            ObservationParser instance with convenient accessor methods
        """
        if obs_tensor is None:
            obs_tensor = self.obs_buf

        if obs_tensor is None:
            raise ValueError("No observation tensor available")

        # Extract the observation for the specified environment
        if len(obs_tensor.shape) == 1:
            obs_data = obs_tensor  # Already single environment
        else:
            obs_data = obs_tensor[env_idx]

        return ObservationParser(obs_data, self, obs_dict)

    def _compute_arr_aligned_pose(self, hand_poses: torch.Tensor) -> torch.Tensor:
        """
        Compute hand pose with orientation aligned to ARR DOFs.

        Compensates for the built-in 90° Y-axis rotation in the floating hand model.
        When ARRx=ARRy=ARRz=0, this returns orientation as identity quaternion [0,0,0,1].

        Args:
            hand_poses: Hand poses [num_envs, 7] with position (3) + quaternion (4)

        Returns:
            ARR-aligned poses [num_envs, 7] with compensated orientation
        """
        # The built-in rotation is 90° around Y axis: [0, sqrt(0.5), 0, sqrt(0.5)]
        # To compensate, we need the inverse rotation (conjugate for unit quaternions)
        sqrt_half = 0.7071067811865476  # sqrt(0.5)

        # Create the built-in rotation quaternion for all environments
        # Note: quat_conjugate of [0, sqrt(0.5), 0, sqrt(0.5)] is [0, -sqrt(0.5), 0, sqrt(0.5)]
        builtin_rotation = torch.tensor(
            [0.0, sqrt_half, 0.0, sqrt_half],
            device=hand_poses.device,
            dtype=hand_poses.dtype,
        )
        inv_rotation = quat_conjugate(builtin_rotation)

        # Extract positions and quaternions
        positions = hand_poses[:, :3]
        quaternions = hand_poses[:, 3:7]

        # Apply quaternion multiplication: q_aligned = q_hand * q_inv
        # Expand inverse rotation to match batch size
        inv_rotation_batch = inv_rotation.unsqueeze(0).expand(quaternions.shape[0], -1)
        q_aligned = quat_mul(quaternions, inv_rotation_batch)

        # Combine position and aligned orientation
        arr_aligned_poses = torch.cat([positions, q_aligned], dim=1)

        return arr_aligned_poses


class ObservationParser:
    """
    Helper class to parse and access observation tensor components.
    """

    def __init__(
        self,
        obs_data: torch.Tensor,
        encoder: "ObservationEncoder",
        obs_dict: Dict[str, torch.Tensor] = None,
    ):
        self.obs_data = obs_data
        self.encoder = encoder
        self.obs_dict = obs_dict

    def get_base_dof(self, joint_name: str, obs_type: str = "pos"):
        """Get base DOF value by joint name."""
        idx = self.encoder.get_obs_index_for_base_joint(joint_name, obs_type)
        return self.obs_data[idx].item()

    def get_finger_dof(self, control_name: str, obs_type: str = "pos"):
        """Get finger DOF value by control name."""
        idx = self.encoder.get_obs_index_for_finger_control(control_name, obs_type)
        return self.obs_data[idx].item()

    def get_finger_pose(self, finger_body_name: str, frame: str = "world"):
        """Get finger pose by body name."""
        start_idx, end_idx = self.encoder.get_obs_range_for_finger_pose(
            finger_body_name, frame
        )
        pose_data = self.obs_data[start_idx:end_idx]
        return {
            "position": pose_data[:3].cpu().numpy(),
            "orientation": pose_data[3:7].cpu().numpy(),
        }

    def get_contact_force(self, finger_idx: int):
        """Get contact force by finger index."""
        start_idx, end_idx = self.encoder.get_obs_range_for_contact_force(finger_idx)
        return self.obs_data[start_idx:end_idx].cpu().numpy()

    def get_raw_finger_dof(self, dof_name: str, obs_type: str = "pos"):
        """
        Get any finger DOF value by raw DOF name (including inactive DOFs).
        Requires obs_dict to be provided during parsing.

        Args:
            dof_name: Name of the raw finger DOF (e.g., "r_f_joint1_1", "r_f_joint3_1", etc.)
            obs_type: Type of observation ("pos", "vel", "target")

        Returns:
            DOF value for the specified finger DOF (single environment)
        """
        if self.obs_dict is None:
            raise ValueError(
                "obs_dict must be provided during parsing to access raw finger DOF values"
            )

        return self.encoder.get_raw_finger_dof(
            dof_name, obs_type, self.obs_dict, env_idx=0
        )
