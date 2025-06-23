"""
Tensor manager component for DexHand environment.

This module provides tensor management functionality for the DexHand environment,
including tensor acquisition, setup, and synchronization.
"""

# Import standard libraries
import torch
import numpy as np
from loguru import logger

# Import IsaacGym
from isaacgym import gymtorch


class TensorManager:
    """
    Manages simulation tensors for the DexHand environment.

    This component provides functionality to:
    - Acquire tensor handles from simulation
    - Set up and initialize tensors
    - Handle tensor synchronization between CPU and GPU
    - Manage tensor memory and buffers
    """

    def __init__(self, parent):
        """
        Initialize the tensor manager.

        Args:
            parent: Parent DexHandBase instance
        """
        self.parent = parent
        self.gym = parent.gym
        self.sim = parent.sim

        # Flag to track initialization state
        self.tensors_initialized = False

        # Tensor handles
        self._dof_state_tensor_handle = None
        self.rigid_body_state_tensor_handle = None
        self._contact_force_tensor_handle = None

        # Simulation tensors
        self.dof_state = None
        self.dof_pos = None
        self.dof_vel = None
        self.actor_root_state_tensor = None
        self.num_dof = None
        self.dof_props = None
        self.rigid_body_states = None
        self.contact_forces = None

        # DOF properties from asset
        self.dof_props_from_asset = None

    @property
    def num_envs(self):
        """Access num_envs from parent (single source of truth)."""
        return self.parent.num_envs

    @property
    def device(self):
        """Access device from parent (single source of truth)."""
        # parent.device should already be a torch.device - fail fast if not
        return self.parent.device

    @property
    def rigid_body_index_to_name(self):
        """Access rigid body name mapping from hand_initializer (single source of truth)."""
        if self.parent.hand_initializer is None:
            raise RuntimeError("hand_initializer is None - initialization failed")
        return self.parent.hand_initializer.rigid_body_index_to_name

    def acquire_tensor_handles(self):
        """
        Acquire handles to simulation tensors.

        Returns:
            Dictionary of tensor handles

        Raises:
            RuntimeError: If any required tensor handle acquisition fails
        """
        # Reset flags
        self.tensors_initialized = False

        logger.info("Acquiring tensor handles from simulation...")

        # NOTE: We don't need to step simulation here when using GPU pipeline
        # The simulation has already been prepared by gym.prepare_sim()

        # Get handle for DOF states
        self.dof_state_tensor_handle = self.gym.acquire_dof_state_tensor(self.sim)
        if self.dof_state_tensor_handle is None:
            raise RuntimeError(
                "Failed to acquire DOF state tensor handle. Cannot continue."
            )

        # Get handle for rigid body states
        self.rigid_body_state_tensor_handle = self.gym.acquire_rigid_body_state_tensor(
            self.sim
        )
        if self.rigid_body_state_tensor_handle is None:
            raise RuntimeError(
                "Failed to acquire rigid body state tensor handle. Cannot continue."
            )

        # Get handle for contact forces
        self._contact_force_tensor_handle = self.gym.acquire_net_contact_force_tensor(
            self.sim
        )
        if self._contact_force_tensor_handle is None:
            raise RuntimeError(
                "Failed to acquire contact force tensor handle. Cannot continue."
            )

        # Get handle for actor root state
        self.actor_root_state_tensor_handle = self.gym.acquire_actor_root_state_tensor(
            self.sim
        )
        if self.actor_root_state_tensor_handle is None:
            raise RuntimeError(
                "Failed to acquire actor root state tensor handle. Cannot continue."
            )

        return {
            "dof_state": self._dof_state_tensor_handle,
            "rigid_body_state": self.rigid_body_state_tensor_handle,
            "contact_force": self._contact_force_tensor_handle,
            "actor_root_state": self.actor_root_state_tensor_handle,
        }

    def setup_tensors(self, contact_force_body_indices=None):
        """
        Set up tensors from handles.

        Args:
            contact_force_body_indices: Indices of bodies to monitor for contact forces

        Returns:
            Dictionary of tensors

        Raises:
            RuntimeError: If any tensor setup fails
        """
        logger.info("Setting up tensors from handles...")

        # Verify tensor handles exist
        if self.dof_state_tensor_handle is None:
            raise RuntimeError(
                "DOF state tensor handle is None. Cannot set up tensors."
            )

        if self.rigid_body_state_tensor_handle is None:
            raise RuntimeError(
                "Rigid body state tensor handle is None. Cannot set up tensors."
            )

        if self._contact_force_tensor_handle is None:
            raise RuntimeError(
                "Contact force tensor handle is None. Cannot set up tensors."
            )

        # Wrap DOF state tensor
        # Wrap DOF state tensor
        try:
            self.dof_state = gymtorch.wrap_tensor(self.dof_state_tensor_handle)

            if self.dof_state is None:
                raise RuntimeError("DOF state tensor is None after wrapping")

            # Verify device matches expectation
            actual_device = self.dof_state.device
            if torch.device(actual_device) != torch.device(self.device):
                raise RuntimeError(
                    f"Device mismatch: TensorManager initialized with device '{self.device}' "
                    f"but Isaac Gym created tensors on device '{actual_device}'. "
                    f"This indicates a configuration error that must be fixed."
                )

            # Check if the tensor has elements before accessing shape
            if hasattr(self.dof_state, "numel") and self.dof_state.numel() == 0:
                raise RuntimeError(
                    "DOF state tensor is empty (zero elements) after wrapping"
                )

        except Exception as e:
            logger.error(f"Error while wrapping DOF state tensor: {e}")
            logger.error(f"DOF state tensor handle: {self._dof_state_tensor_handle}")
            raise RuntimeError(f"Failed to wrap DOF state tensor: {e}")

        logger.debug(f"DOF state tensor shape: {self.dof_state.shape}")

        # Get number of DOFs per environment
        # DOF state is shaped (num_envs * num_dofs_per_env, 2)
        # where 2 is (position, velocity)
        # We need to reshape to (num_envs, num_dofs_per_env, 2)
        if self.dof_state.shape[0] % self.num_envs != 0:
            raise RuntimeError(
                f"DOF state tensor shape {self.dof_state.shape[0]} is not divisible by num_envs {self.num_envs}"
            )

        dofs_per_env = self.dof_state.shape[0] // self.num_envs
        self.num_dof = dofs_per_env

        logger.info(
            f"Total DOFs: {self.dof_state.shape[0]}, Environments: {self.num_envs}, DOFs per env: {self.num_dof}"
        )

        # Reshape DOF state to (num_envs, num_dofs_per_env, 2)
        # Update self.dof_state to be the reshaped view for consistency
        self.dof_state = self.dof_state.view(self.num_envs, self.num_dof, 2)

        # Extract position and velocity components
        self.dof_pos = self.dof_state[..., 0]
        self.dof_vel = self.dof_state[..., 1]

        logger.debug(f"DOF position tensor shape: {self.dof_pos.shape}")
        logger.debug(f"DOF velocity tensor shape: {self.dof_vel.shape}")

        # Get DOF properties
        logger.debug("Getting DOF properties...")

        # Check if we have DOF properties from the asset
        if self.dof_props_from_asset is not None:
            logger.debug("Using DOF properties from asset")
            self.dof_props = self.dof_props_from_asset
        else:
            # Following the pattern from reference implementations, create a default DOF properties tensor
            # DOF properties have 6 values: stiffness, damping, friction, armature, min, max
            # We'll create this directly without trying to use acquire_dof_attribute_tensor
            # which is not available in all Isaac Gym versions

            logger.debug("Creating default DOF properties tensor")
            # Create properties tensor based on the dof_state shape
            self.dof_props = torch.zeros((self.num_dof, 6), device=self.device)
            # Set reasonable defaults
            self.dof_props[:, 0] = 100.0  # stiffness
            self.dof_props[:, 1] = 5.0  # damping
            self.dof_props[:, 4] = -1.0  # min
            self.dof_props[:, 5] = 1.0  # max

        logger.debug(f"DOF properties tensor shape: {self.dof_props.shape}")

        # Wrap rigid body state tensor
        # Wrap rigid body state tensor
        rigid_body_states_flat = gymtorch.wrap_tensor(
            self.rigid_body_state_tensor_handle
        )

        if rigid_body_states_flat is None or rigid_body_states_flat.numel() == 0:
            raise RuntimeError(
                "Rigid body state tensor is empty or None after wrapping"
            )

        logger.debug(
            f"Rigid body states flat tensor shape: {rigid_body_states_flat.shape}"
        )

        # Reshape rigid body states to (num_envs, num_bodies_per_env, 13)
        if rigid_body_states_flat.shape[0] % self.num_envs != 0:
            raise RuntimeError(
                f"Rigid body tensor shape {rigid_body_states_flat.shape[0]} is not divisible by num_envs {self.num_envs}"
            )

        num_bodies_per_env = rigid_body_states_flat.shape[0] // self.num_envs
        self.rigid_body_states = rigid_body_states_flat.view(
            self.num_envs, num_bodies_per_env, 13
        )
        logger.debug(
            f"Rigid body states tensor reshaped to: {self.rigid_body_states.shape}"
        )

        # Wrap actor root state tensor (correct Isaac Gym API)
        # Wrap actor root state tensor
        actor_root_state_flat = gymtorch.wrap_tensor(
            self.actor_root_state_tensor_handle
        )

        if actor_root_state_flat is None or actor_root_state_flat.numel() == 0:
            raise RuntimeError(
                "Actor root state tensor is empty or None after wrapping"
            )

        logger.debug(
            f"Actor root state flat tensor shape: {actor_root_state_flat.shape}"
        )

        # Reshape to (num_envs, num_actors_per_env, 13) for consistency
        # The tensor contains root states for all actors across all environments
        if actor_root_state_flat.shape[0] % self.num_envs != 0:
            raise RuntimeError(
                f"Actor root state tensor shape {actor_root_state_flat.shape[0]} is not divisible by num_envs {self.num_envs}"
            )

        num_actors_per_env = actor_root_state_flat.shape[0] // self.num_envs
        self.actor_root_state_tensor = actor_root_state_flat.view(
            self.num_envs, num_actors_per_env, 13
        )
        logger.debug(
            f"Actor root state tensor reshaped to: {self.actor_root_state_tensor.shape}"
        )

        # Wrap contact force tensor
        # Wrap contact force tensor
        contact_forces_flat = gymtorch.wrap_tensor(self._contact_force_tensor_handle)

        if contact_forces_flat is None or contact_forces_flat.numel() == 0:
            raise RuntimeError("Contact force tensor is empty or None after wrapping")

        logger.debug(f"Contact forces flat tensor shape: {contact_forces_flat.shape}")

        # Reshape contact forces for contact force bodies
        if contact_force_body_indices is None or len(contact_force_body_indices) == 0:
            raise RuntimeError(
                "No contact force body indices provided. Cannot set up contact forces."
            )

        # Number of contact force bodies
        num_contact_bodies = len(contact_force_body_indices[0])
        logger.debug(f"Number of contact force bodies: {num_contact_bodies}")

        # Create tensor for contact forces
        self.contact_forces = torch.zeros(
            (self.num_envs, num_contact_bodies, 3), device=self.device
        )
        logger.debug(f"Contact forces tensor shape: {self.contact_forces.shape}")

        # Copy contact forces for each contact body
        for i in range(self.num_envs):
            for j in range(num_contact_bodies):
                if i < len(contact_force_body_indices) and j < len(
                    contact_force_body_indices[i]
                ):
                    # Get the rigid body index for this contact body
                    body_idx = contact_force_body_indices[i][j]

                    if body_idx < contact_forces_flat.shape[0]:
                        # Copy contact force (x, y, z)
                        self.contact_forces[i, j, :] = contact_forces_flat[body_idx]

        # Mark tensors as initialized
        self.tensors_initialized = True
        logger.info("Tensors initialized successfully")

        return {
            "dof_state": self.dof_state,
            "dof_pos": self.dof_pos,
            "dof_vel": self.dof_vel,
            "actor_root_state_tensor": self.actor_root_state_tensor,
            "num_dof": self.num_dof,
            "dof_props": self.dof_props,
            "rigid_body_states": self.rigid_body_states,
            "contact_forces": self.contact_forces,
        }

    def refresh_tensors(self, contact_force_body_indices=None):
        """
        Refresh tensor data from simulation.

        Args:
            contact_force_body_indices: Indices of bodies to monitor for contact forces

        Returns:
            Dictionary of refreshed tensors

        Raises:
            RuntimeError: If tensors are not initialized or refresh fails
        """
        try:
            # Verify tensors are initialized
            if not self.tensors_initialized:
                raise RuntimeError("Tensors not initialized. Cannot refresh.")

            # Refresh DOF state - verify shape first
            if self.dof_state is None:
                raise RuntimeError("DOF state tensor is None. Cannot refresh.")

            expected_size = self.num_envs * self.num_dof * 2
            if self.dof_state.numel() != expected_size:
                raise RuntimeError(
                    f"DOF state tensor size mismatch. Expected {expected_size} elements, got {self.dof_state.numel()}"
                )

            # Reshape to [num_envs, num_dof, 2] and extract position and velocity
            self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
            self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

            # Refresh contact forces
            if (
                contact_force_body_indices is None
                or len(contact_force_body_indices) == 0
            ):
                raise RuntimeError(
                    "No contact force body indices provided. Cannot refresh contact forces."
                )

            contact_forces_flat = gymtorch.wrap_tensor(
                self._contact_force_tensor_handle
            )
            if contact_forces_flat is None or contact_forces_flat.numel() == 0:
                raise RuntimeError(
                    "Contact force tensor is empty or None during refresh"
                )

            # Debug contact forces detection
            if logger._core.min_level <= 10:  # Only if DEBUG level is enabled
                nonzero_forces = torch.nonzero(contact_forces_flat.abs() > 0.001)
                if nonzero_forces.numel() > 0:
                    # Get unique rigid body indices with contact
                    unique_contact_indices = torch.unique(nonzero_forces[:, 0])

                    # Log each body with contact force
                    for idx in unique_contact_indices:
                        idx_int = idx.item()
                        body_name = self.rigid_body_index_to_name.get(
                            idx_int, "Unknown body"
                        )
                        force_vec = contact_forces_flat[idx_int]
                        force_mag = torch.norm(force_vec).item()
                        logger.debug(
                            f"Contact on '{body_name}' (idx={idx_int}): magnitude={force_mag:.2f}N"
                        )

            # Number of contact force bodies
            num_contact_bodies = len(contact_force_body_indices[0])

            # Copy contact forces for each contact body
            for i in range(self.num_envs):
                if i >= len(contact_force_body_indices):
                    raise RuntimeError(
                        f"Contact force body indices missing for environment {i}"
                    )

                for j in range(num_contact_bodies):
                    if j >= len(contact_force_body_indices[i]):
                        raise RuntimeError(
                            f"Contact force body index {j} missing for environment {i}"
                        )

                    # Get the rigid body index for this contact body
                    body_idx = contact_force_body_indices[i][j]

                    if body_idx >= contact_forces_flat.shape[0]:
                        raise RuntimeError(
                            f"Contact body index {body_idx} exceeds contact forces size {contact_forces_flat.shape[0]}"
                        )

                    # Copy contact force (x, y, z)
                    self.contact_forces[i, j, :] = contact_forces_flat[body_idx]

            return {
                "dof_pos": self.dof_pos,
                "dof_vel": self.dof_vel,
                "actor_root_state_tensor": self.actor_root_state_tensor,
                "contact_forces": self.contact_forces,
            }
        except Exception as e:
            logger.critical(f"CRITICAL ERROR in refresh_tensors: {e}")
            import traceback

            traceback.print_exc()
            raise

    def tensor_clamp(self, tensor, min_val, max_val):
        """
        Clamp tensor values between min and max.

        Args:
            tensor: Input tensor
            min_val: Minimum value(s)
            max_val: Maximum value(s)

        Returns:
            Clamped tensor
        """
        return torch.max(torch.min(tensor, max_val), min_val)

    def to_device(self, tensor, device=None):
        """
        Move tensor to specified device.

        Args:
            tensor: Input tensor
            device: Target device (if None, use self.device)

        Returns:
            Tensor on target device
        """
        if device is None:
            device = self.device

        if tensor.device != device:
            return tensor.to(device)
        return tensor

    def set_dof_properties(self, dof_props):
        """
        Set DOF properties from an external source (like hand_initializer).

        Args:
            dof_props: DOF properties dictionary or numpy array

        Returns:
            None
        """
        try:
            if isinstance(dof_props, dict):
                # Convert dictionary to tensor format
                # Assuming keys: stiffness, damping, friction, armature, lower, upper
                props_tensor = torch.zeros((self.num_dof, 6), device=self.device)

                # Map dictionary keys to tensor columns
                if "stiffness" in dof_props:
                    props_tensor[:, 0] = torch.tensor(
                        dof_props["stiffness"], device=self.device
                    )
                if "damping" in dof_props:
                    props_tensor[:, 1] = torch.tensor(
                        dof_props["damping"], device=self.device
                    )
                if "friction" in dof_props:
                    props_tensor[:, 2] = torch.tensor(
                        dof_props["friction"], device=self.device
                    )
                if "armature" in dof_props:
                    props_tensor[:, 3] = torch.tensor(
                        dof_props["armature"], device=self.device
                    )
                if "lower" in dof_props:
                    props_tensor[:, 4] = torch.tensor(
                        dof_props["lower"], device=self.device
                    )
                if "upper" in dof_props:
                    props_tensor[:, 5] = torch.tensor(
                        dof_props["upper"], device=self.device
                    )

                self.dof_props_from_asset = props_tensor
            elif isinstance(dof_props, np.ndarray):
                # Handle structured arrays from Isaac Gym's get_asset_dof_properties or get_actor_dof_properties
                if dof_props.dtype.names is not None:
                    # Structured array - extract relevant fields
                    logger.debug(
                        f"Processing structured array with fields: {dof_props.dtype.names}"
                    )
                    num_dof = len(dof_props)
                    props_tensor = torch.zeros((num_dof, 6), device=self.device)

                    # Map field names to tensor columns
                    field_mapping = {
                        "stiffness": 0,
                        "damping": 1,
                        "friction": 2,
                        "armature": 3,
                        "lower": 4,
                        "upper": 5,
                    }

                    # Extract fields that exist in the array
                    logger.debug("Extracting DOF properties fields:")
                    for field_name, col_idx in field_mapping.items():
                        if field_name in dof_props.dtype.names:
                            try:
                                field_data = dof_props[field_name]
                                # Convert to tensor safely
                                field_tensor = torch.tensor(
                                    field_data.astype(np.float32), device=self.device
                                )
                                props_tensor[:, col_idx] = field_tensor
                                # Log summary for important fields
                                if field_name in ["lower", "upper"]:
                                    logger.debug(
                                        f"  {field_name}: range [{field_tensor.min():.3f}, {field_tensor.max():.3f}]"
                                    )
                            except Exception as e:
                                logger.error(
                                    f"Error converting field {field_name}: {e}"
                                )
                        else:
                            logger.warning(
                                f"  {field_name}: MISSING from structured array"
                            )

                    # Also check hasLimits field to debug why limits are 0
                    if "hasLimits" in dof_props.dtype.names:
                        has_limits = dof_props["hasLimits"]
                        logger.debug(
                            f"  DOFs with limits enabled: {np.sum(has_limits)}/{len(has_limits)}"
                        )
                        logger.debug(
                            f"  Joints without limits: {np.where(~has_limits)[0]}"
                        )

                    self.dof_props_from_asset = props_tensor
                else:
                    # Regular numpy array - convert directly
                    self.dof_props_from_asset = torch.tensor(
                        dof_props, device=self.device
                    )
            elif isinstance(dof_props, torch.Tensor):
                # Already a tensor, just move to device if needed
                self.dof_props_from_asset = self.to_device(dof_props)
            else:
                logger.error(f"Unsupported DOF properties type: {type(dof_props)}")
                return

            logger.info(
                f"DOF properties set from external source with shape: {self.dof_props_from_asset.shape}"
            )
        except Exception as e:
            logger.error(f"Error setting DOF properties: {e}")
            import traceback

            traceback.print_exc()
