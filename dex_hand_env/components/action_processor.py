"""
Action processor component for DexHand environment.

This module provides action processing functionality for the DexHand environment,
including action scaling, mapping, and PD control.
"""

# Import standard libraries
import torch
from loguru import logger
from functools import wraps

# Import IsaacGym
from isaacgym import gymtorch

# Import constants
from dex_hand_env.constants import (
    NUM_BASE_DOFS,
    NUM_ACTIVE_FINGER_DOFS,
    FINGER_COUPLING_MAP,
)


def initialization_only(func):
    """Decorator to ensure method is only called during initialization."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if hasattr(self, "_initialized") and self._initialized:
            raise RuntimeError(
                f"{func.__name__} can only be called during initialization"
            )
        return func(self, *args, **kwargs)

    return wrapper


def post_initialization_only(func):
    """Decorator to ensure method is only called after initialization is complete."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "_initialized") or not self._initialized:
            raise RuntimeError(
                f"{func.__name__} can only be called after finalize_setup()"
            )
        return func(self, *args, **kwargs)

    return wrapper


class ActionProcessor:
    """
    Processes actions for the DexHand environment.

    This component provides functionality to:
    - Map policy actions to robot DOFs
    - Apply action scaling and transformations
    - Handle different control modes (position, position_delta)
    - Manage PD control targets
    """

    def __init__(self, parent):
        """
        Initialize the action processor.

        Args:
            parent: Parent DexHandBase instance
        """
        self.parent = parent
        self.gym = parent.gym
        self.sim = parent.sim

        # Control settings
        self.action_control_mode = "position"  # position or position_delta
        self.policy_controls_hand_base = True
        self.policy_controls_fingers = True

        # Constants
        self.NUM_BASE_DOFS = NUM_BASE_DOFS
        self.NUM_ACTIVE_FINGER_DOFS = NUM_ACTIVE_FINGER_DOFS

        # Finger coupling map
        self.finger_coupling_map = FINGER_COUPLING_MAP

        # DOF properties and limits (will be accessed from tensor_manager)
        self.dof_lower_limits = None
        self.dof_upper_limits = None
        self.num_dof = 0

        # Velocity limits (must be set before setup)
        self.policy_finger_velocity_limit = None
        self.policy_base_lin_velocity_limit = None
        self.policy_base_ang_velocity_limit = None

        # Tensors
        self.prev_active_targets = None  # Previous active targets (6D + 12D)
        self.current_targets = None  # Current full DOF targets (26 DOFs)
        self.actions = None

        # Precomputed limits for active DOFs
        self.active_lower_limits = None
        self.active_upper_limits = None
        self.max_deltas = None  # Precomputed velocity-based deltas

        # Precomputed coupling mappings
        self.coupling_indices = (
            None  # Shape: (num_couplings, 2) with [finger_idx, dof_idx]
        )
        self.coupling_scales = None  # Shape: (num_couplings,) with scale factors
        self.middle_finger_spread_idx = None  # Index of r_f_joint3_1

        # Precomputed mask for action to active target mapping
        self.active_target_mask = (
            None  # Boolean mask: which active targets are controlled by policy
        )

        # Function pointers for control mode
        self._compute_targets_fn = None

        # Flag to ensure setup is complete
        self._initialized = False

    @property
    def num_envs(self):
        """Access num_envs from parent (single source of truth)."""
        return self.parent.num_envs

    @property
    def device(self):
        """Access device from parent (single source of truth)."""
        return self.parent.device

    @property
    def physics_manager(self):
        """Access physics_manager from parent (single source of truth)."""
        return self.parent.physics_manager

    @property
    def dof_names(self):
        """Access DOF names from hand_initializer (single source of truth)."""
        return self.parent.hand_initializer.dof_names

    @property
    def dof_props(self):
        """Access DOF properties from tensor_manager (single source of truth)."""
        return self.parent.tensor_manager.dof_props

    @property
    def control_dt(self):
        """Access control_dt from physics manager (single source of truth)."""
        if self.physics_manager is None:
            raise RuntimeError("physics_manager not set. Cannot access control_dt.")
        return self.physics_manager.control_dt

    def initialize_from_config(self, config):
        """
        Initialize the action processor with a configuration dictionary.

        Args:
            config: Dictionary containing:
                - control_mode: "position" or "position_delta"
                - num_dof: Number of DOFs
                - policy_controls_hand_base: bool (optional, default: True)
                - policy_controls_fingers: bool (optional, default: True)
                - finger_vel_limit: float (required)
                - base_lin_vel_limit: float (required)
                - base_ang_vel_limit: float (required)
        """
        # Set control mode
        self.action_control_mode = config["control_mode"]
        if self.action_control_mode not in ["position", "position_delta"]:
            raise ValueError(f"Invalid control mode: {self.action_control_mode}")

        # Set control options
        self.policy_controls_hand_base = config.get("policy_controls_hand_base", True)
        self.policy_controls_fingers = config.get("policy_controls_fingers", True)

        # Set velocity limits
        self.policy_finger_velocity_limit = config["finger_vel_limit"]
        self.policy_base_lin_velocity_limit = config["base_lin_vel_limit"]
        self.policy_base_ang_velocity_limit = config["base_ang_vel_limit"]

        # Set up DOF properties
        self.num_dof = config["num_dof"]

        # Extract DOF limits from tensor_manager's dof_props
        if isinstance(self.dof_props, torch.Tensor):
            # Format is [stiffness, damping, friction, armature, min, max]
            self.dof_lower_limits = self.dof_props[:, 4].clone().to(device=self.device)
            self.dof_upper_limits = self.dof_props[:, 5].clone().to(device=self.device)
        else:
            raise RuntimeError("DOF properties must be a tensor")

        # Initialize tensors
        # Active targets: base (6) + fingers (12)
        num_active = self.NUM_BASE_DOFS + self.NUM_ACTIVE_FINGER_DOFS
        self.prev_active_targets = torch.zeros(
            (self.num_envs, num_active), device=self.device
        )
        self.current_targets = torch.zeros(
            (self.num_envs, self.num_dof), device=self.device
        )

        # Precompute active DOF limits
        self._precompute_active_limits()

        # Precompute coupling mappings
        self._precompute_coupling_mappings()

        # Precompute action to active target mask
        self._precompute_action_mask()

        # Assign function pointers based on control mode
        if self.action_control_mode == "position":
            self._compute_targets_fn = self._compute_targets_position
        else:  # position_delta
            self._compute_targets_fn = self._compute_targets_position_delta

        logger.info(
            f"ActionProcessor initialized: control_mode={self.action_control_mode}, "
            f"base_control={self.policy_controls_hand_base}, "
            f"finger_control={self.policy_controls_fingers}"
        )

        # Note: finalize_setup() must be called after control_dt is available

    @initialization_only
    def finalize_setup(self):
        """
        Complete setup after control_dt is available.
        Must be called before process_actions can be used.
        """
        if self._initialized:
            raise RuntimeError("ActionProcessor already finalized")

        # Precompute max deltas now that control_dt is available
        self._precompute_max_deltas()

        self._initialized = True
        logger.debug(f"ActionProcessor finalized with control_dt={self.control_dt}")

    def process_actions(self, actions):
        """
        Process actions to generate DOF position targets.

        Note: For two-stage initialization, this method can be called before finalize_setup().
        In that case, it will set zero targets to prevent undefined behavior.

        Args:
            actions: Action tensor from policy (batch_size, num_actions)

        Returns:
            Success flag
        """
        # Actions must never be None
        if actions is None:
            raise RuntimeError("Actions cannot be None")

        # Store actions
        self.actions = actions.clone()

        # Special handling for two-stage initialization
        # If called before finalize_setup(), set zero targets
        if not self._initialized:
            # Ensure current_targets exists
            if self.current_targets is None:
                self.current_targets = torch.zeros(
                    (self.num_envs, self.num_dof), device=self.device
                )
            # Set all targets to zero
            self.current_targets.zero_()

            # Apply zero targets to simulation
            self.gym.set_dof_position_target_tensor(
                self.sim, gymtorch.unwrap_tensor(self.current_targets)
            )
            return True

        # Normal processing after initialization
        # Step 1: Compute active targets (6D + 12D) from previous active targets and actions
        active_targets = self._compute_targets_fn(actions)

        # Step 2: Apply coupling to get full DOF targets (6D + 20D)
        full_targets = self.apply_coupling(active_targets)

        # Update active targets for next iteration
        self.prev_active_targets = active_targets.clone()

        # Store full targets
        self.current_targets = full_targets

        # Debug: Multi-environment CPU/GPU issue tracking (see roadmap.md #5)
        if hasattr(self.parent, "_debug_step_counter"):
            self.parent._debug_step_counter += 1
        else:
            self.parent._debug_step_counter = 0

        if (
            self.parent._debug_step_counter % 200 == 50
            and self.parent._debug_step_counter > 0
            and self.num_envs > 1
        ):
            logger.debug(
                f"ActionProcessor Step {self.parent._debug_step_counter}: Multi-env targets"
            )
            logger.debug(
                f"  Device: {self.device}, Shape: {self.current_targets.shape}"
            )
            # Check if targets are identical across environments
            if self.num_envs >= 2:
                identical = torch.allclose(
                    self.current_targets[0], self.current_targets[1]
                )
                logger.debug(f"  Env 0 vs Env 1 targets identical: {identical}")

        # Apply targets to simulation
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.current_targets)
        )

        return True

    @post_initialization_only
    def _compute_targets_position(self, actions):
        """
        Compute targets for position control mode.

        Strategy:
        1. Scale action from [-1, 1] to [dof_min, dof_max]
        2. Compare with previous target, compute diff
        3. Clamp diff to [-max_delta, max_delta] based on velocity limits
        4. Apply clamped diff to previous target

        Args:
            actions: Raw actions in [-1, 1]

        Returns:
            Active targets (6D + 12D)
        """
        # Start with previous targets
        new_targets = self.prev_active_targets.clone()

        # Create scaled actions tensor (full 18D)
        scaled_actions = torch.zeros_like(self.prev_active_targets)

        # Scale and place actions in the controlled portions
        scaled_actions[:, self.active_target_mask] = self._scale_actions_to_limits(
            actions
        )

        # Compute diff only for controlled portions
        diff = torch.zeros_like(self.prev_active_targets)
        diff[:, self.active_target_mask] = (
            scaled_actions[:, self.active_target_mask]
            - self.prev_active_targets[:, self.active_target_mask]
        )

        # Clamp diff based on velocity limits
        clamped_diff = torch.clamp(diff, -self.max_deltas, self.max_deltas)

        # Apply diff to get new targets
        new_targets = self.prev_active_targets + clamped_diff

        return new_targets

    @post_initialization_only
    def _compute_targets_position_delta(self, actions):
        """
        Compute targets for position_delta control mode.

        Strategy:
        1. Scale action from [-1, 1] to [-max_delta, max_delta]
        2. Apply delta to previous target
        3. Clamp to [dof_min, dof_max]

        Args:
            actions: Raw actions in [-1, 1]

        Returns:
            Active targets (6D + 12D)
        """
        # Start with previous targets
        new_targets = self.prev_active_targets.clone()

        # Create scaled deltas tensor (full 18D)
        scaled_deltas = torch.zeros_like(self.prev_active_targets)

        # Scale actions to velocity deltas for controlled portions
        scaled_deltas[:, self.active_target_mask] = (
            actions * self.max_deltas[self.active_target_mask]
        )

        # Apply deltas
        new_targets = self.prev_active_targets + scaled_deltas

        # Clamp to DOF limits
        clamped_targets = torch.clamp(
            new_targets, self.active_lower_limits, self.active_upper_limits
        )

        return clamped_targets

    @post_initialization_only
    def _scale_actions_to_limits(self, actions):
        """
        Scale actions from [-1, 1] to DOF limits for position mode.

        Args:
            actions: Raw actions in [-1, 1] (only for controlled DOFs)

        Returns:
            Scaled actions in DOF limit ranges (only for controlled DOFs)
        """
        # Get limits only for controlled DOFs
        controlled_lower = self.active_lower_limits[self.active_target_mask]
        controlled_upper = self.active_upper_limits[self.active_target_mask]

        # Map from [-1, 1] to [lower, upper]
        # action = -1 → lower limit
        # action = +1 → upper limit
        scaled = (actions + 1.0) * 0.5 * (
            controlled_upper - controlled_lower
        ) + controlled_lower

        return scaled

    @initialization_only
    def _precompute_active_limits(self):
        """
        Precompute DOF limits for active DOFs during initialization.
        Always maintains full 6D+12D limits regardless of control options.
        """
        # Base DOFs (first 6) - always maintain all 6
        base_lower = self.dof_lower_limits[: self.NUM_BASE_DOFS]
        base_upper = self.dof_upper_limits[: self.NUM_BASE_DOFS]

        # For finger DOFs, we need to map from 12 active controls to their primary DOFs
        # Using the first joint in each coupling group
        finger_lower_list = []
        finger_upper_list = []

        for i in range(self.NUM_ACTIVE_FINGER_DOFS):
            if i in self.finger_coupling_map:
                # Get first joint in coupling group
                first_joint = self.finger_coupling_map[i][0]
                joint_name = (
                    first_joint[0] if isinstance(first_joint, tuple) else first_joint
                )
                # Find DOF index
                for j, name in enumerate(self.dof_names):
                    if name == joint_name:
                        finger_lower_list.append(self.dof_lower_limits[j])
                        finger_upper_list.append(self.dof_upper_limits[j])
                        break

        finger_lower = torch.stack(finger_lower_list)
        finger_upper = torch.stack(finger_upper_list)

        # Always maintain full 6D+12D active limits
        self.active_lower_limits = torch.cat([base_lower, finger_lower])
        self.active_upper_limits = torch.cat([base_upper, finger_upper])

    @initialization_only
    def _precompute_max_deltas(self):
        """
        Precompute maximum position deltas based on velocity limits and control_dt.
        Called once after control_dt is available.
        """
        self.max_deltas = torch.zeros(
            self.NUM_BASE_DOFS + self.NUM_ACTIVE_FINGER_DOFS, device=self.device
        )

        # Base DOFs
        self.max_deltas[0:3] = (
            self.control_dt * self.policy_base_lin_velocity_limit
        )  # Linear
        self.max_deltas[3:6] = (
            self.control_dt * self.policy_base_ang_velocity_limit
        )  # Angular

        # Finger DOFs
        self.max_deltas[6:] = self.control_dt * self.policy_finger_velocity_limit

    @initialization_only
    def _precompute_coupling_mappings(self):
        """
        Precompute coupling indices and scales for vectorized operations.
        """
        if not self.dof_names:
            raise RuntimeError(
                "DOF names not available. Cannot precompute coupling mappings."
            )

        # Build lists for coupling mappings
        coupling_indices_list = []
        coupling_scales_list = []

        # Build DOF name to index mapping
        dof_name_to_idx = {name: i for i, name in enumerate(self.dof_names)}

        # Process each finger control
        for finger_idx in range(self.NUM_ACTIVE_FINGER_DOFS):
            if finger_idx not in self.finger_coupling_map:
                continue

            joint_mapping = self.finger_coupling_map[finger_idx]

            for joint_spec in joint_mapping:
                # Parse joint specification
                if isinstance(joint_spec, tuple):
                    joint_name, coupling_scale = joint_spec
                else:
                    joint_name = joint_spec
                    coupling_scale = 1.0

                # Get DOF index
                if joint_name not in dof_name_to_idx:
                    raise RuntimeError(f"Joint '{joint_name}' not found in DOF names")

                dof_idx = dof_name_to_idx[joint_name]
                coupling_indices_list.append([finger_idx, dof_idx])
                coupling_scales_list.append(coupling_scale)

        # Convert to tensors
        self.coupling_indices = torch.tensor(
            coupling_indices_list, dtype=torch.long, device=self.device
        )
        self.coupling_scales = torch.tensor(
            coupling_scales_list, dtype=torch.float32, device=self.device
        )

        # Find middle finger spread index
        if "r_f_joint3_1" in dof_name_to_idx:
            self.middle_finger_spread_idx = dof_name_to_idx["r_f_joint3_1"]
        else:
            raise RuntimeError("Middle finger spread joint 'r_f_joint3_1' not found")

    @initialization_only
    def _precompute_action_mask(self):
        """
        Precompute boolean mask for which active targets are controlled by policy.
        """
        # Create boolean mask for 18 active targets (6 base + 12 fingers)
        self.active_target_mask = torch.zeros(
            self.NUM_BASE_DOFS + self.NUM_ACTIVE_FINGER_DOFS,
            dtype=torch.bool,
            device=self.device,
        )

        # Set True for controlled portions
        if self.policy_controls_hand_base:
            self.active_target_mask[: self.NUM_BASE_DOFS] = True

        if self.policy_controls_fingers:
            self.active_target_mask[self.NUM_BASE_DOFS :] = True

    def reset_targets(self, env_ids=None):
        """
        Reset previous targets to match current DOF positions.
        Should be called after environment reset to avoid jumps.

        Args:
            env_ids: Optional tensor of environment IDs to reset. If None, reset all.
        """
        if env_ids is None:
            # Reset all environments
            self.prev_active_targets.zero_()
        else:
            # Reset specific environments
            self.prev_active_targets[env_ids] = 0.0

        # Also reset current targets to ensure consistency
        if self.current_targets is not None:
            if env_ids is None:
                self.current_targets.zero_()
            else:
                self.current_targets[env_ids] = 0.0

        # Apply the reset targets to the simulation to ensure DOFs move to reset position
        if self.current_targets is not None:
            self.gym.set_dof_position_target_tensor(
                self.sim, gymtorch.unwrap_tensor(self.current_targets)
            )

    @post_initialization_only
    def apply_coupling(self, active_targets):
        """
        Apply coupling to map active targets (6D + 12D) to full DOF targets (26D).
        Uses precomputed indices and scales for vectorized operation.

        Args:
            active_targets: Active target positions (batch_size, 18)

        Returns:
            Full DOF targets (batch_size, 26)
        """
        # Initialize full targets with zeros
        full_targets = torch.zeros((self.num_envs, self.num_dof), device=self.device)

        # Copy base targets directly (no coupling)
        full_targets[:, : self.NUM_BASE_DOFS] = active_targets[:, : self.NUM_BASE_DOFS]

        # Get finger targets
        finger_targets = active_targets[:, self.NUM_BASE_DOFS :]

        # Vectorized coupling application
        # Extract finger values for each coupling
        finger_indices = self.coupling_indices[:, 0]  # Which finger control
        dof_indices = self.coupling_indices[:, 1]  # Which DOF to set

        # Gather finger values for all couplings
        gathered_values = finger_targets[
            :, finger_indices
        ]  # (batch_size, num_couplings)

        # Apply coupling scales
        scaled_values = gathered_values * self.coupling_scales.unsqueeze(0)

        # Scatter to full DOF targets
        # Use advanced indexing
        batch_indices = torch.arange(self.num_envs, device=self.device).unsqueeze(1)
        batch_indices = batch_indices.expand(-1, len(dof_indices))

        full_targets[batch_indices, dof_indices] = scaled_values

        # Set middle finger spread to 0
        if self.middle_finger_spread_idx is not None:
            full_targets[:, self.middle_finger_spread_idx] = 0.0

        return full_targets
