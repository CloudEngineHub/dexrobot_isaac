"""
Action processor component for DexHand environment.

This module provides action processing coordination for the DexHand environment.
The ActionProcessor orchestrates the action processing pipeline by coordinating:
- ActionRules: Pre-action rules, action rules, post-action filters, and coupling
- ActionScaling: Mathematical scaling and transformation utilities
- Initialization and two-stage setup management
- Integration with Isaac Gym simulation

The processor itself focuses on coordination rather than implementation details.
"""

# Import standard libraries
from loguru import logger
from functools import wraps
from typing import Callable, Dict, Any, Optional

# Import IsaacGym first (before torch)
from isaacgym import gymtorch

# Then import torch
import torch

# Import constants
from dexhand_env.constants import (
    NUM_BASE_DOFS,
    NUM_ACTIVE_FINGER_DOFS,
    FINGER_COUPLING_MAP,
)

# Import action components
from dexhand_env.components.action import ActionRules, ActionScaling


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
    Coordinates action processing for the DexHand environment.

    This component orchestrates the action processing pipeline:
    - Coordinates ActionRules and ActionScaling components
    - Manages the rule-based action processing flow
    - Handles two-stage initialization for control_dt dependency
    - Integrates with Isaac Gym simulation for target application
    - Provides API for rule registration and configuration

    The processor focuses on coordination and delegation rather than
    implementing mathematical operations directly.
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

        # Tensors with clearer naming
        self.active_prev_targets = (
            None  # Previous active targets (18D: 6 base + 12 finger)
        )
        self.active_rule_targets = None  # Output of pre-action rule (18D)
        self.active_raw_targets = None  # Output of action rule (18D)
        self.active_next_targets = None  # Output of post-action filters (18D)
        self.full_dof_targets = None  # Full DOF targets after coupling (26D)
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

        # Action components
        self.action_rules = ActionRules(parent=self)
        self.action_scaling = ActionScaling(parent=self)

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
                - post_action_filters: list of str (optional, default: ["velocity_clamp", "position_clamp"])
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

        # Initialize tensors with clearer naming
        # Active targets: base (6) + fingers (12)
        num_active = self.NUM_BASE_DOFS + self.NUM_ACTIVE_FINGER_DOFS
        self.active_prev_targets = torch.zeros(
            (self.num_envs, num_active), device=self.device
        )
        self.full_dof_targets = torch.zeros(
            (self.num_envs, self.num_dof), device=self.device
        )

        # Configure which post-action filters are enabled
        self._enabled_post_action_filters = config.get(
            "post_action_filters", ["velocity_clamp", "position_clamp"]
        )

        # Precompute active DOF limits
        self._precompute_active_limits()

        # Precompute coupling mappings
        self._precompute_coupling_mappings()

        # Precompute action to active target mask
        self._precompute_action_mask()

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

        # Initialize post-action filters now that control_dt is available
        self.action_rules.initialize_post_action_filters()

    def apply_pre_action_rule(
        self, active_prev_targets: torch.Tensor, state: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Apply pre-action rule separately (called before main processing).

        Args:
            active_prev_targets: Previous active targets (18D)
            state: State dict with 'obs_dict' and 'env' keys

        Returns:
            active_rule_targets: Output of pre-action rule (18D)
        """
        return self.action_rules.apply_pre_action_rule(active_prev_targets, state)

    def process_actions(self, actions: torch.Tensor, active_rule_targets: torch.Tensor):
        """
        Process actions with pre-computed rule targets.

        Args:
            actions: Action tensor from policy (batch_size, num_actions)
            active_rule_targets: Pre-computed rule targets (18D)

        Returns:
            Success flag
        """
        # Actions must never be None
        if actions is None:
            raise RuntimeError("Actions cannot be None")

        # Store actions
        self.actions = actions.clone()
        self.active_rule_targets = active_rule_targets

        # Special handling for two-stage initialization
        # If called before finalize_setup(), set zero targets
        if not self._initialized:
            # full_dof_targets must exist from initialize_from_config
            if self.full_dof_targets is None:
                raise RuntimeError(
                    "full_dof_targets is None - initialize_from_config() failed"
                )
            # Set all targets to zero
            self.full_dof_targets.zero_()

            # Apply zero targets to simulation
            self.gym.set_dof_position_target_tensor(
                self.sim, gymtorch.unwrap_tensor(self.full_dof_targets)
            )
            return True

        # Normal processing after initialization
        # Step 1: Apply action rule
        config = {
            "control_mode": self.action_control_mode,
            "policy_controls_base": self.policy_controls_hand_base,
            "policy_controls_fingers": self.policy_controls_fingers,
        }
        self.active_raw_targets = self.action_rules.apply_action_rule(
            self.active_prev_targets, active_rule_targets, actions, config
        )

        # Step 2: Apply post-action filters
        self.active_next_targets = self.action_rules.apply_post_action_filters(
            self.active_prev_targets, active_rule_targets, self.active_raw_targets
        )

        # Step 3: Apply coupling rule
        self.full_dof_targets = self.action_rules.apply_coupling_rule(
            self.active_next_targets
        )

        # Update state for next iteration
        self.active_prev_targets = self.active_next_targets.clone()

        # Debug logging for multi-environment debugging
        self._debug_multi_env_targets()

        # Apply targets to simulation
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.full_dof_targets)
        )

        return True

    def _debug_multi_env_targets(self):
        """Debug logging for multi-environment CPU/GPU issue tracking (see roadmap.md #5)."""
        # Initialize debug counter if not set (should be done in parent __init__)
        if not hasattr(self.parent, "_debug_step_counter"):
            self.parent._debug_step_counter = 0
        self.parent._debug_step_counter += 1

        if (
            self.parent._debug_step_counter % 200 == 50
            and self.parent._debug_step_counter > 0
            and self.num_envs > 1
        ):
            logger.debug(
                f"ActionProcessor Step {self.parent._debug_step_counter}: Multi-env targets"
            )
            logger.debug(
                f"  Device: {self.device}, Shape: {self.full_dof_targets.shape}"
            )
            # Check if targets are identical across environments
            if self.num_envs >= 2:
                identical = torch.allclose(
                    self.full_dof_targets[0], self.full_dof_targets[1]
                )
                logger.debug(f"  Env 0 vs Env 1 targets identical: {identical}")

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

        # Find middle finger spread index - fail fast if missing
        if "r_f_joint3_1" not in dof_name_to_idx:
            raise RuntimeError(
                "r_f_joint3_1 (middle finger spread) not found in DOF names - this indicates initialization bug"
            )
        self.middle_finger_spread_idx = dof_name_to_idx["r_f_joint3_1"]

        # Pre-compute inverse mapping for efficient extract_active_targets_from_full_dof
        # Map from DOF indices to finger control indices and scales
        self.inverse_dof_to_finger = torch.full(
            (self.num_dof,), -1, dtype=torch.long, device=self.device
        )
        self.inverse_dof_scales = torch.ones(
            self.num_dof, dtype=torch.float32, device=self.device
        )

        # Populate inverse mapping using existing coupling data
        for i, (finger_idx, dof_idx) in enumerate(self.coupling_indices):
            self.inverse_dof_to_finger[dof_idx] = finger_idx
            self.inverse_dof_scales[dof_idx] = self.coupling_scales[i]

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

    def reset_targets(self, env_ids=None, dof_positions=None):
        """
        Reset targets to specified DOF positions or zero.
        Should be called after environment reset to avoid jumps.

        Args:
            env_ids: Optional tensor of environment IDs to reset. If None, reset all.
            dof_positions: Optional tensor of DOF positions to set targets to.
                          If None, targets are set to zero (default behavior).
                          Shape: (len(env_ids), num_dofs) or (num_envs, num_dofs) if env_ids is None.
        """
        if self.full_dof_targets is None:
            raise RuntimeError("full_dof_targets is None - initialization failed")

        if dof_positions is not None:
            # Set targets to match provided DOF positions
            if env_ids is None:
                # Reset all environments with provided positions
                active_targets = self.extract_active_targets_from_full_dof(
                    dof_positions
                )
                self.active_prev_targets[:] = active_targets
                self.full_dof_targets[:] = dof_positions
            else:
                # Reset specific environments with provided positions
                active_targets = self.extract_active_targets_from_full_dof(
                    dof_positions
                )
                self.active_prev_targets[env_ids, :] = active_targets
                self.full_dof_targets[env_ids, :] = dof_positions
        else:
            # Default behavior: set targets to zero
            if env_ids is None:
                # Reset all environments
                self.active_prev_targets.zero_()
                self.full_dof_targets.zero_()
            else:
                # Reset specific environments
                self.active_prev_targets[env_ids] = 0.0
                self.full_dof_targets[env_ids] = 0.0

        # Apply the reset targets to the simulation to ensure DOFs move to reset position
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.full_dof_targets)
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
        full_targets[:, self.middle_finger_spread_idx] = 0.0

        return full_targets

    def extract_active_targets_from_full_dof(self, full_dof_positions):
        """
        Extract active targets (18D) from full DOF positions (26D).
        This is the inverse of apply_coupling() operation.
        Uses precomputed inverse mapping for efficient vectorized operation.

        Args:
            full_dof_positions: Full DOF positions (batch_size, 26)

        Returns:
            Active targets (batch_size, 18) - 6 base + 12 finger controls
        """
        # Ensure inverse mapping is available (set during initialize_from_config)
        if self.inverse_dof_to_finger is None:
            raise RuntimeError(
                "Inverse mapping not available. Cannot extract active targets."
            )

        batch_size = full_dof_positions.shape[0]
        active_targets = torch.zeros(
            (batch_size, self.NUM_BASE_DOFS + self.NUM_ACTIVE_FINGER_DOFS),
            device=self.device,
        )

        # Extract base targets directly (1:1 mapping)
        active_targets[:, : self.NUM_BASE_DOFS] = full_dof_positions[
            :, : self.NUM_BASE_DOFS
        ]

        # Extract finger targets using precomputed inverse mapping (vectorized)
        finger_dof_start = self.NUM_BASE_DOFS
        finger_dof_positions = full_dof_positions[:, finger_dof_start:]
        finger_indices = self.inverse_dof_to_finger[finger_dof_start:]
        scales = self.inverse_dof_scales[finger_dof_start:]

        # Create mask for valid mappings (finger_indices >= 0)
        valid_mask = finger_indices >= 0

        if valid_mask.any():
            # Vectorized extraction: divide by scales and assign to correct finger indices
            valid_dof_positions = finger_dof_positions[:, valid_mask]
            valid_finger_indices = finger_indices[valid_mask]
            valid_scales = scales[valid_mask]

            # Scale the DOF positions and scatter to finger targets
            scaled_positions = valid_dof_positions / valid_scales.unsqueeze(0)
            active_targets[
                :, self.NUM_BASE_DOFS + valid_finger_indices
            ] = scaled_positions

        return active_targets

    # ============================================================================
    # Public API for Rule Registration (delegates to ActionRules)
    # ============================================================================

    def set_pre_action_rule(
        self, rule: Optional[Callable[[torch.Tensor, Dict[str, Any]], torch.Tensor]]
    ):
        """
        Set custom pre-action rule.

        Args:
            rule: Function (active_prev_targets, state) -> active_rule_targets
                  or None to use identity function
        """
        self.action_rules.set_pre_action_rule(rule)

    def set_action_rule(
        self,
        rule: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]], torch.Tensor
        ],
    ):
        """
        Set custom action rule.

        Args:
            rule: Function (active_prev_targets, active_rule_targets, actions, config) -> active_raw_targets
        """
        self.action_rules.set_action_rule(rule)

    def set_coupling_rule(self, rule: Callable[[torch.Tensor], torch.Tensor]):
        """
        Set custom coupling rule.

        Args:
            rule: Function (active_targets) -> full_dof_targets
        """
        self.action_rules.set_coupling_rule(rule)

    def register_post_action_filter(
        self,
        name: str,
        filter_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    ):
        """
        Register a custom post-action filter.

        Args:
            name: Name for the filter (used in config to enable)
            filter_fn: Function (active_prev_targets, active_rule_targets, active_targets) -> filtered_targets
        """
        self.action_rules.register_post_action_filter(name, filter_fn)

    @post_initialization_only
    def unscale_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Convert actions from normalized space [-1, +1] to physical units.

        This method reverses the scaling applied during action processing to show
        what physical values the normalized actions represent. Useful for debugging
        and visualization purposes.

        Args:
            actions: Normalized actions in range [-1, +1], shape (num_envs, num_actions)

        Returns:
            torch.Tensor: Actions in physical units (meters for base translation,
                         radians for base rotation and finger joints)
        """
        if actions is None or actions.numel() == 0:
            return (
                torch.zeros_like(actions)
                if actions is not None
                else torch.zeros((self.num_envs, 0), device=self.device)
            )

        if self.action_control_mode == "position":
            # For position mode: reverse scale_to_limits operation
            # Map [-1,1] to [lower_limits, upper_limits] for policy-controlled DOFs
            return self.action_scaling.scale_to_limits(
                actions,
                self.active_lower_limits[self.active_target_mask],
                self.active_upper_limits[self.active_target_mask],
            )
        else:  # position_delta mode
            # For position_delta mode: convert to velocity deltas using max_deltas
            # Actions represent fraction of maximum velocity change
            return actions * self.max_deltas[self.active_target_mask]
