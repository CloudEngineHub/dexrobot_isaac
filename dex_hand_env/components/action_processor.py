"""
Action processor component for DexHand environment.

This module provides action processing functionality for the DexHand environment,
including action scaling, mapping, and PD control.
"""

# Import standard libraries
import torch
from loguru import logger

# Import IsaacGym
from isaacgym import gymtorch

# Import constants
from dex_hand_env.constants import (
    NUM_BASE_DOFS,
    NUM_ACTIVE_FINGER_DOFS,
    FINGER_COUPLING_MAP,
)


class ActionProcessor:
    """
    Processes actions for the DexHand environment.

    This component provides functionality to:
    - Map policy actions to robot DOFs
    - Apply action scaling and transformations
    - Handle different control modes (position, position_delta)
    - Manage PD control targets
    """

    def __init__(
        self, gym, sim, num_envs, device, dof_props, hand_asset, physics_manager
    ):
        """
        Initialize the action processor.

        Args:
            gym: The isaacgym gym instance
            sim: The isaacgym simulation instance
            num_envs: Number of environments
            device: PyTorch device
            dof_props: DOF properties tensor
            hand_asset: Hand asset for getting DOF names
            physics_manager: PhysicsManager instance for accessing control_dt
        """
        self.gym = gym
        self.sim = sim
        self.num_envs = num_envs
        self.device = device
        self.hand_asset = hand_asset

        # Store DOF names if asset is provided
        self.dof_names = []
        if hand_asset is not None:
            self.dof_names = self.gym.get_asset_dof_names(hand_asset)
            logger.info(
                f"ActionProcessor initialized with {len(self.dof_names)} DOF names from asset"
            )

        # Control settings
        self.action_control_mode = "position"  # position or position_delta
        self.policy_controls_hand_base = True
        self.policy_controls_fingers = True

        # Constants for action dimensions (imported from constants.py)
        self.NUM_BASE_DOFS = NUM_BASE_DOFS
        self.NUM_ACTIVE_FINGER_DOFS = NUM_ACTIVE_FINGER_DOFS

        # Finger DOF coupling mapping (imported from constants.py)
        self.finger_coupling_map = FINGER_COUPLING_MAP

        # DOF limits
        self.dof_props = dof_props
        self.dof_lower_limits = None
        self.dof_upper_limits = None
        self.num_dof = 0

        # Default targets (used when DOFs are not controlled by policy)
        self.default_base_targets = torch.zeros(6, device=self.device)
        self.default_finger_targets = torch.zeros(12, device=self.device)

        # Velocity limits for safety
        self.policy_finger_velocity_limit = 2.0
        self.policy_base_lin_velocity_limit = 1.0
        self.policy_base_ang_velocity_limit = 1.5

        # Reference to physics manager for accessing control_dt (single source of truth)
        self.physics_manager = physics_manager

        # Initialize with empty tensors - will be properly initialized later
        self.dof_pos = None
        self.prev_active_targets = None
        self.current_targets = None
        self.actions = None

        # Action space scaling coefficients (computed during setup)
        # These convert from normalized [-1, 1] to physical units [min, max]
        self.action_space_scale = None
        self.action_space_bias = None

        # Cache for DOF name to index mapping
        self._dof_name_to_idx_cache = None

        # Function pointer for unscaling (assigned during setup based on control mode)
        self._unscale_actions_fn = None

    @property
    def control_dt(self):
        """Access control_dt from physics manager (single source of truth)."""
        if self.physics_manager is None:
            raise RuntimeError("physics_manager not set. Cannot access control_dt.")
        return self.physics_manager.control_dt

    def setup(self, num_dof, dof_props):
        """
        Set up action processor with DOF information.
        Note: This does basic setup. Call setup_action_scaling() after control_dt is determined.

        Args:
            num_dof: Number of DOFs in the model
            dof_props: DOF properties tensor (required)
        """
        self.num_dof = num_dof

        # Initialize previous targets tensor
        self.prev_active_targets = torch.zeros(
            (self.num_envs, self.NUM_BASE_DOFS + self.NUM_ACTIVE_FINGER_DOFS),
            device=self.device,
            dtype=torch.float,
        )

        # Initialize with default values
        # Base position targets (RELATIVE motion from spawn point)
        self.prev_active_targets[:, 0] = 0.0  # ARTx - relative X displacement
        self.prev_active_targets[:, 1] = 0.0  # ARTy - relative Y displacement
        self.prev_active_targets[
            :, 2
        ] = 0.0  # ARTz - relative Z displacement (0 = stay at spawn height)

        # Initialize rotation targets - default is identity quaternion (0,0,0 in axis-angle)
        self.prev_active_targets[:, 3:6] = 0.0  # ARRx, ARRy, ARRz

        # Create current targets tensor
        self.current_targets = torch.zeros(
            (self.num_envs, self.num_dof), device=self.device
        )

        # Initialize DOF limits
        self.dof_props = dof_props

        # Check if it's a tensor (from TensorManager) or a dictionary
        if isinstance(dof_props, torch.Tensor):
            logger.debug(f"DOF props is a tensor with shape: {dof_props.shape}")
            # Format is [stiffness, damping, friction, armature, min, max]
            # Extract limits from the tensor (indices 4 and 5 are min and max)
            self.dof_lower_limits = dof_props[:, 4].clone().to(device=self.device)
            self.dof_upper_limits = dof_props[:, 5].clone().to(device=self.device)
        elif (
            isinstance(dof_props, dict)
            and "lower" in dof_props
            and "upper" in dof_props
        ):
            # Extract DOF limits from dictionary
            self.dof_lower_limits = torch.tensor(
                dof_props["lower"], dtype=torch.float, device=self.device
            )
            self.dof_upper_limits = torch.tensor(
                dof_props["upper"], dtype=torch.float, device=self.device
            )
        else:
            raise RuntimeError(
                f"DOF properties format not recognized: {type(dof_props)}. Expected torch.Tensor or dict with 'lower'/'upper' keys. Cannot proceed without valid DOF limits."
            )

        # Compute action scaling coefficients
        self._compute_action_scaling_coeffs()

        # Assign unscaling function based on control mode (no runtime branching)
        if self.action_control_mode == "position":
            self._unscale_actions_fn = self._unscale_actions_position
        else:
            self._unscale_actions_fn = self._unscale_actions_position_delta

    def set_control_mode(self, mode):
        """
        Set the control mode for action processing.
        Note: Must be called before setup() to take effect.

        Args:
            mode: Control mode string ("position" or "position_delta")
        """
        valid_modes = ["position", "position_delta"]
        if mode not in valid_modes:
            raise ValueError(
                f"Invalid control mode: {mode}. Must be one of {valid_modes}"
            )

        # Only allow setting control mode before setup
        if hasattr(self, "action_space_scale") and self.action_space_scale is not None:
            raise RuntimeError(
                "Cannot change control mode after setup. Control mode must be set before calling setup()."
            )

        self.action_control_mode = mode

    def set_control_options(
        self, policy_controls_hand_base=None, policy_controls_fingers=None
    ):
        """
        Set which parts of the hand are controlled by the policy vs rule-based control.

        When policy_controls_hand_base=False, the hand base is controlled by rule-based controllers
        instead of policy actions. The base DOFs can still move, but their motion is
        determined by programmatic rules rather than learned policy.

        Args:
            policy_controls_hand_base: Whether the POLICY controls the hand base (6 DOFs)
                                      If False, use rule-based control for base motion
            policy_controls_fingers: Whether the POLICY controls the finger joints (12 DOFs)
                                    If False, use rule-based control for finger motion
        """
        if policy_controls_hand_base is not None:
            self.policy_controls_hand_base = policy_controls_hand_base
        if policy_controls_fingers is not None:
            self.policy_controls_fingers = policy_controls_fingers

        # Validate control options - at least one must be True
        if not self.policy_controls_hand_base and not self.policy_controls_fingers:
            raise ValueError(
                "At least one of policy_controls_hand_base or policy_controls_fingers must be True"
            )

    def set_default_targets(self, base_targets=None, finger_targets=None):
        """
        Set default targets for uncontrolled DOFs.

        Args:
            base_targets: Default targets for base DOFs
            finger_targets: Default targets for finger DOFs
        """
        if base_targets is not None:
            if isinstance(base_targets, list):
                base_targets = torch.tensor(base_targets, device=self.device)
            self.default_base_targets = base_targets

        if finger_targets is not None:
            if isinstance(finger_targets, list):
                finger_targets = torch.tensor(finger_targets, device=self.device)
            self.default_finger_targets = finger_targets

    def set_velocity_limits(
        self, finger_vel_limit=None, base_lin_vel_limit=None, base_ang_vel_limit=None
    ):
        """
        Set component-wise velocity limits for position_delta mode action scaling.

        These limits are applied per component (each joint/axis independently):
        - finger_vel_limit: Applied to each finger joint individually
        - base_lin_vel_limit: Applied to each base linear axis (x, y, z) individually
        - base_ang_vel_limit: Applied to each base angular axis (rx, ry, rz) individually

        Args:
            finger_vel_limit: Maximum velocity for each finger joint (rad/s)
            base_lin_vel_limit: Maximum velocity for each base linear axis (m/s)
            base_ang_vel_limit: Maximum velocity for each base angular axis (rad/s)
        """
        if finger_vel_limit is not None:
            self.policy_finger_velocity_limit = finger_vel_limit
        if base_lin_vel_limit is not None:
            self.policy_base_lin_velocity_limit = base_lin_vel_limit
        if base_ang_vel_limit is not None:
            self.policy_base_ang_velocity_limit = base_ang_vel_limit

    def process_actions(
        self,
        actions,
        dof_pos,
        joint_to_control=None,
        active_joint_names=None,
        task_targets=None,
    ):
        """
        Process actions to generate DOF position targets.

        Args:
            actions: Action tensor from policy
            dof_pos: Current DOF positions
            joint_to_control: Mapping from joint names to control names
            active_joint_names: List of active joint names
            task_targets: Optional task-specific targets for uncontrolled DOFs

        Returns:
            Success flag
        """
        try:
            # Validate inputs
            if not self._validate_inputs(
                actions, dof_pos, joint_to_control, active_joint_names
            ):
                return False

            # Store actions for reference
            self.actions = actions.clone()
            self.dof_pos = dof_pos

            # Initialize targets with previous targets (maintain position unless commanded)
            # CRITICAL: In position control, targets should persist between steps
            if self.current_targets is not None:
                targets = self.current_targets.clone()
            else:
                # Only on first step, initialize with current positions
                targets = self.dof_pos.clone()

            # Split actions into base and finger components
            action_idx = 0

            # Process base DOFs
            action_idx = self._process_base_dofs(targets, action_idx, task_targets)

            # Process finger DOFs
            action_idx = self._process_finger_dofs(
                targets, action_idx, task_targets, joint_to_control, active_joint_names
            )

            # Apply target positions with PD control
            return self._apply_pd_control(targets)

        except Exception as e:
            logger.error(f"Error in process_actions: {e}")
            logger.exception("Traceback:")
            return False

    def _map_normalized_to_range(self, normalized_value, range_min, range_max):
        """
        Map a normalized value from [-1, 1] to [range_min, range_max].

        This is the core mapping function used throughout the action processor:
        - In position mode: maps to DOF position limits
        - In position_delta mode: maps to velocity limits (converted to position deltas)

        Args:
            normalized_value: Value in [-1, 1] range from RL policy
            range_min: Minimum value of target range
            range_max: Maximum value of target range

        Returns:
            Mapped value in [range_min, range_max] range
        """
        return (normalized_value + 1.0) * 0.5 * (range_max - range_min) + range_min

    def _get_dof_name_to_idx_mapping(self):
        """
        Get cached DOF name to index mapping.

        Returns:
            Dictionary mapping DOF names to indices
        """
        if self._dof_name_to_idx_cache is None:
            self._dof_name_to_idx_cache = {}
            for i, name in enumerate(self.dof_names):
                self._dof_name_to_idx_cache[name] = i
        return self._dof_name_to_idx_cache

    def _expand_to_batch(self, tensor, num_envs):
        """
        Expand a 1D tensor to batch dimension if needed.

        Args:
            tensor: Tensor to expand (1D or already batched)
            num_envs: Number of environments

        Returns:
            Tensor of shape (num_envs, original_size)
        """
        if len(tensor.shape) == 1:
            return tensor.unsqueeze(0).expand(num_envs, -1)
        return tensor

    def _apply_targets_to_slice(
        self, targets_tensor, slice_obj, new_values, target_name="targets"
    ):
        """
        Apply new values to a slice of the targets tensor with shape validation.

        Args:
            targets_tensor: Full targets tensor to update
            slice_obj: Slice object or range for indexing
            new_values: Values to apply
            target_name: Name for error messages
        """
        target_slice = targets_tensor[:, slice_obj]
        if target_slice.shape != new_values.shape:
            raise RuntimeError(
                f"Cannot assign {target_name} with shape {new_values.shape} "
                f"to tensor slice with shape {target_slice.shape}"
            )
        targets_tensor[:, slice_obj] = new_values

    def _compute_position_targets(self, scaled_actions, prev_targets):
        """
        Compute position targets based on control mode.

        Args:
            scaled_actions: Actions already scaled to appropriate units
            prev_targets: Previous target positions (for delta mode)

        Returns:
            Position targets
        """
        if self.action_control_mode == "position":
            # Direct position control: scaled actions are the targets
            return scaled_actions
        elif self.action_control_mode == "position_delta":
            # Delta control: add scaled actions (velocities) to previous targets
            return prev_targets + scaled_actions
        else:
            raise ValueError(f"Unknown control mode: {self.action_control_mode}")

    def _validate_inputs(self, actions, dof_pos, joint_to_control, active_joint_names):
        """
        Validate inputs for action processing.

        Returns:
            bool: True if validation passes, False otherwise
        """
        if actions is None:
            logger.warning("Actions is None, skipping action processing")
            return False

        # Validate physics_manager is set for position_delta mode
        if (
            self.action_control_mode == "position_delta"
            and self.physics_manager is None
        ):
            raise RuntimeError(
                "physics_manager must be set before processing actions in position_delta mode."
            )

        # Validate joint mappings
        if self.policy_controls_fingers and (
            joint_to_control is None or active_joint_names is None
        ):
            logger.error(
                "joint_to_control and active_joint_names must be provided when controlling fingers"
            )
            return False

        # Fail fast on invalid tensor shapes
        if dof_pos.shape[1] == 0:
            logger.error(f"DOF position tensor has invalid shape {dof_pos.shape}")
            return False

        return True

    def _scale_base_actions(self, base_actions):
        """
        Scale base actions based on control mode.

        Args:
            base_actions: Raw base actions from policy

        Returns:
            Scaled base actions
        """
        num_dofs = min(self.NUM_BASE_DOFS, base_actions.shape[1])

        # Get limits based on control mode
        min_limits, max_limits = self._get_control_mode_limits(num_dofs, is_base=True)

        # Map normalized actions to appropriate limits
        scaled_base_actions = self._map_normalized_to_range(
            base_actions[:, :num_dofs], min_limits, max_limits
        )

        return scaled_base_actions

    def _process_base_dofs(self, targets, action_idx, task_targets):
        """
        Process base DOF actions and update targets.

        Args:
            targets: Target tensor to update
            action_idx: Current action index
            task_targets: Optional task-specific targets

        Returns:
            Updated action index
        """
        if self.policy_controls_hand_base:
            # Extract base actions
            if self.actions.shape[1] > 0 and self.NUM_BASE_DOFS > 0:
                base_actions = self.actions[
                    :, : min(self.NUM_BASE_DOFS, self.actions.shape[1])
                ]

                # Scale base actions
                scaled_base_actions = self._scale_base_actions(base_actions)

                # Compute position targets based on control mode
                base_pos_targets = self._compute_position_targets(
                    scaled_base_actions,
                    self.prev_active_targets[:, : self.NUM_BASE_DOFS],
                )

                # Apply targets to the base DOFs
                self._apply_targets_to_slice(
                    targets,
                    slice(0, self.NUM_BASE_DOFS),
                    base_pos_targets,
                    "base_pos_targets",
                )
                # Update previous targets
                self.prev_active_targets[:, : self.NUM_BASE_DOFS] = base_pos_targets

                # Update action index
                action_idx += self.NUM_BASE_DOFS
            else:
                # Get targets from task or use defaults
                if task_targets is not None and "base_targets" in task_targets:
                    targets[:, : self.NUM_BASE_DOFS] = task_targets["base_targets"]
                else:
                    # Expand default_base_targets to match batch dimension
                    expanded_targets = self._expand_to_batch(
                        self.default_base_targets, self.num_envs
                    )
                    targets[:, : self.NUM_BASE_DOFS] = expanded_targets
        else:
            # Hand base not controlled by policy - use task targets or defaults
            if task_targets is not None and "base_targets" in task_targets:
                if (
                    targets[:, : self.NUM_BASE_DOFS].shape
                    == task_targets["base_targets"].shape
                ):
                    targets[:, : self.NUM_BASE_DOFS] = task_targets["base_targets"]
                else:
                    logger.error(
                        "Cannot assign task_targets['base_targets'] to targets[:, :self.NUM_BASE_DOFS]"
                    )
            else:
                # Expand default_base_targets to match batch dimension
                expanded_targets = self._expand_to_batch(
                    self.default_base_targets, self.num_envs
                )
                self._apply_targets_to_slice(
                    targets,
                    slice(0, self.NUM_BASE_DOFS),
                    expanded_targets,
                    "default_base_targets",
                )

        return action_idx

    def _apply_raw_finger_targets(self, finger_targets, dof_pos):
        """
        Apply raw finger targets (physically meaningful values) with proper coupling.

        Args:
            finger_targets: torch.Tensor of shape (num_envs, NUM_ACTIVE_FINGER_DOFS) with raw joint targets in radians
            dof_pos: current DOF positions tensor
        """
        if finger_targets.shape[1] != self.NUM_ACTIVE_FINGER_DOFS:
            logger.warning(
                f"Expected {self.NUM_ACTIVE_FINGER_DOFS} finger targets, got {finger_targets.shape[1]}"
            )
            return

        # Get cached DOF name to index mapping
        dof_name_to_idx = self._get_dof_name_to_idx_mapping()

        # Apply coupling mapping with raw values
        self._apply_coupling_to_targets(
            self.current_targets, finger_targets, dof_name_to_idx, is_raw=True
        )

    def _apply_coupling_to_targets(
        self, targets, source_values, dof_name_to_idx, is_raw=False
    ):
        """
        Apply coupling mapping to convert source values to target DOF values.

        Args:
            targets: Target tensor to update
            source_values: Source values (actions or raw targets)
            dof_name_to_idx: DOF name to index mapping
            is_raw: Whether source values are raw targets (True) or normalized actions (False)
        """
        for action_idx, joint_mapping in self.finger_coupling_map.items():
            if action_idx >= source_values.shape[1]:
                continue

            source_value = source_values[:, action_idx]

            for joint_spec in joint_mapping:
                joint_name, coupling_scale = self._parse_joint_spec(joint_spec)

                if joint_name in dof_name_to_idx:
                    dof_idx = dof_name_to_idx[joint_name]

                    if is_raw:
                        # Raw values: just apply DOF coupling scale
                        targets[:, dof_idx] = source_value * coupling_scale
                    else:
                        # Normalized actions: compute full target with both action space scaling and DOF coupling
                        targets[:, dof_idx] = self._compute_joint_target(
                            source_value, dof_idx, coupling_scale
                        )

        # Special case: r_f_joint3_1 (middle finger spread) is always fixed to 0
        # This is a physical constraint of the DexHand model
        if "r_f_joint3_1" in dof_name_to_idx:
            targets[:, dof_name_to_idx["r_f_joint3_1"]] = 0.0

    def _process_finger_dofs(
        self, targets, action_idx, task_targets, joint_to_control, active_joint_names
    ):
        """
        Process finger DOF actions and update targets.

        Args:
            targets: Target tensor to update
            action_idx: Current action index
            task_targets: Optional task-specific targets
            joint_to_control: Mapping from joint names to control names
            active_joint_names: List of active joint names

        Returns:
            Updated action index
        """
        if self.policy_controls_fingers:
            # Extract finger actions
            if self.actions.shape[1] > action_idx and self.NUM_ACTIVE_FINGER_DOFS > 0:
                finger_end_idx = min(
                    action_idx + self.NUM_ACTIVE_FINGER_DOFS, self.actions.shape[1]
                )
                finger_actions = self.actions[:, action_idx:finger_end_idx]

                # Finger targets from actions
                finger_pos_targets = None

                if self.action_control_mode == "position":
                    # Direct position targets
                    finger_pos_targets = finger_actions
                elif self.action_control_mode == "position_delta":
                    # Incremental position changes
                    finger_pos_targets = (
                        self.prev_active_targets[:, self.NUM_BASE_DOFS :]
                        + finger_actions
                    )
                else:
                    logger.error(f"Unknown control mode: {self.action_control_mode}")

                # Apply targets to the finger DOFs using coupling logic
                if finger_pos_targets is not None and finger_pos_targets.shape[1] >= 12:
                    self._apply_finger_coupling(
                        targets, finger_actions, finger_pos_targets
                    )

                    # Update previous targets
                    self.prev_active_targets[
                        :, self.NUM_BASE_DOFS :
                    ] = finger_pos_targets
            else:
                # Get targets from task or use defaults
                self._apply_default_finger_targets(
                    targets, task_targets, joint_to_control, active_joint_names
                )

        return action_idx

    def _apply_finger_coupling(self, targets, finger_actions, finger_pos_targets):
        """
        Apply finger coupling logic to map actions to DOF targets.

        Args:
            targets: Target tensor to update
            finger_actions: Raw finger actions from policy
            finger_pos_targets: Computed finger position targets
        """
        # Get cached DOF name to index mapping
        dof_name_to_idx = self._get_dof_name_to_idx_mapping()

        # Apply coupling mapping
        self._apply_coupling_to_targets(
            targets, finger_actions, dof_name_to_idx, is_raw=False
        )

    def _compute_joint_target(self, raw_action_value, dof_idx, scale):
        """
        Compute target value for a single joint.

        Args:
            raw_action_value: Raw action value [-1, 1]
            dof_idx: DOF index for the joint
            scale: Coupling scale factor

        Returns:
            Target value for the joint
        """
        if self.action_control_mode == "position":
            # Position mode: map to DOF limits
            if self.dof_lower_limits is None or self.dof_upper_limits is None:
                raise RuntimeError(
                    f"DOF limits not available for computing joint target at DOF {dof_idx}"
                )

            dof_min = self.dof_lower_limits[dof_idx]
            dof_max = self.dof_upper_limits[dof_idx]
            scaled_action = self._map_normalized_to_range(
                raw_action_value, dof_min, dof_max
            )
            return scaled_action * scale

        elif self.action_control_mode == "position_delta":
            # Position delta mode: map to velocity limits as deltas
            max_pos_delta = self.control_dt * self.policy_finger_velocity_limit
            scaled_delta = (
                self._map_normalized_to_range(
                    raw_action_value, -max_pos_delta, max_pos_delta
                )
                * scale
            )

            # Add delta to previous target
            if self.current_targets is not None:
                prev_target = self.current_targets[:, dof_idx]
            else:
                # First step: use current DOF position as base
                prev_target = self.dof_pos[:, dof_idx]
            return prev_target + scaled_delta
        else:
            raise ValueError(f"Unknown control mode: {self.action_control_mode}")

    def _apply_default_finger_targets(
        self, targets, task_targets, joint_to_control, active_joint_names
    ):
        """
        Apply default or task-specific finger targets when not controlled by policy.

        Args:
            targets: Target tensor to update
            task_targets: Optional task-specific targets
            joint_to_control: Mapping from joint names to control names
            active_joint_names: List of active joint names
        """
        if task_targets is not None and "finger_targets" in task_targets:
            for i, name in enumerate(self.dof_names[self.NUM_BASE_DOFS :]):
                # Skip if not in joint_to_control mapping
                if name not in joint_to_control:
                    continue

                # Map from joint name to control index
                try:
                    control_name = joint_to_control[name]
                    try:
                        control_idx = active_joint_names.index(control_name)

                        finger_dof_idx = i + self.NUM_BASE_DOFS
                        targets[:, finger_dof_idx] = task_targets["finger_targets"][
                            :, control_idx
                        ]
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Error mapping task targets for {name}: {e}")
                except KeyError:
                    logger.warning(
                        f"Joint {name} not found in joint_to_control mapping"
                    )
        else:
            # Use stored DOF names if available
            if not self.dof_names:
                logger.warning(
                    "DOF names not available from asset. Cannot set default finger targets."
                )
                return

            for i, name in enumerate(self.dof_names[self.NUM_BASE_DOFS :]):
                # Skip if not in joint_to_control mapping
                if name not in joint_to_control:
                    continue

                # Map from joint name to control index
                try:
                    control_name = joint_to_control[name]
                    try:
                        control_idx = active_joint_names.index(control_name)

                        finger_dof_idx = i + self.NUM_BASE_DOFS
                        if control_idx < len(self.default_finger_targets):
                            targets[:, finger_dof_idx] = self.default_finger_targets[
                                control_idx
                            ]
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Error setting default target for {name}: {e}")
                except KeyError:
                    logger.warning(
                        f"Joint {name} not found in joint_to_control mapping"
                    )

    def _apply_pd_control(self, targets):
        """
        Apply PD control targets to the simulation.

        Args:
            targets: Target tensor with DOF positions

        Returns:
            bool: Success flag
        """
        try:
            # Make sure DOF limits have the right shape
            if (
                self.dof_lower_limits is None
                or self.dof_lower_limits.shape[0] != self.num_dof
            ):
                logger.debug(
                    f"Resizing DOF limits from {0 if self.dof_lower_limits is None else self.dof_lower_limits.shape[0]} to {self.num_dof}"
                )
                self.dof_lower_limits = torch.full(
                    (self.num_dof,), -1.0, device=self.device
                )
                self.dof_upper_limits = torch.full(
                    (self.num_dof,), 1.0, device=self.device
                )

            # Use the correctly processed targets from the finger coupling logic above
            # instead of creating a new tensor that overwrites the scaled values
            direct_targets = targets.clone()

            # CRITICAL: Directly set the base DOF targets from prev_active_targets
            # This skips all the target tensor transformations that might zero things out
            if self.policy_controls_hand_base:
                num_dofs_to_copy = min(
                    self.NUM_BASE_DOFS,
                    direct_targets.shape[1],
                    self.prev_active_targets.shape[1],
                )
                direct_targets[:, :num_dofs_to_copy] = self.prev_active_targets[
                    :, :num_dofs_to_copy
                ]

            # Note: Finger targets are already correctly processed above with action scaling
            # No need to reprocess them here as that would overwrite the scaled values

            # Clamp targets to joint limits
            direct_targets = torch.clamp(
                direct_targets,
                self.dof_lower_limits.unsqueeze(0),
                self.dof_upper_limits.unsqueeze(0),
            )

            # Store targets
            self.current_targets = direct_targets.clone()

            # Set PD control targets
            self.gym.set_dof_position_target_tensor(
                self.sim, gymtorch.unwrap_tensor(self.current_targets)
            )

            # Debug: Check if any targets are non-zero AND print actual DOF positions
            non_zero_targets = torch.nonzero(self.current_targets[0])
            if len(non_zero_targets) > 0:
                logger.debug(f"Setting {len(non_zero_targets)} non-zero targets:")
                for idx in non_zero_targets[:3]:  # Show first 3
                    dof_idx = idx.item()
                    target_val = self.current_targets[0, dof_idx].item()
                    actual_pos = self.dof_pos[0, dof_idx].item()
                    joint_name = (
                        self.dof_names[dof_idx]
                        if dof_idx < len(self.dof_names)
                        else f"DOF_{dof_idx}"
                    )
                    logger.debug(
                        f"  {joint_name} (DOF {dof_idx}): target = {target_val:.6f}, actual = {actual_pos:.6f}"
                    )
            else:
                logger.debug("All targets are zero")

            return True

        except Exception as e:
            logger.error(f"Error setting DOF targets: {e}")
            logger.exception("Traceback:")
            return False

    def _compute_action_scaling_coeffs(self):
        """
        Compute scaling coefficients for converting actions from [-1,+1] to physical units.
        This allows for efficient unscaling later.
        """
        # Determine total action space size
        total_actions = 0
        if self.policy_controls_hand_base:
            total_actions += self.NUM_BASE_DOFS
        if self.policy_controls_fingers:
            total_actions += self.NUM_ACTIVE_FINGER_DOFS

        if total_actions == 0:
            self.action_space_scale = torch.zeros((0,), device=self.device)
            self.action_space_bias = torch.zeros((0,), device=self.device)
            return

        # For mapping normalized actions [-1, 1] to physical units [min, max]:
        # physical = (action + 1) * action_space_scale + action_space_bias
        # where action_space_scale = 0.5 * (max - min) and action_space_bias = min
        self.action_space_scale = torch.zeros(total_actions, device=self.device)
        self.action_space_bias = torch.zeros(total_actions, device=self.device)
        action_idx = 0

        # Base DOF scaling coefficients
        if self.policy_controls_hand_base:
            action_idx = self._compute_base_scaling_coeffs(action_idx)

        # Finger DOF scaling coefficients
        if self.policy_controls_fingers:
            action_idx = self._compute_finger_scaling_coeffs(action_idx)

    def _compute_base_scaling_coeffs(self, action_idx):
        """
        Compute scaling coefficients for base DOFs.

        Args:
            action_idx: Current action index

        Returns:
            Updated action index
        """
        for i in range(self.NUM_BASE_DOFS):
            if self.action_control_mode == "position":
                # Position mode: map to DOF limits
                min_val = self.dof_lower_limits[i]
                max_val = self.dof_upper_limits[i]
            else:
                # Position delta mode: map to velocity limits
                if i < 3:  # Linear DOFs
                    max_delta = self.control_dt * self.policy_base_lin_velocity_limit
                else:  # Angular DOFs
                    max_delta = self.control_dt * self.policy_base_ang_velocity_limit
                min_val = -max_delta
                max_val = max_delta

            self.action_space_scale[action_idx] = 0.5 * (max_val - min_val)
            self.action_space_bias[action_idx] = min_val
            action_idx += 1
        return action_idx

    def _compute_finger_scaling_coeffs(self, action_idx):
        """
        Compute scaling coefficients for finger DOFs.

        Args:
            action_idx: Current action index

        Returns:
            Updated action index
        """
        for finger_action_idx in range(self.NUM_ACTIVE_FINGER_DOFS):
            # Get primary joint and DOF coupling scale
            joint_name, coupling_scale = self._get_finger_action_primary_joint(
                finger_action_idx
            )

            if self.action_control_mode == "position":
                # Find DOF index to get limits
                dof_idx = self._find_dof_index(joint_name)
                if dof_idx < 0:
                    raise RuntimeError(
                        f"Failed to find DOF index for joint '{joint_name}' in finger action {finger_action_idx}"
                    )

                # Apply DOF coupling scale to the physical limits
                min_val = self.dof_lower_limits[dof_idx] * coupling_scale
                max_val = self.dof_upper_limits[dof_idx] * coupling_scale
            else:
                # position_delta mode - apply DOF coupling scale to velocity limits
                max_delta = (
                    self.control_dt * self.policy_finger_velocity_limit * coupling_scale
                )
                min_val = -max_delta
                max_val = max_delta

            self.action_space_scale[action_idx] = 0.5 * (max_val - min_val)
            self.action_space_bias[action_idx] = min_val
            action_idx += 1
        return action_idx

    def unscale_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Convert actions from normalized space [-1, +1] to physical units.

        Uses the unscaling function assigned during initialization based on control mode.

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

        # Use the pre-assigned unscaling function
        return self._unscale_actions_fn(actions)

    def _get_finger_action_primary_joint(self, finger_action_idx):
        """
        Get the primary joint name and DOF coupling scale for a finger action.

        Args:
            finger_action_idx: Index of the finger action

        Returns:
            Tuple of (joint_name, coupling_scale)
        """
        if finger_action_idx not in self.finger_coupling_map:
            raise RuntimeError(
                f"Finger action index {finger_action_idx} not found in coupling map"
            )

        joint_mapping = self.finger_coupling_map[finger_action_idx]
        first_joint_spec = joint_mapping[0]
        return self._parse_joint_spec(first_joint_spec)

    def _find_dof_index(self, joint_name: str, start_offset: int = 0) -> int:
        """
        Find the DOF index for a given joint name.

        Args:
            joint_name: Name of the joint to find
            start_offset: Offset to add to the found index

        Returns:
            DOF index if found, -1 otherwise
        """
        search_range = (
            self.dof_names[start_offset:] if start_offset > 0 else self.dof_names
        )

        for i, name in enumerate(search_range):
            if name == joint_name:
                return i + start_offset

        return -1

    def _get_control_mode_limits(self, dof_indices, is_base=True):
        """
        Get position/velocity limits based on control mode.

        For position mode: returns DOF position limits
        For position_delta mode: returns velocity limits converted to position deltas

        Args:
            dof_indices: DOF indices or count
            is_base: Whether this is for base DOFs (affects velocity limits)

        Returns:
            Tuple of (min_limits, max_limits) tensors
        """
        if self.action_control_mode == "position":
            # Use DOF position limits
            if isinstance(dof_indices, int):
                # It's a count - get first N DOFs
                num_dofs = dof_indices
                return (
                    self.dof_lower_limits[:num_dofs],
                    self.dof_upper_limits[:num_dofs],
                )
            else:
                # It's actual indices
                return (
                    self.dof_lower_limits[dof_indices],
                    self.dof_upper_limits[dof_indices],
                )

        elif self.action_control_mode == "position_delta":
            # Use velocity limits converted to position deltas
            if isinstance(dof_indices, int):
                num_dofs = dof_indices
            else:
                num_dofs = len(dof_indices)

            velocity_limits = torch.zeros(num_dofs, device=self.device)

            if is_base:
                # Base DOFs have different limits for linear vs angular
                velocity_limits[:3] = (
                    self.control_dt * self.policy_base_lin_velocity_limit
                )
                if num_dofs > 3:
                    velocity_limits[3:] = (
                        self.control_dt * self.policy_base_ang_velocity_limit
                    )
            else:
                # Finger DOFs all use same limit
                velocity_limits[:] = self.control_dt * self.policy_finger_velocity_limit

            return -velocity_limits, velocity_limits
        else:
            raise ValueError(f"Unknown control mode: {self.action_control_mode}")

    def _parse_joint_spec(self, joint_spec):
        """
        Parse a joint specification from the coupling map.

        Args:
            joint_spec: Either a string (joint name) or tuple (joint_name, coupling_scale)

        Returns:
            Tuple of (joint_name, coupling_scale)
        """
        if isinstance(joint_spec, str):
            return joint_spec, 1.0
        elif isinstance(joint_spec, tuple):
            return joint_spec
        else:
            raise ValueError(f"Invalid joint spec type: {type(joint_spec)}")

    def _unscale_actions_position(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Unscale actions for position control mode.
        Maps from normalized space [-1, 1] to physical units [dof_min, dof_max] for each DOF.

        Formula: physical = (normalized + 1) * action_space_scale + action_space_bias
        where action_space_scale = 0.5 * (max - min) and action_space_bias = min
        """
        return (actions + 1.0) * self.action_space_scale + self.action_space_bias

    def _unscale_actions_position_delta(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Unscale actions for position_delta control mode.
        Maps from normalized space [-1, 1] to physical velocity deltas [-max_velocity_delta, +max_velocity_delta].

        Formula: physical_delta = (normalized + 1) * action_space_scale + action_space_bias
        For symmetric limits: action_space_scale = max_delta, action_space_bias = -max_delta
        """
        return (actions + 1.0) * self.action_space_scale + self.action_space_bias
