"""
Action processor component for DexHand environment.

This module provides action processing functionality for the DexHand environment,
including action scaling, mapping, and PD control.
"""

# Import standard libraries
import torch
import numpy as np

# Import IsaacGym
from isaacgym import gymapi, gymtorch


class ActionProcessor:
    """
    Processes actions for the DexHand environment.

    This component provides functionality to:
    - Map policy actions to robot DOFs
    - Apply action scaling and transformations
    - Handle different control modes (position, position_delta)
    - Manage PD control targets
    """

    def __init__(self, gym, sim, num_envs, device, dof_props=None, hand_asset=None):
        """
        Initialize the action processor.

        Args:
            gym: The isaacgym gym instance
            sim: The isaacgym simulation instance
            num_envs: Number of environments
            device: PyTorch device
            dof_props: DOF properties tensor (optional)
            hand_asset: Hand asset for getting DOF names (optional)
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
            print(f"ActionProcessor initialized with {len(self.dof_names)} DOF names from asset")

        # Control settings
        self.action_control_mode = "position"  # position or position_delta
        self.policy_controls_hand_base = True
        self.policy_controls_fingers = True

        # Constants for action dimensions
        self.NUM_BASE_DOFS = 6
        self.NUM_ACTIVE_FINGER_DOFS = 12  # 12 finger controls mapping to 19 DOFs with coupling

        # Finger DOF coupling mapping (12 actions -> 19 DOFs)
        # Actions map to finger DOFs as follows:
        # 0: r_f_joint1_1 (thumb spread)
        # 1: r_f_joint1_2 (thumb MCP)
        # 2: r_f_joint1_3, r_f_joint1_4 (thumb DIP - coupled)
        # 3: r_f_joint2_1, r_f_joint4_1, r_f_joint5_1 (finger spread - coupled, 5_1 is 2x)
        # 4: r_f_joint2_2 (index MCP)
        # 5: r_f_joint2_3, r_f_joint2_4 (index DIP - coupled)
        # 6: r_f_joint3_2 (middle MCP)
        # 7: r_f_joint3_3, r_f_joint3_4 (middle DIP - coupled)
        # 8: r_f_joint4_2 (ring MCP)
        # 9: r_f_joint4_3, r_f_joint4_4 (ring DIP - coupled)
        # 10: r_f_joint5_2 (pinky MCP)
        # 11: r_f_joint5_3, r_f_joint5_4 (pinky DIP - coupled)
        # Note: r_f_joint3_1 is fixed at 0 (not controlled)
        self.finger_coupling_map = {
            0: ["r_f_joint1_1"],  # thumb spread
            1: ["r_f_joint1_2"],  # thumb MCP
            2: ["r_f_joint1_3", "r_f_joint1_4"],  # thumb DIP (coupled)
            3: [("r_f_joint2_1", 1.0), ("r_f_joint4_1", 1.0), ("r_f_joint5_1", 2.0)],  # finger spread (5_1 is 2x)
            4: ["r_f_joint2_2"],  # index MCP
            5: ["r_f_joint2_3", "r_f_joint2_4"],  # index DIP (coupled)
            6: ["r_f_joint3_2"],  # middle MCP
            7: ["r_f_joint3_3", "r_f_joint3_4"],  # middle DIP (coupled)
            8: ["r_f_joint4_2"],  # ring MCP
            9: ["r_f_joint4_3", "r_f_joint4_4"],  # ring DIP (coupled)
            10: ["r_f_joint5_2"],  # pinky MCP
            11: ["r_f_joint5_3", "r_f_joint5_4"]  # pinky DIP (coupled)
        }

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

        # Control timestep for position_delta mode (must be set before processing actions)
        self.control_dt = None

        # Initialize with empty tensors - will be properly initialized later
        self.dof_pos = None
        self.prev_active_targets = None
        self.current_targets = None
        self.actions = None

    def setup(self, num_dof, dof_props=None):
        """
        Set up action processor with DOF information.

        Args:
            num_dof: Number of DOFs in the model
            dof_props: DOF properties tensor (optional)
        """
        self.num_dof = num_dof

        # Initialize previous targets tensor
        self.prev_active_targets = torch.zeros(
            (self.num_envs, self.NUM_BASE_DOFS + self.NUM_ACTIVE_FINGER_DOFS),
            device=self.device, dtype=torch.float
        )

        # Initialize with default values
        # Base position targets (RELATIVE motion from spawn point)
        self.prev_active_targets[:, 0] = 0.0  # ARTx - relative X displacement
        self.prev_active_targets[:, 1] = 0.0  # ARTy - relative Y displacement
        self.prev_active_targets[:, 2] = 0.0  # ARTz - relative Z displacement (0 = stay at spawn height)

        # Initialize rotation targets - default is identity quaternion (0,0,0 in axis-angle)
        self.prev_active_targets[:, 3:6] = 0.0  # ARRx, ARRy, ARRz

        # Create current targets tensor
        self.current_targets = torch.zeros((self.num_envs, self.num_dof), device=self.device)

        # Initialize DOF limits
        if dof_props is not None:
            self.dof_props = dof_props

            # Check if it's a tensor (from TensorManager) or a dictionary
            if isinstance(dof_props, torch.Tensor):
                print(f"DOF props is a tensor with shape: {dof_props.shape}")
                # Format is [stiffness, damping, friction, armature, min, max]
                # Extract limits from the tensor (indices 4 and 5 are min and max)
                self.dof_lower_limits = dof_props[:, 4].clone().to(device=self.device)
                self.dof_upper_limits = dof_props[:, 5].clone().to(device=self.device)
            elif isinstance(dof_props, dict) and 'lower' in dof_props and 'upper' in dof_props:
                # Extract DOF limits from dictionary
                self.dof_lower_limits = torch.tensor(dof_props['lower'], dtype=torch.float, device=self.device)
                self.dof_upper_limits = torch.tensor(dof_props['upper'], dtype=torch.float, device=self.device)
            else:
                print("Warning: DOF properties format not recognized, using default limits")
                # Default limits
                self.dof_lower_limits = torch.full((self.num_dof,), -1.0, device=self.device)
                self.dof_upper_limits = torch.full((self.num_dof,), 1.0, device=self.device)
        else:
            # Default limits
            self.dof_lower_limits = torch.full((self.num_dof,), -1.0, device=self.device)
            self.dof_upper_limits = torch.full((self.num_dof,), 1.0, device=self.device)

    def set_control_mode(self, mode):
        """
        Set the control mode for action processing.

        Args:
            mode: Control mode string ("position" or "position_delta")
        """
        valid_modes = ["position", "position_delta"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid control mode: {mode}. Must be one of {valid_modes}")

        self.action_control_mode = mode

    def set_control_options(self, policy_controls_hand_base=None, policy_controls_fingers=None):
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
            raise ValueError("At least one of policy_controls_hand_base or policy_controls_fingers must be True")

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

    def set_velocity_limits(self, finger_vel_limit=None, base_lin_vel_limit=None, base_ang_vel_limit=None):
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

    def set_control_dt(self, control_dt):
        """
        Set the control timestep for position_delta mode.

        Args:
            control_dt: Control timestep in seconds
        """
        if control_dt is None or control_dt <= 0:
            raise ValueError(f"Invalid control_dt: {control_dt}. Must be positive.")
        self.control_dt = control_dt

    def process_actions(self, actions, dof_pos, joint_to_control=None, active_joint_names=None, task_targets=None):
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
            if actions is None:
                print("Warning: Actions is None, skipping action processing")
                return False

            # Validate control_dt is set for position_delta mode
            if self.action_control_mode == "position_delta" and self.control_dt is None:
                raise RuntimeError("control_dt must be set before processing actions in position_delta mode. Call set_control_dt() first.")

            # Store actions for reference
            self.actions = actions.clone()
            self.dof_pos = dof_pos

            # Validate joint mappings
            if (self.policy_controls_fingers and
                (joint_to_control is None or active_joint_names is None)):
                print("Error: joint_to_control and active_joint_names must be provided when controlling fingers")
                return False

            # Fail fast on invalid tensor shapes
            if self.dof_pos.shape[1] == 0:
                print(f"Error: DOF position tensor has invalid shape {self.dof_pos.shape}")
                return False

            # Initialize targets with current positions
            targets = self.dof_pos.clone()

            # Split actions into base and finger components
            action_idx = 0

            # Apply base actions
            if self.policy_controls_hand_base:
                # Extract base actions
                if self.actions.shape[1] > 0 and self.NUM_BASE_DOFS > 0:
                    base_actions = self.actions[:, :min(self.NUM_BASE_DOFS, self.actions.shape[1])]

                    # Base targets from actions (position control)
                    base_pos_targets = None

                    # Scale base actions based on control mode
                    scaled_base_actions = torch.zeros_like(base_actions)
                    for i in range(min(self.NUM_BASE_DOFS, base_actions.shape[1])):
                        if self.action_control_mode == "position":
                            # Position mode: map action from [-1, +1] to [dof_min, dof_max]
                            dof_min = self.dof_lower_limits[i]
                            dof_max = self.dof_upper_limits[i]
                            scaled_base_actions[:, i] = (base_actions[:, i] + 1.0) * 0.5 * (dof_max - dof_min) + dof_min
                        elif self.action_control_mode == "position_delta":
                            # Position delta mode: map action from [-1, +1] to [-max_pos_delta, +max_pos_delta]
                            # where max_pos_delta = control_dt * max_velocity
                            if i < 3:  # Translation DOFs (x, y, z)
                                max_pos_delta = self.control_dt * self.policy_base_lin_velocity_limit
                            else:  # Rotation DOFs (rx, ry, rz)
                                max_pos_delta = self.control_dt * self.policy_base_ang_velocity_limit
                            scaled_base_actions[:, i] = base_actions[:, i] * max_pos_delta

                    if self.action_control_mode == "position":
                        # Direct position targets
                        base_pos_targets = scaled_base_actions
                    elif self.action_control_mode == "position_delta":
                        # Incremental position changes
                        base_pos_targets = self.prev_active_targets[:, :self.NUM_BASE_DOFS] + scaled_base_actions
                    else:
                        raise ValueError(f"Unknown control mode: {self.action_control_mode}")

                    # Apply targets to the base DOFs
                    if targets[:, :self.NUM_BASE_DOFS].shape == base_pos_targets.shape:
                        # Assign to targets tensor
                        targets[:, :self.NUM_BASE_DOFS] = base_pos_targets

                        # Update previous targets
                        self.prev_active_targets[:, :self.NUM_BASE_DOFS] = base_pos_targets
                    else:
                        raise RuntimeError(f"Cannot assign base_pos_targets {base_pos_targets.shape} to targets[:, :self.NUM_BASE_DOFS] with shape {targets[:, :self.NUM_BASE_DOFS].shape}")

                    # Update action index
                    action_idx += self.NUM_BASE_DOFS
                else:
                    # Get targets from task or use defaults
                    if task_targets is not None and 'base_targets' in task_targets:
                        targets[:, :self.NUM_BASE_DOFS] = task_targets['base_targets']
                    else:
                        # Expand default_base_targets to match batch dimension
                        if len(self.default_base_targets.shape) == 1:
                            expanded_targets = self.default_base_targets.unsqueeze(0).expand(self.num_envs, -1)
                        else:
                            expanded_targets = self.default_base_targets
                        targets[:, :self.NUM_BASE_DOFS] = expanded_targets
            else:
                # Hand base not controlled by policy - use task targets or defaults
                if task_targets is not None and 'base_targets' in task_targets:
                    if targets[:, :self.NUM_BASE_DOFS].shape == task_targets['base_targets'].shape:
                        targets[:, :self.NUM_BASE_DOFS] = task_targets['base_targets']
                    else:
                        print(f"Error: Cannot assign task_targets['base_targets'] to targets[:, :self.NUM_BASE_DOFS]")
                else:
                    # Expand default_base_targets to match batch dimension
                    if len(self.default_base_targets.shape) == 1:
                        expanded_targets = self.default_base_targets.unsqueeze(0).expand(self.num_envs, -1)
                    else:
                        expanded_targets = self.default_base_targets

                    if targets[:, :self.NUM_BASE_DOFS].shape == expanded_targets.shape:
                        targets[:, :self.NUM_BASE_DOFS] = expanded_targets
                    else:
                        print(f"Error: Cannot assign default_base_targets to targets[:, :self.NUM_BASE_DOFS]")

            # Apply finger actions
            if self.policy_controls_fingers:
                # Extract finger actions
                if self.actions.shape[1] > action_idx and self.NUM_ACTIVE_FINGER_DOFS > 0:
                    finger_end_idx = min(action_idx + self.NUM_ACTIVE_FINGER_DOFS, self.actions.shape[1])
                    finger_actions = self.actions[:, action_idx:finger_end_idx]

                    # Finger targets from actions
                    finger_pos_targets = None

                    if self.action_control_mode == "position":
                        # Direct position targets
                        finger_pos_targets = finger_actions
                    elif self.action_control_mode == "position_delta":
                        # Incremental position changes
                        finger_pos_targets = self.prev_active_targets[:, self.NUM_BASE_DOFS:] + finger_actions
                    else:
                        print(f"Unknown control mode: {self.action_control_mode}")

                    # Apply targets to the finger DOFs using coupling logic
                    if finger_pos_targets is not None and finger_pos_targets.shape[1] >= 12:
                        # Create DOF name to index mapping
                        dof_name_to_idx = {}
                        for i, name in enumerate(self.dof_names):
                            dof_name_to_idx[name] = i

                        # Apply coupling mapping
                        for action_idx, joint_mapping in self.finger_coupling_map.items():
                            if action_idx >= finger_actions.shape[1]:
                                continue

                            # Use the original action value for scaling, not accumulated targets
                            raw_action_value = finger_actions[:, action_idx]

                            # Handle different mapping types
                            for joint_spec in joint_mapping:
                                if isinstance(joint_spec, str):
                                    # Simple 1:1 mapping
                                    joint_name = joint_spec
                                    scale = 1.0
                                elif isinstance(joint_spec, tuple):
                                    # Joint with scaling factor
                                    joint_name, scale = joint_spec
                                else:
                                    continue

                                # Apply to target if joint exists
                                if joint_name in dof_name_to_idx:
                                    dof_idx = dof_name_to_idx[joint_name]

                                   
                                    # Scale action based on control mode
                                    if self.action_control_mode == "position":
                                        # Position mode: map action from [-1, +1] to [dof_min, dof_max]
                                        if self.dof_lower_limits is not None and self.dof_upper_limits is not None:
                                            dof_min = self.dof_lower_limits[dof_idx]
                                            dof_max = self.dof_upper_limits[dof_idx]
                                            scaled_action = (raw_action_value + 1.0) * 0.5 * (dof_max - dof_min) + dof_min
                                            final_target = scaled_action * scale
                                            targets[:, dof_idx] = final_target
                                        else:
                                            # Fallback: use raw action value
                                            targets[:, dof_idx] = raw_action_value * scale
                                    elif self.action_control_mode == "position_delta":
                                        # Position delta mode: map action from [-1, +1] to [-max_pos_delta, +max_pos_delta]
                                        # where max_pos_delta = control_dt * max_finger_velocity
                                        max_pos_delta = self.control_dt * self.policy_finger_velocity_limit
                                        scaled_delta = raw_action_value * max_pos_delta * scale
                                        
                                        # CRITICAL: Use previous target + delta, NOT current DOF position + delta
                                        # This ensures that targets accumulate properly and stationary errors don't reset targets
                                        if self.current_targets is not None:
                                            prev_target = self.current_targets[:, dof_idx]
                                        else:
                                            # First step: use current DOF position as base
                                            prev_target = self.dof_pos[:, dof_idx]
                                        targets[:, dof_idx] = prev_target + scaled_delta

                                    else:
                                        raise ValueError(f"Unknown control mode: {self.action_control_mode}")

                        # Fix r_f_joint3_1 to 0 (middle finger spread)
                        if "r_f_joint3_1" in dof_name_to_idx:
                            dof_idx = dof_name_to_idx["r_f_joint3_1"]
                            targets[:, dof_idx] = 0.0

                        # Update previous targets
                        self.prev_active_targets[:, self.NUM_BASE_DOFS:] = finger_pos_targets
                else:
                    # Get targets from task or use defaults
                    if task_targets is not None and 'finger_targets' in task_targets:
                        for i, name in enumerate(self.dof_names[self.NUM_BASE_DOFS:]):
                            # Skip if not in joint_to_control mapping
                            if name not in joint_to_control:
                                continue

                            # Map from joint name to control index
                            try:
                                control_name = joint_to_control[name]
                                try:
                                    control_idx = active_joint_names.index(control_name)

                                    finger_dof_idx = i + self.NUM_BASE_DOFS
                                    targets[:, finger_dof_idx] = task_targets['finger_targets'][:, control_idx]
                                except (ValueError, IndexError) as e:
                                    print(f"Warning: Error mapping task targets for {name}: {e}")
                            except KeyError:
                                print(f"Warning: Joint {name} not found in joint_to_control mapping")
                    else:
                        # Use stored DOF names if available
                        if not self.dof_names:
                            print("Warning: DOF names not available from asset. Cannot set default finger targets.")
                            return False

                        for i, name in enumerate(self.dof_names[self.NUM_BASE_DOFS:]):
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
                                        targets[:, finger_dof_idx] = self.default_finger_targets[control_idx]
                                except (ValueError, IndexError) as e:
                                    print(f"Warning: Error setting default target for {name}: {e}")
                            except KeyError:
                                print(f"Warning: Joint {name} not found in joint_to_control mapping")

            # Apply target positions with PD control
            try:
                # Make sure DOF limits have the right shape
                if self.dof_lower_limits is None or self.dof_lower_limits.shape[0] != self.num_dof:
                    print(f"Resizing DOF limits from {0 if self.dof_lower_limits is None else self.dof_lower_limits.shape[0]} to {self.num_dof}")
                    self.dof_lower_limits = torch.full((self.num_dof,), -1.0, device=self.device)
                    self.dof_upper_limits = torch.full((self.num_dof,), 1.0, device=self.device)

                # Use the correctly processed targets from the finger coupling logic above
                # instead of creating a new tensor that overwrites the scaled values
                direct_targets = targets.clone()

                # CRITICAL: Directly set the base DOF targets from prev_active_targets
                # This skips all the target tensor transformations that might zero things out
                if self.policy_controls_hand_base:
                    for env_idx in range(self.num_envs):
                        for dof_idx in range(min(self.NUM_BASE_DOFS, direct_targets.shape[1])):
                            if dof_idx < self.prev_active_targets.shape[1]:
                                # Just copy directly from prev_active_targets which is known good
                                direct_targets[env_idx, dof_idx] = self.prev_active_targets[env_idx, dof_idx]

                # Note: Finger targets are already correctly processed above with action scaling
                # No need to reprocess them here as that would overwrite the scaled values

                # Clamp targets to joint limits
                direct_targets = torch.clamp(
                    direct_targets,
                    self.dof_lower_limits.unsqueeze(0),
                    self.dof_upper_limits.unsqueeze(0)
                )

                # Store targets
                self.current_targets = direct_targets.clone()

                # Set PD control targets
                self.gym.set_dof_position_target_tensor(
                    self.sim,
                    gymtorch.unwrap_tensor(self.current_targets)
                )

                # Debug: Check if any targets are non-zero AND print actual DOF positions
                non_zero_targets = torch.nonzero(self.current_targets[0])
                if len(non_zero_targets) > 0:
                    print(f"DEBUG: Setting {len(non_zero_targets)} non-zero targets:")
                    for idx in non_zero_targets[:3]:  # Show first 3
                        dof_idx = idx.item()
                        target_val = self.current_targets[0, dof_idx].item()
                        actual_pos = self.dof_pos[0, dof_idx].item()
                        joint_name = self.dof_names[dof_idx] if dof_idx < len(self.dof_names) else f"DOF_{dof_idx}"
                        print(f"  {joint_name} (DOF {dof_idx}): target = {target_val:.6f}, actual = {actual_pos:.6f}")
                else:
                    print("DEBUG: All targets are zero")

                return True

            except Exception as e:
                print(f"Error setting DOF targets: {e}")
                import traceback
                traceback.print_exc()
                return False

        except Exception as e:
            print(f"Error in process_actions: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _apply_raw_finger_targets(self, finger_targets, dof_pos):
        """
        Apply raw finger targets (physically meaningful values) with proper coupling.

        Args:
            finger_targets: torch.Tensor of shape (num_envs, NUM_ACTIVE_FINGER_DOFS) with raw joint targets in radians
            dof_pos: current DOF positions tensor
        """
        if finger_targets.shape[1] != self.NUM_ACTIVE_FINGER_DOFS:
            print(f"Warning: Expected {self.NUM_ACTIVE_FINGER_DOFS} finger targets, got {finger_targets.shape[1]}")
            return

        # Create DOF name to index mapping
        dof_name_to_idx = {}
        for i, name in enumerate(self.dof_names):
            dof_name_to_idx[name] = i

        # Apply coupling mapping with raw values (no scaling needed)
        for action_idx, joint_mapping in self.finger_coupling_map.items():
            if action_idx >= finger_targets.shape[1]:
                continue

            # Use the raw finger target value
            raw_target_value = finger_targets[:, action_idx]

            # Handle different mapping types
            for joint_spec in joint_mapping:
                if isinstance(joint_spec, str):
                    # Simple 1:1 mapping
                    joint_name = joint_spec
                    scale = 1.0
                elif isinstance(joint_spec, tuple):
                    # Joint with scaling factor
                    joint_name, scale = joint_spec
                else:
                    continue

                # Apply to target if joint exists
                if joint_name in dof_name_to_idx:
                    dof_idx = dof_name_to_idx[joint_name]

                    # Apply raw target with scaling factor
                    final_target = raw_target_value * scale
                    self.current_targets[:, dof_idx] = final_target

        # Fix r_f_joint3_1 to 0 (middle finger spread)
        if "r_f_joint3_1" in dof_name_to_idx:
            dof_idx = dof_name_to_idx["r_f_joint3_1"]
            self.current_targets[:, dof_idx] = 0.0
