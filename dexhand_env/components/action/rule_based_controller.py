"""
Rule-based controller for DexHand environment.

This module provides functionality for applying manual control rules
to hand components not controlled by the policy.
"""

import torch
from loguru import logger
from isaacgym import gymtorch


class RuleBasedController:
    """
    Manages rule-based control for hand parts not controlled by the policy.

    Allows registration of custom controller functions for base and finger DOFs
    and applies them during physics steps when those parts are not policy-controlled.
    """

    def __init__(self, parent):
        """Initialize the rule-based controller."""
        self.parent = parent

        # Initialize rule-based controller functions
        self.rule_based_base_controller = None
        self.rule_based_finger_controller = None

    def set_controllers(self, base_controller=None, finger_controller=None):
        """
        Set rule-based control functions for hand parts not controlled by the policy.

        Control functions should have the signature:
            def controller(env) -> torch.Tensor

        Where:
            - env: The environment instance (self), providing access to all properties
            - Returns: torch.Tensor of appropriate shape with target values in physical units

        Args:
            base_controller: Callable that returns (num_envs, 6) tensor with base DOF targets
                           in physical units (meters for translation, radians for rotation).
                           Only used if control_hand_base is False.
            finger_controller: Callable that returns (num_envs, 12) tensor with finger targets
                             in physical units (radians).
                             Only used if control_fingers is False.

        Example:
            def my_base_controller(env):
                t = env.episode_step_count[0] * env.dt  # Get simulation time
                targets = torch.zeros((env.num_envs, 6), device=env.device)
                targets[:, 0] = 0.1 * torch.sin(t)  # Oscillate in X
                return targets

            rule_controller.set_controllers(base_controller=my_base_controller)
        """
        self.rule_based_base_controller = base_controller
        self.rule_based_finger_controller = finger_controller

        # Validate controllers
        if base_controller is not None and self.parent.policy_controls_hand_base:
            logger.warning(
                "Base controller provided but policy_controls_hand_base=True. Controller will be ignored."
            )
        if finger_controller is not None and self.parent.policy_controls_fingers:
            logger.warning(
                "Finger controller provided but policy_controls_fingers=True. Controller will be ignored."
            )

    def apply_rule_based_control(self):
        """
        Apply rule-based control using registered controller functions.
        Called automatically during pre_physics_step.
        """
        # action_processor and dof_pos must be initialized by this point
        # If they're not, that indicates an initialization bug

        # Apply base controller if available and base is not policy-controlled
        if (
            not self.parent.policy_controls_hand_base
            and self.rule_based_base_controller is not None
        ):
            try:
                base_targets = self.rule_based_base_controller(self.parent)
                if base_targets.shape == (
                    self.parent.num_envs,
                    self.parent.action_processor.NUM_BASE_DOFS,
                ):
                    # Directly set base DOF targets (raw physical values)
                    self.parent.action_processor.current_targets[
                        :, 0 : self.parent.action_processor.NUM_BASE_DOFS
                    ] = base_targets
                else:
                    logger.error(
                        f"Base controller returned shape {base_targets.shape}, expected ({self.parent.num_envs}, {self.parent.action_processor.NUM_BASE_DOFS})"
                    )
            except Exception as e:
                logger.error(f"Error in base controller: {e}")
                import traceback

                traceback.print_exc()

        # Apply finger controller if available and fingers are not policy-controlled
        if (
            not self.parent.policy_controls_fingers
            and self.rule_based_finger_controller is not None
        ):
            try:
                finger_targets = self.rule_based_finger_controller(self.parent)
                if finger_targets.shape == (
                    self.parent.num_envs,
                    self.parent.action_processor.NUM_ACTIVE_FINGER_DOFS,
                ):
                    # Apply finger coupling by creating full active targets
                    active_targets = torch.zeros(
                        (self.parent.num_envs, 18),
                        device=self.parent.device,  # 6 base + 12 fingers
                    )
                    # Copy current base targets
                    active_targets[
                        :, :6
                    ] = self.parent.action_processor.current_targets[:, :6]
                    # Set finger targets
                    active_targets[:, 6:] = finger_targets

                    # Apply coupling to get full DOF targets and overwrite current_targets
                    self.parent.action_processor.current_targets = (
                        self.parent.action_processor.apply_coupling(active_targets)
                    )
                else:
                    logger.error(
                        f"Finger controller returned shape {finger_targets.shape}, expected ({self.parent.num_envs}, {self.parent.action_processor.NUM_ACTIVE_FINGER_DOFS})"
                    )
            except Exception as e:
                logger.error(f"Error in finger controller: {e}")
                import traceback

                traceback.print_exc()

        # Only apply targets if we actually modified them via rule-based control
        # If only policy controls both base and fingers, don't call set_dof_position_target_tensor
        # as it was already called by process_actions()
        if (
            not self.parent.policy_controls_hand_base
            and self.rule_based_base_controller is not None
        ) or (
            not self.parent.policy_controls_fingers
            and self.rule_based_finger_controller is not None
        ):
            try:
                self.parent.gym.set_dof_position_target_tensor(
                    self.parent.sim,
                    gymtorch.unwrap_tensor(
                        self.parent.action_processor.current_targets
                    ),
                )
            except Exception as e:
                logger.error(f"Error setting DOF targets: {e}")
                import traceback

                traceback.print_exc()
