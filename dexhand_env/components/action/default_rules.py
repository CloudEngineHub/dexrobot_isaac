"""
Default action rules for DexHand environment.

This module provides default action rule implementations for different control modes
(position and position_delta) that can be used across different tasks.
"""

from typing import Callable
from loguru import logger


class DefaultActionRules:
    """
    Factory for default action rules used by the DexHand environment.

    Provides position and position_delta action rules that handle scaling
    and applying policy actions to DOF targets while preserving rule-based
    control for non-policy-controlled DOFs.
    """

    @staticmethod
    def create_position_action_rule(action_processor) -> Callable:
        """
        Create a position mode action rule.

        Args:
            action_processor: ActionProcessor instance for accessing scaling utilities

        Returns:
            Callable action rule function
        """

        def position_action_rule(
            active_prev_targets, active_rule_targets, actions, config
        ):
            """Default position mode action rule using ActionScaling utilities."""
            # Start with rule targets - preserves rule-based control for uncontrolled DOFs
            targets = active_rule_targets.clone()
            scaling = action_processor.action_scaling

            # Only update the DOFs that the policy controls
            if config["policy_controls_base"]:
                # Scale base actions from [-1, 1] to DOF limits
                base_lower = action_processor.active_lower_limits[:6]
                base_upper = action_processor.active_upper_limits[:6]
                scaled_base = scaling.scale_to_limits(
                    actions[:, :6], base_lower, base_upper
                )
                targets[:, :6] = scaled_base

            if config["policy_controls_fingers"]:
                # Get finger action indices
                finger_start = 6 if config["policy_controls_base"] else 0
                finger_end = finger_start + 12

                # Scale finger actions from [-1, 1] to DOF limits
                finger_lower = action_processor.active_lower_limits[6:]
                finger_upper = action_processor.active_upper_limits[6:]
                scaled_fingers = scaling.scale_to_limits(
                    actions[:, finger_start:finger_end], finger_lower, finger_upper
                )
                targets[:, 6:] = scaled_fingers

            return targets

        return position_action_rule

    @staticmethod
    def create_position_delta_action_rule(action_processor) -> Callable:
        """
        Create a position_delta mode action rule.

        Args:
            action_processor: ActionProcessor instance for accessing scaling utilities

        Returns:
            Callable action rule function
        """

        def position_delta_action_rule(
            active_prev_targets, active_rule_targets, actions, config
        ):
            """Default position_delta mode action rule using ActionScaling utilities."""
            # Start with rule targets
            targets = active_rule_targets.clone()
            ap = action_processor
            scaling = ap.action_scaling

            if config["policy_controls_base"]:
                # Apply base deltas using ActionScaling utility
                targets[:, :6] = scaling.apply_velocity_deltas(
                    active_prev_targets[:, :6], actions[:, :6], ap.max_deltas[:6]
                )

            if config["policy_controls_fingers"]:
                # Get finger action indices
                finger_start = 6 if config["policy_controls_base"] else 0
                finger_end = finger_start + 12

                # Apply finger deltas using ActionScaling utility
                targets[:, 6:] = scaling.apply_velocity_deltas(
                    active_prev_targets[:, 6:],
                    actions[:, finger_start:finger_end],
                    ap.max_deltas[6:],
                )

            # Clamp to limits using ActionScaling utility
            targets = scaling.clamp_to_limits(
                targets, ap.active_lower_limits, ap.active_upper_limits
            )

            return targets

        return position_delta_action_rule

    @staticmethod
    def setup_default_action_rule(action_processor, control_mode: str):
        """
        Set up a default action rule based on control mode.

        Args:
            action_processor: ActionProcessor instance to configure
            control_mode: Control mode ("position" or "position_delta")
        """
        if control_mode == "position":
            action_rule = DefaultActionRules.create_position_action_rule(
                action_processor
            )
            action_processor.set_action_rule(action_rule)
            logger.debug("Configured default position action rule")
        elif control_mode == "position_delta":
            action_rule = DefaultActionRules.create_position_delta_action_rule(
                action_processor
            )
            action_processor.set_action_rule(action_rule)
            logger.debug("Configured default position_delta action rule")
        else:
            raise ValueError(f"Unknown control mode: {control_mode}")
