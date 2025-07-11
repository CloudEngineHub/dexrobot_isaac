"""
Action scaling utilities for DexHand environment.

This module provides general mathematical scaling utilities for action processing.
Contains no task-specific logic - pure mathematical operations for scaling and clamping.
"""

import torch


class ActionScaling:
    """
    Provides general mathematical utilities for action scaling.

    This component provides pure mathematical functions:
    - Scale actions from [-1, 1] to target ranges
    - Apply velocity-based deltas
    - Clamp values to limits

    No task-specific logic or conditional behavior.
    """

    def __init__(self, parent):
        """Initialize the action scaling utilities."""
        self.parent = parent

    @staticmethod
    def scale_to_limits(
        actions: torch.Tensor, lower_limits: torch.Tensor, upper_limits: torch.Tensor
    ) -> torch.Tensor:
        """
        Scale actions from [-1, 1] to specified limits.

        Args:
            actions: Raw actions in [-1, 1]
            lower_limits: Lower limits for scaling
            upper_limits: Upper limits for scaling

        Returns:
            Scaled actions in limit ranges
        """
        # Map from [-1, 1] to [lower, upper]
        # action = -1 → lower limit
        # action = +1 → upper limit
        return (actions + 1.0) * 0.5 * (upper_limits - lower_limits) + lower_limits

    @staticmethod
    def apply_velocity_deltas(
        prev_targets: torch.Tensor, actions: torch.Tensor, max_deltas: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply velocity-scaled deltas to previous targets.

        Args:
            prev_targets: Previous target positions
            actions: Raw actions in [-1, 1]
            max_deltas: Maximum allowed deltas per timestep

        Returns:
            New targets with applied deltas
        """
        deltas = actions * max_deltas
        return prev_targets + deltas

    @staticmethod
    def clamp_to_limits(
        targets: torch.Tensor, lower_limits: torch.Tensor, upper_limits: torch.Tensor
    ) -> torch.Tensor:
        """
        Clamp targets to specified limits.

        Args:
            targets: Target values to clamp
            lower_limits: Lower limits
            upper_limits: Upper limits

        Returns:
            Clamped targets
        """
        return torch.clamp(targets, lower_limits, upper_limits)

    @staticmethod
    def apply_velocity_clamp(
        new_targets: torch.Tensor, prev_targets: torch.Tensor, max_deltas: torch.Tensor
    ) -> torch.Tensor:
        """
        Clamp target changes to respect velocity limits.

        Args:
            new_targets: Proposed new targets
            prev_targets: Previous targets
            max_deltas: Maximum allowed change per timestep

        Returns:
            Velocity-clamped targets
        """
        delta = new_targets - prev_targets
        clamped_delta = torch.clamp(delta, -max_deltas, max_deltas)
        return prev_targets + clamped_delta
