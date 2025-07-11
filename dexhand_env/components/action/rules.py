"""
Action rules system for DexHand environment.

This module provides the functional rule system for action processing,
including pre-action rules, action rules, coupling rules, and post-action filters.
"""

from typing import Callable, Dict, Any, Optional
import torch
from loguru import logger


class ActionRules:
    """
    Manages the functional rule system for action processing.

    This component provides functionality to:
    - Manage pre-action, action, and coupling rules
    - Apply post-action filters (velocity clamp, position clamp)
    - Register custom rules and filters
    """

    def __init__(self, parent):
        """Initialize the action rules system."""
        self.parent = parent

        # Rule components
        self._pre_action_rule = None  # PreActionRule function
        self._action_rule = None  # ActionRule function
        self._post_action_filter_registry = (
            {}
        )  # Dict of name -> PostActionFilter function
        self._enabled_post_action_filters = (
            []
        )  # List of enabled filter names from config
        self._coupling_rule = None  # CouplingRule function

        # Initialize with defaults
        self._initialize_default_rules()

    @property
    def control_dt(self):
        """Access control_dt from physics manager (single source of truth)."""
        return self.parent.physics_manager.control_dt

    def _initialize_default_rules(self):
        """Initialize default functional rules."""
        # Pre-action rule: identity by default
        self._pre_action_rule = None  # Will use identity in apply_pre_action_rule

        # Action rule: must be set explicitly, no default
        self._action_rule = None

        # Coupling rule: use existing apply_coupling wrapped in functional interface
        self._coupling_rule = self.parent.apply_coupling

    def initialize_post_action_filters(self):
        """Initialize post-action filter registry with built-in filters."""
        self._post_action_filter_registry = {
            "velocity_clamp": self._velocity_clamp_filter,
            "position_clamp": self._position_clamp_filter,
        }

        # Get enabled filters from parent configuration
        self._enabled_post_action_filters = getattr(
            self.parent,
            "_enabled_post_action_filters",
            ["velocity_clamp", "position_clamp"],
        )

        logger.debug(
            f"Post-action filters enabled: {self._enabled_post_action_filters}"
        )
        logger.debug(
            f"Post-action filters registered: {list(self._post_action_filter_registry.keys())}"
        )

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
        if self._pre_action_rule is not None:
            return self._pre_action_rule(active_prev_targets, state)
        else:
            # Default: identity function
            return active_prev_targets.clone()

    def apply_action_rule(
        self,
        active_prev_targets: torch.Tensor,
        active_rule_targets: torch.Tensor,
        actions: torch.Tensor,
        config: Dict[str, Any],
    ) -> torch.Tensor:
        """Apply the main action rule."""
        if self._action_rule is None:
            raise RuntimeError(
                "No action rule set. Call set_action_rule() to define how actions are processed."
            )
        return self._action_rule(
            active_prev_targets, active_rule_targets, actions, config
        )

    def apply_post_action_filters(
        self,
        active_prev_targets: torch.Tensor,
        active_rule_targets: torch.Tensor,
        active_targets: torch.Tensor,
    ) -> torch.Tensor:
        """Apply all enabled post-action filters."""
        filtered_targets = active_targets

        for filter_name in self._enabled_post_action_filters:
            if filter_name in self._post_action_filter_registry:
                filter_fn = self._post_action_filter_registry[filter_name]
                filtered_targets = filter_fn(
                    active_prev_targets,
                    active_rule_targets,
                    filtered_targets,
                )
            else:
                logger.warning(
                    f"Post-action filter '{filter_name}' enabled but not registered"
                )

        return filtered_targets

    def apply_coupling_rule(self, active_targets: torch.Tensor) -> torch.Tensor:
        """Apply coupling rule to convert active targets to full DOF targets."""
        return self._coupling_rule(active_targets)

    def _velocity_clamp_filter(
        self,
        active_prev_targets: torch.Tensor,
        active_rule_targets: torch.Tensor,
        active_targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Clamp target changes to respect velocity limits.

        Args:
            active_prev_targets: Previous targets (18D)
            active_rule_targets: Rule targets (18D) - not used
            active_targets: Current targets to filter (18D)

        Returns:
            Filtered targets with velocity constraints applied (18D)
        """
        # Calculate deltas
        delta = active_targets - active_prev_targets

        # Apply velocity-based clamping using precomputed max_deltas
        clamped_delta = torch.clamp(
            delta, -self.parent.max_deltas, self.parent.max_deltas
        )

        return active_prev_targets + clamped_delta

    def _position_clamp_filter(
        self,
        active_prev_targets: torch.Tensor,
        active_rule_targets: torch.Tensor,
        active_targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Clamp targets to DOF position limits.

        Args:
            active_prev_targets: Previous targets (18D) - not used
            active_rule_targets: Rule targets (18D) - not used
            active_targets: Current targets to filter (18D)

        Returns:
            Filtered targets with position constraints applied (18D)
        """
        # Apply position limits using precomputed active limits
        return torch.clamp(
            active_targets,
            self.parent.active_lower_limits,
            self.parent.active_upper_limits,
        )

    # ============================================================================
    # Public API for Rule Registration
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
        self._pre_action_rule = rule

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
        self._action_rule = rule

    def set_coupling_rule(self, rule: Callable[[torch.Tensor], torch.Tensor]):
        """
        Set custom coupling rule.

        Args:
            rule: Function (active_targets) -> full_dof_targets
        """
        self._coupling_rule = rule

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
        self._post_action_filter_registry[name] = filter_fn
        logger.debug(f"Registered post-action filter: {name}")
