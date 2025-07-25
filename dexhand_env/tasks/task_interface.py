"""
Task interface for DexHand environment.

This module defines the interface that all task implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Callable, List

# Import PyTorch
import torch


class DexTask(ABC):
    """
    Abstract base class for dexterous manipulation tasks.

    This class defines the interface that all task implementations must follow.
    It specifies methods for computing task-specific rewards, checking success
    and failure criteria, and resetting task-specific state.
    """

    @property
    def task_states(self):
        """Direct access to task states dict for all tasks."""
        if self.parent_env is None:
            raise RuntimeError("parent_env is None - initialization failed")
        return self.parent_env.observation_encoder.task_states

    def register_task_state(self, name: str, shape: tuple, dtype=torch.float32):
        """
        Register a task state with the observation encoder.

        Provides clean access to task state registration without exposing
        internal observation encoder structure to task implementations.

        Args:
            name: Name of the task state
            shape: Shape tuple for the state tensor (e.g., (num_envs,))
            dtype: PyTorch data type (default: torch.float32)

        Returns:
            The registered task state tensor

        Example:
            ```python
            # Register boolean transition flags
            self.register_task_state("just_transitioned", (self.num_envs,), dtype=torch.bool)

            # Register float timer states
            self.register_task_state("stage_timer", (self.num_envs,), dtype=torch.float32)
            ```
        """
        if self.parent_env is None:
            raise RuntimeError("parent_env is None - initialization failed")
        return self.parent_env.observation_encoder.register_task_state(
            name, shape, dtype=dtype
        )

    @property
    def observation_encoder(self):
        """
        Access observation encoder for DOF operations and state queries.

        Provides clean access to observation encoder functionality without exposing
        internal parent environment structure to task implementations.

        Returns:
            ObservationEncoder instance

        Example:
            ```python
            # Get DOF value by name
            thumb_rotation = self.observation_encoder.get_raw_finger_dof(
                "r_f_joint1_1", "pos", obs_dict
            )

            # Access DOF name mappings
            dof_idx = self.observation_encoder.raw_dof_name_to_index["r_f_joint1_1"]
            ```
        """
        if self.parent_env is None:
            raise RuntimeError("parent_env is None - initialization failed")
        return self.parent_env.observation_encoder

    @abstractmethod
    def compute_task_reward_terms(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute task-specific reward components.

        Args:
            obs_dict: Dictionary of observations

        Returns:
            Dictionary of task-specific reward components
        """
        pass

    @abstractmethod
    def check_task_success_criteria(
        self, obs_dict: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Check task-specific success criteria.

        Args:
            obs_dict: Optional dictionary of observations. If provided, can be used
                     for efficiency to avoid recomputing observations.

        Returns:
            Dictionary of task-specific success criteria (name -> boolean tensor)
        """
        pass

    @abstractmethod
    def check_task_failure_criteria(
        self, obs_dict: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Check task-specific failure criteria.

        Args:
            obs_dict: Optional dictionary of observations. If provided, can be used
                     for efficiency to avoid recomputing observations.

        Returns:
            Dictionary of task-specific failure criteria (name -> boolean tensor)
        """
        pass

    @abstractmethod
    def reset_task_state(self, env_ids: torch.Tensor):
        """
        Reset task-specific state for the specified environments.

        Args:
            env_ids: Environment indices to reset
        """
        pass

    @abstractmethod
    def create_task_objects(self, gym, sim, env_ptr, env_id: int):
        """
        Add task-specific objects to the environment.

        This method is called during environment setup to allow tasks to add
        their own actors/objects (like targets, obstacles, etc.).

        Args:
            gym: Gym instance
            sim: Simulation instance
            env_ptr: Pointer to the environment to add objects to
            env_id: Index of the environment being created
        """
        pass

    @abstractmethod
    def load_task_assets(self):
        """
        Load task-specific assets and define task-specific variables.

        This method should load additional assets (cubes, tools, targets, etc.)
        or define task parameters (reward scales, thresholds, etc.) needed for
        the environment.
        """
        pass

    def get_task_observations(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get task-specific observations.

        Args:
            obs_dict: Dictionary of current observations

        Returns:
            Dictionary of task-specific observations, or None if there are no
            task-specific observations.
        """
        return None

    def get_task_dof_targets(
        self,
        num_envs: int,
        device: str,
        base_controlled: bool = True,
        fingers_controlled: bool = True,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get task-specific target positions for DoFs not controlled by the policy.

        This method allows tasks to provide dynamic target positions for DoFs that are not
        controlled by the policy. For example, if the base is not controlled by the
        policy, the task can provide targets for the base DOFs that change over time
        or react to the state of the environment.

        Tasks can implement custom control rules here, such as:
        - Trajectory following for the hand base
        - Pre-defined grasping motions for fingers
        - State-dependent target positions based on object locations
        - Task-phase dependent behaviors

        Args:
            num_envs: Number of environments
            device: PyTorch device
            base_controlled: Whether the base is controlled by the policy
            fingers_controlled: Whether the fingers are controlled by the policy

        Returns:
            Dictionary with optional keys:
            - "base_targets": Tensor of shape (num_envs, 6) for base DoF targets
            - "finger_targets": Tensor of shape (num_envs, 12) for finger DoF targets
            Return None if using default targets (from cfg) for uncontrolled DoFs.

        Examples:
            ```python
            def get_task_dof_targets(self, num_envs, device, base_controlled, fingers_controlled):
                targets = {}

                # If base not controlled by policy, move it in a circle
                if not base_controlled:
                    # Use episode time for smooth trajectory (assumes control_dt is available)
                    # This creates a full circle every ~6.28 seconds
                    episode_time = self.episode_step_count.float() * self.control_dt
                    base_targets = torch.zeros((num_envs, 6), device=device)
                    base_targets[:, 0] = 0.3 * torch.sin(episode_time)  # x position
                    base_targets[:, 1] = 0.3 * torch.cos(episode_time)  # y position
                    base_targets[:, 2] = 0.5  # z position (fixed height)
                    targets["base_targets"] = base_targets

                # If fingers not controlled by policy, execute grasp sequence
                if not fingers_controlled and hasattr(self, 'object_pos'):
                    # Compute finger targets based on object position
                    finger_targets = self._compute_grasp_targets(self.object_pos)
                    targets["finger_targets"] = finger_targets

                return targets
            ```
        """
        return None

    def set_tensor_references(self, root_state_tensor: torch.Tensor):
        """
        Set references to simulation tensors needed by the task.

        This method is called by the environment after tensors are initialized,
        allowing tasks to access simulation state for reset operations.

        Args:
            root_state_tensor: Root state tensor for all actors
        """
        # Default implementation stores the reference if task needs it
        self.root_state_tensor = root_state_tensor

    def initialize_task_states(self):
        """
        Initialize task states that need to be registered with observation encoder.

        This method is called early in initialization, before observation encoder setup,
        to ensure task states are available when computing observation dimensions.

        Tasks that track temporal state (e.g., grasp duration) should override this
        method to register their states with the observation encoder.

        Default implementation does nothing - tasks can override if needed.
        """
        pass

    def finalize_setup(self):
        """
        Complete setup after physics manager and observation encoder are available.

        This method is called after all components are initialized, allowing
        tasks to:
        - Access control_dt from physics manager
        - Register task states with observation encoder
        - Perform any other setup that requires access to environment components

        Default implementation does nothing - tasks can override if needed.
        """
        pass

    # ============================================================================
    # Action Processing Interface
    # ============================================================================

    @property
    def pre_action_rule(self) -> Optional[Callable]:
        """
        Return custom pre-action rule or None for default.

        Pre-action rules process previous targets with state/observations to produce
        rule targets that can be modified by the main action rule.

        Function signature: (active_prev_targets, state) -> active_rule_targets
        - active_prev_targets: torch.Tensor (num_envs, 18) - Previous active targets
        - state: Dict with 'obs_dict' and 'env' keys

        Returns:
            Optional callable implementing pre-action rule or None to use identity function

        Example:
            ```python
            @property
            def pre_action_rule(self):
                def rule(active_prev_targets, state):
                    # Apply some transformation based on task state
                    targets = active_prev_targets.clone()
                    # ... task-specific logic ...
                    return targets
                return rule
            ```
        """
        return None

    @property
    def action_rule(self) -> Optional[Callable]:
        """
        Return custom action rule or None for default position/position_delta behavior.

        Action rules process policy actions with rule targets to produce raw targets.
        This is the main action processing step where actions are interpreted.

        Function signature: (active_prev_targets, active_rule_targets, actions, config) -> active_raw_targets
        - active_prev_targets: torch.Tensor (num_envs, 18) - Previous active targets
        - active_rule_targets: torch.Tensor (num_envs, 18) - Output from pre-action rule
        - actions: torch.Tensor (num_envs, num_actions) - Policy actions
        - config: Dict with control configuration (policy_controls_base, etc.)

        Returns:
            Optional callable implementing action rule or None to use default behavior

        Example:
            ```python
            @property
            def action_rule(self):
                def rule(active_prev_targets, active_rule_targets, actions, config):
                    # Custom action interpretation
                    targets = active_rule_targets.clone()

                    # Example: Policy controls fingers only, task controls base
                    if config["policy_controls_fingers"]:
                        finger_actions = actions[:, :12]
                        targets[:, 6:] = self._process_finger_actions(finger_actions)

                    # Task-specific base control
                    targets[:, :6] = self._compute_task_base_targets()

                    return targets
                return rule
            ```
        """
        return None

    @property
    def post_action_filters(self) -> List[str]:
        """
        Return list of additional post-action filter names.

        Post-action filters are applied after the main action rule to enforce
        constraints like velocity limits or position bounds. Built-in filters
        include "velocity_clamp" and "position_clamp".

        Returns:
            List of filter names to apply in addition to configured filters

        Example:
            ```python
            @property
            def post_action_filters(self):
                return ["task_specific_constraint", "safety_filter"]
            ```
        """
        return []

    def register_custom_filters(self, action_processor) -> None:
        """
        Register custom post-action filters with the action processor.

        This method is called during initialization to register any custom
        filters that the task provides. The filters will be available for
        use if listed in post_action_filters.

        Args:
            action_processor: ActionProcessor instance to register filters with

        Example:
            ```python
            def register_custom_filters(self, action_processor):
                def task_constraint_filter(active_prev_targets, active_rule_targets, active_targets):
                    # Apply task-specific constraints
                    constrained_targets = active_targets.clone()
                    # ... constraint logic ...
                    return constrained_targets

                action_processor.register_post_action_filter(
                    "task_specific_constraint",
                    task_constraint_filter
                )
            ```
        """
        pass
