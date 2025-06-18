"""
Task interface for DexHand environment.

This module defines the interface that all task implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional

# Import PyTorch
import torch


class DexTask(ABC):
    """
    Abstract base class for dexterous manipulation tasks.

    This class defines the interface that all task implementations must follow.
    It specifies methods for computing task-specific rewards, checking success
    and failure criteria, and resetting task-specific state.
    """

    @abstractmethod
    def compute_task_rewards(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute task rewards.

        Args:
            obs_dict: Dictionary of observations

        Returns:
            Tuple of (reward tensor, reward terms dictionary)
        """
        pass

    @abstractmethod
    def check_task_reset(self) -> torch.Tensor:
        """
        Check if task-specific reset conditions are met.

        Returns:
            Boolean tensor indicating which environments should reset
        """
        pass

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
    def check_task_success_criteria(self) -> Dict[str, torch.Tensor]:
        """
        Check task-specific success criteria.

        Returns:
            Dictionary of task-specific success criteria (name -> boolean tensor)
        """
        pass

    @abstractmethod
    def check_task_failure_criteria(self) -> Dict[str, torch.Tensor]:
        """
        Check task-specific failure criteria.

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
