"""
Base task implementation for DexHand.

This module provides a minimal task implementation that satisfies the DexTask interface
without adding any specific task behavior. It can be used as a starting point for new tasks
or for testing the basic environment functionality.
"""

from typing import Dict, Optional, Tuple

# Import PyTorch
import torch

from dexhand_env.tasks.task_interface import DexTask


class BaseTask(DexTask):
    """
    Minimal task implementation for DexHand.

    This task provides the minimal implementation required by the DexTask interface,
    without adding any specific task behavior. It returns empty reward terms,
    no success/failure criteria, and doesn't add any task-specific actors.

    Use this as a base class for new tasks or for testing the basic environment.
    """

    def __init__(self, sim, gym, device, num_envs, cfg):
        """
        Initialize the base task.

        Args:
            sim: Simulation instance
            gym: Gym instance
            device: PyTorch device
            num_envs: Number of environments
            cfg: Configuration dictionary
        """
        self.sim = sim
        self.gym = gym
        self.device = device
        self.num_envs = num_envs
        self.cfg = cfg

        # Initialize state tracking for reward computation
        self.prev_dof_vel = None
        self.prev_hand_vel = None
        self.prev_hand_ang_vel = None
        self.prev_contacts = None

        # Reference to parent environment (set by DexHandBase)
        self.parent_env = None

    def compute_task_rewards(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute task rewards using the reward calculator.

        Args:
            obs_dict: Dictionary of observations

        Returns:
            Tuple of (reward tensor, reward terms dictionary)
        """
        # Check if we have access to parent environment
        if self.parent_env is None:
            # Fallback to zero rewards if parent not set
            rewards = torch.zeros(self.num_envs, device=self.device)
            reward_terms = {}
            return rewards, reward_terms

        # Debug: Check what's in obs_dict
        if not obs_dict:
            # Empty obs_dict, return zero rewards
            rewards = torch.zeros(self.num_envs, device=self.device)
            reward_terms = {}
            return rewards, reward_terms

        # Get current states from parent environment
        hand_vel = self.parent_env.rigid_body_states[
            :, self.parent_env.hand_local_rigid_body_index, 7:10
        ]
        hand_ang_vel = self.parent_env.rigid_body_states[
            :, self.parent_env.hand_local_rigid_body_index, 10:13
        ]
        dof_vel = self.parent_env.dof_vel
        dof_pos = self.parent_env.dof_pos

        # Get DOF limits from tensor manager's DOF properties
        # dof_props shape: (num_dofs, 6) where index 4 is lower limit, 5 is upper limit
        dof_lower_limits = self.parent_env.tensor_manager.dof_props[:, 4]
        dof_upper_limits = self.parent_env.tensor_manager.dof_props[:, 5]

        # Initialize previous states on first call
        if self.prev_dof_vel is None:
            self.prev_dof_vel = dof_vel.clone()
            self.prev_hand_vel = hand_vel.clone()
            self.prev_hand_ang_vel = hand_ang_vel.clone()
            # Initialize contact state
            if "contact_forces" in obs_dict:
                # contact_forces is flattened to (num_envs, num_bodies * 3)
                contact_forces_flat = obs_dict["contact_forces"]
                num_bodies = contact_forces_flat.shape[1] // 3
                contact_forces_3d = contact_forces_flat.view(
                    self.num_envs, num_bodies, 3
                )
                contact_force_norm = torch.norm(contact_forces_3d, dim=2)
                self.prev_contacts = contact_force_norm > 0.1
            else:
                # If contact forces not available yet, initialize to zeros
                # Assume 5 contact bodies (based on typical hand config)
                self.prev_contacts = torch.zeros(
                    (self.num_envs, 5), dtype=torch.bool, device=self.device
                )

        # Compute common reward terms
        common_rewards = self.parent_env.reward_calculator.compute_common_reward_terms(
            obs_dict=obs_dict,
            hand_vel=hand_vel,
            hand_ang_vel=hand_ang_vel,
            dof_vel=dof_vel,
            dof_pos=dof_pos,
            dof_lower_limits=dof_lower_limits,
            dof_upper_limits=dof_upper_limits,
            prev_dof_vel=self.prev_dof_vel,
            prev_hand_vel=self.prev_hand_vel,
            prev_hand_ang_vel=self.prev_hand_ang_vel,
            prev_contacts=self.prev_contacts,
        )

        # Base task has no task-specific rewards
        task_rewards = self.compute_task_reward_terms(obs_dict)

        # Compute total reward
        (
            total_rewards,
            reward_components,
        ) = self.parent_env.reward_calculator.compute_total_reward(
            common_rewards=common_rewards,
            task_rewards=task_rewards,
        )

        # Update previous states for next iteration
        self.prev_dof_vel = dof_vel.clone()
        self.prev_hand_vel = hand_vel.clone()
        self.prev_hand_ang_vel = hand_ang_vel.clone()
        # Update contact state
        contact_forces_flat = obs_dict["contact_forces"]
        num_bodies = contact_forces_flat.shape[1] // 3
        contact_forces_3d = contact_forces_flat.view(self.num_envs, num_bodies, 3)
        contact_force_norm = torch.norm(contact_forces_3d, dim=2)
        self.prev_contacts = contact_force_norm > 0.1

        return total_rewards, reward_components

    def check_task_reset(self) -> torch.Tensor:
        """
        Check if task-specific reset conditions are met.

        Returns:
            Boolean tensor indicating which environments should reset
        """
        # Base task has no specific reset conditions
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def compute_task_reward_terms(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute task-specific reward components.

        The base task doesn't provide any specific rewards beyond the common rewards
        handled by DexHandBase.

        Args:
            obs_dict: Dictionary of observations

        Returns:
            Empty dictionary of task-specific reward components
        """
        return {}

    def check_task_success_criteria(self) -> Dict[str, torch.Tensor]:
        """
        Check task-specific success criteria.

        The base task doesn't define any success criteria.

        Returns:
            Empty dictionary of task-specific success criteria
        """
        return {}

    def check_task_failure_criteria(self) -> Dict[str, torch.Tensor]:
        """
        Check task-specific failure criteria.

        The base task doesn't define any failure criteria.

        Returns:
            Empty dictionary of task-specific failure criteria
        """
        return {}

    def reset_task_state(self, env_ids: torch.Tensor):
        """
        Reset task-specific state for the specified environments.

        The base task doesn't have any specific state to reset.

        Args:
            env_ids: Environment indices to reset
        """
        pass

    def create_task_objects(self, gym, sim, env_ptr, env_id: int):
        """
        Add task-specific objects to the environment.

        The base task doesn't add any specific objects.

        Args:
            gym: Gym instance
            sim: Simulation instance
            env_ptr: Pointer to the environment to add objects to
            env_id: Index of the environment being created
        """
        pass

    def load_task_assets(self):
        """
        Load task-specific assets and define task-specific variables.

        The base task doesn't load any specific assets.
        """
        pass

    def get_task_observations(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get task-specific observations.

        The base task doesn't provide any task-specific observations.

        Args:
            obs_dict: Dictionary of current observations

        Returns:
            None, indicating no task-specific observations
        """
        return None

    def set_tensor_references(self, root_state_tensor: torch.Tensor):
        """
        Set references to simulation tensors needed by the task.

        The base task doesn't need tensor references.

        Args:
            root_state_tensor: Root state tensor for all actors
        """
        pass
