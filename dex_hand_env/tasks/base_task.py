"""
Base task implementation for DexHand.

This module provides a minimal task implementation that satisfies the DexTask interface
without adding any specific task behavior. It can be used as a starting point for new tasks
or for testing the basic environment functionality.
"""

from typing import Dict, Optional, Tuple

# Import PyTorch
import torch

from dex_hand_env.tasks.task_interface import DexTask


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
    
    def compute_task_rewards(self, obs_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute task rewards.
        
        Args:
            obs_dict: Dictionary of observations
            
        Returns:
            Tuple of (reward tensor, reward terms dictionary)
        """
        # Base task returns zero rewards
        rewards = torch.zeros(self.num_envs, device=self.device)
        reward_terms = {}
        return rewards, reward_terms
    
    def check_task_reset(self) -> torch.Tensor:
        """
        Check if task-specific reset conditions are met.
        
        Returns:
            Boolean tensor indicating which environments should reset
        """
        # Base task has no specific reset conditions
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    
    def compute_task_reward_terms(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
    
    def get_task_observations(self, obs_dict: Dict[str, torch.Tensor]) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get task-specific observations.
        
        The base task doesn't provide any task-specific observations.
        
        Args:
            obs_dict: Dictionary of current observations
            
        Returns:
            None, indicating no task-specific observations
        """
        return None