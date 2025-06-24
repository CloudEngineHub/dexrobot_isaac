"""
RL Games wrapper for DexHand environment.

This module provides a wrapper to make the DexHand environment compatible
with the rl_games library for reinforcement learning training.
"""

from typing import Dict, Any
import numpy as np
from loguru import logger

# Import rl_games before torch
from rl_games.common import env_configurations, vecenv

# Import torch last (after Isaac Gym modules are loaded)
import torch


class RLGamesWrapper(vecenv.IVecEnv):
    """Wrapper to make DexHand environment compatible with rl_games."""

    def __init__(self, config_name: str, num_actors: int, **kwargs):
        """
        Initialize the RL Games wrapper.

        Args:
            config_name: Name of the configuration
            num_actors: Number of parallel environments
            **kwargs: Additional arguments passed to environment
        """
        self.env = kwargs.pop("env")

        # Get observation and action spaces from the environment
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.num_agents = 1  # Single agent per environment

        # Get number of observations and actions
        if hasattr(self.observation_space, "shape"):
            self.num_obs = self.observation_space.shape[0]
        else:
            # For dict observations, flatten them
            self.num_obs = self._get_dict_obs_size()

        self.num_actions = self.action_space.shape[0]

        # Store number of environments
        self._num_envs = num_actors

        logger.info(f"RLGamesWrapper initialized with {self._num_envs} environments")
        logger.info(f"Observation space: {self.num_obs}")
        logger.info(f"Action space: {self.num_actions}")

    def _get_dict_obs_size(self) -> int:
        """Calculate total observation size for dict observations."""
        total_size = 0
        for key, space in self.observation_space.spaces.items():
            if hasattr(space, "shape"):
                total_size += np.prod(space.shape)
        return total_size

    def step(self, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Step the environment.

        Args:
            actions: Actions to take in the environment

        Returns:
            Dictionary containing:
                - obs: Observations
                - rewards: Rewards
                - dones: Done flags
                - infos: Additional information
        """
        # Step the environment
        obs, rewards, dones, infos = self.env.step(actions)

        # Convert observations to tensor if needed
        if isinstance(obs, dict):
            # Flatten dict observations for rl_games
            obs_list = []
            for key in sorted(obs.keys()):
                obs_list.append(obs[key])
            obs = torch.cat(obs_list, dim=-1)

        # Ensure all outputs are tensors
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, device=self.env.device)
        if not isinstance(dones, torch.Tensor):
            dones = torch.tensor(dones, device=self.env.device)

        # Return as tuple like standard gym environments
        return obs, rewards, dones, infos

    def reset(self) -> torch.Tensor:
        """
        Reset the environment.

        Returns:
            Initial observations
        """
        obs = self.env.reset()

        # Convert observations to tensor if needed
        if isinstance(obs, dict):
            # Flatten dict observations for rl_games
            obs_list = []
            for key in sorted(obs.keys()):
                obs_list.append(obs[key])
            obs = torch.cat(obs_list, dim=-1)

        return obs

    def get_number_of_agents(self) -> int:
        """Get number of agents per environment."""
        return self.num_agents

    def get_env_info(self) -> Dict[str, Any]:
        """Get environment information."""
        info = {
            "num_envs": self._num_envs,
            "num_agents": self.num_agents,
            "num_obs": self.num_obs,
            "num_actions": self.num_actions,
            "action_space": self.action_space,
            "observation_space": self.observation_space,
        }
        return info

    @property
    def num_envs(self) -> int:
        """Get number of environments."""
        return self._num_envs

    @property
    def device(self) -> str:
        """Get device."""
        return self.env.device


def create_rlgames_env(task_name: str, **kwargs) -> RLGamesWrapper:
    """
    Create an RL Games compatible environment.

    Args:
        task_name: Name of the task to create
        **kwargs: Additional arguments for environment creation

    Returns:
        RLGamesWrapper instance
    """
    # Import here to avoid circular imports
    from dexhand_env.factory import make_env

    # Extract parameters
    num_envs = kwargs.pop("num_actors", 1024)
    sim_device = kwargs.pop("sim_device", "cuda:0")
    rl_device = kwargs.pop("rl_device", "cuda:0")
    graphics_device_id = kwargs.pop("graphics_device_id", 0)
    headless = kwargs.pop("headless", False)
    cfg = kwargs.pop("cfg", None)

    # Create the base environment
    env = make_env(
        task_name=task_name,
        num_envs=num_envs,
        sim_device=sim_device,
        rl_device=rl_device,
        graphics_device_id=graphics_device_id,
        headless=headless,
        cfg=cfg,
    )

    # Wrap it for rl_games
    wrapped_env = RLGamesWrapper("rlgpu", num_envs, env=env)

    return wrapped_env


def register_rlgames_env():
    """Register the DexHand environment with rl_games."""
    # Register the vectorized environment type
    vecenv.register(
        "RLGPU_DEXHAND",
        lambda config_name, num_actors, **kwargs: create_rlgames_env(
            task_name=kwargs.get("task_name", "BaseTask"),
            num_actors=num_actors,
            **kwargs,
        ),
    )

    # Register the environment configuration
    env_configurations.register(
        "rlgpu_dexhand",
        {
            "vecenv_type": "RLGPU_DEXHAND",
            "env_creator": lambda **kwargs: create_rlgames_env(**kwargs),
        },
    )

    logger.info("DexHand environment registered with rl_games")
