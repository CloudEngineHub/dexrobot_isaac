"""
RL integration for DexHand environment.

This module provides integration with RL libraries like rl_games.
"""

from rl_games.common import env_configurations, vecenv
from loguru import logger


def register_rlgames_env():
    """Register the DexHand environment with rl_games."""
    from dexhand_env.factory import make_env

    def create_env(**kwargs):
        # Extract parameters
        num_envs = kwargs.pop("num_actors", 1024)
        sim_device = kwargs.pop("sim_device", "cuda:0")
        rl_device = kwargs.pop("rl_device", "cuda:0")
        graphics_device_id = kwargs.pop("graphics_device_id", 0)
        headless = kwargs.pop("headless", False)
        cfg = kwargs.pop("cfg", None)
        task_name = kwargs.pop("task_name", "BaseTask")

        # Create and return the environment directly
        return make_env(
            task_name=task_name,
            num_envs=num_envs,
            sim_device=sim_device,
            rl_device=rl_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            cfg=cfg,
        )

    # Register vecenv type for DexHand
    # Since DexHand already implements the standard Gym interface,
    # we can use a simple wrapper that just passes through calls
    class DexHandVecEnv(vecenv.IVecEnv):
        def __init__(self, config_name, num_actors, **kwargs):
            self.env = env_configurations.configurations[config_name]["env_creator"](
                **kwargs
            )

        def step(self, actions):
            return self.env.step(actions)

        def reset(self):
            return self.env.reset()

        def get_number_of_agents(self):
            return 1  # Single agent per environment

        def get_env_info(self):
            return {
                "action_space": self.env.action_space,
                "observation_space": self.env.observation_space,
                "num_envs": self.env.num_envs,
            }

    # Register the vecenv implementation
    vecenv.register(
        "RLGPU",
        lambda config_name, num_actors, **kwargs: DexHandVecEnv(
            config_name, num_actors, **kwargs
        ),
    )

    # Register with rl_games
    env_configurations.register(
        "rlgpu_dexhand",
        {
            "vecenv_type": "RLGPU",
            "env_creator": create_env,
        },
    )

    logger.info("DexHand environment registered with rl_games")


__all__ = ["register_rlgames_env"]
