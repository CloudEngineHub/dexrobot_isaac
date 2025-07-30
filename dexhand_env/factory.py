"""
Factory for creating DexHand environments.

This module provides factory functions for creating DexHand environments
with different tasks.
"""

# Import loguru
from loguru import logger

# Import tasks first (they will import Isaac Gym)
from dexhand_env.tasks.dexhand_base import DexHandBase
from dexhand_env.tasks.base_task import BaseTask
from dexhand_env.tasks.box_grasping_task import BoxGraspingTask

# Import PyTorch after Isaac Gym modules
import torch


def create_dex_env(
    task_name,
    cfg,
    rl_device,
    sim_device,
    graphics_device_id,
    virtual_screen_capture=False,
    force_render=False,
    video_config=None,
):
    """
    Create a DexHand environment with the specified task.

    Args:
        task_name: Name of the task to create
        cfg: Configuration dictionary
        rl_device: Device for RL computations
        sim_device: Device for simulation
        graphics_device_id: Graphics device ID
        virtual_screen_capture: Whether to enable virtual screen capture
        force_render: Whether to force rendering
        video_config: Optional video recording configuration

    Returns:
        A DexHand environment with the specified task
    """
    logger.info(f"Creating DexHand environment with task: {task_name}")

    # Create the task component based on the task name
    try:
        if task_name == "BaseTask":
            # Base task with minimal functionality
            logger.debug("Creating BaseTask...")
            # Ensure device is properly set - rl_device is the one used for tensors
            task = BaseTask(
                None, None, torch.device(rl_device), cfg["env"]["numEnvs"], cfg
            )
        elif task_name == "BoxGrasping":
            # Box grasping task
            logger.debug("Creating BoxGraspingTask...")
            task = BoxGraspingTask(
                None, None, torch.device(rl_device), cfg["env"]["numEnvs"], cfg
            )
        else:
            raise ValueError(f"Unknown task: {task_name}")

        logger.debug("Task created successfully, creating environment...")

        # Derive headless from explicit viewer configuration
        headless = not cfg["env"]["viewer"]

        # Create the environment with the task component
        env = DexHandBase(
            cfg,
            task,
            rl_device,
            sim_device,
            graphics_device_id,
            headless,
            virtual_screen_capture,
            force_render,
            video_config,
        )

        logger.debug("Environment created successfully")

        return env

    except Exception as e:
        logger.error(f"ERROR in create_dex_env: {e}")
        import traceback

        traceback.print_exc()
        raise


def make_env(
    task_name: str,
    num_envs: int,
    sim_device: str,
    rl_device: str,
    graphics_device_id: int,
    cfg: dict = None,
    virtual_screen_capture: bool = False,
    force_render: bool = False,
    video_config: dict = None,
):
    """
    Create a DexHand environment for RL training.

    This is the main entry point for creating environments compatible with
    RL libraries like rl_games.

    Args:
        task_name: Name of the task (e.g., "BaseTask", "DexGrasp")
        num_envs: Number of parallel environments
        sim_device: Device for physics simulation (e.g., "cuda:0", "cpu")
        rl_device: Device for RL algorithm (e.g., "cuda:0", "cpu")
        graphics_device_id: GPU device ID for rendering
        cfg: Optional configuration dictionary (will load from file if not provided)
        virtual_screen_capture: Whether to enable virtual screen capture
        force_render: Whether to force rendering even in headless mode
        video_config: Optional video recording configuration

    Returns:
        DexHandBase environment instance
    """
    # Load configuration if not provided
    if cfg is None:
        from dexhand_env.utils.config_utils import load_config

        config_path = f"dexhand_env/cfg/task/{task_name}.yaml"
        cfg = load_config(config_path)

    # Ensure numEnvs is set in config
    if "numEnvs" not in cfg["env"]:
        cfg["env"]["numEnvs"] = num_envs
    elif cfg["env"]["numEnvs"] != num_envs:
        logger.info(f"Updating numEnvs from {cfg['env']['numEnvs']} to {num_envs}")
        cfg["env"]["numEnvs"] = num_envs

    # Create environment using existing factory function
    env = create_dex_env(
        task_name=task_name,
        cfg=cfg,
        rl_device=rl_device,
        sim_device=sim_device,
        graphics_device_id=graphics_device_id,
        virtual_screen_capture=virtual_screen_capture,
        force_render=force_render,
        video_config=video_config,
    )

    return env
