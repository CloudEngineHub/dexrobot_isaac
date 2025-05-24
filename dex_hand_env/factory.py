"""
Factory for creating DexHand environments.

This module provides factory functions for creating DexHand environments
with different tasks.
"""

# Import PyTorch
import torch

from dex_hand_env.tasks.dex_hand_base import DexHandBase
from dex_hand_env.tasks.dex_grasp_task import DexGraspTask
from dex_hand_env.tasks.base_task import BaseTask


def create_dex_env(task_name, cfg, rl_device, sim_device, graphics_device_id, headless,
                   virtual_screen_capture=False, force_render=False):
    """
    Create a DexHand environment with the specified task.
    
    Args:
        task_name: Name of the task to create
        cfg: Configuration dictionary
        rl_device: Device for RL computations
        sim_device: Device for simulation
        graphics_device_id: Graphics device ID
        headless: Whether to run headless
        virtual_screen_capture: Whether to enable virtual screen capture
        force_render: Whether to force rendering
        
    Returns:
        A DexHand environment with the specified task
    """
    print(f"Creating DexHand environment with task: {task_name}")
    
    # Create the task component based on the task name
    try:
        if task_name == "DexGrasp":
            # We need to pass the sim and gym instances, but they're not available yet
            # We'll create a placeholder and update it after initializing the environment
            print("Creating DexGraspTask...")
            task = DexGraspTask(None, None, torch.device(sim_device), cfg["env"]["numEnvs"], cfg)
        elif task_name == "DexHand" or task_name == "Base":
            # Base task with minimal functionality
            print("Creating BaseTask...")
            task = BaseTask(None, None, torch.device(sim_device), cfg["env"]["numEnvs"], cfg)
        else:
            raise ValueError(f"Unknown task: {task_name}")
        
        print("Task created successfully, creating environment...")
        
        # Create the environment with the task component
        env = DexHandBase(
            cfg,
            task,
            rl_device,
            sim_device,
            graphics_device_id,
            headless,
            virtual_screen_capture,
            force_render
        )
        
        print("Environment created, updating task with sim and gym instances...")
        
        # Update the task with the sim and gym instances
        if env.sim is None:
            raise ValueError("Environment simulation is None. Initialization failed.")
            
        task.sim = env.sim
        task.gym = env.gym
        
        # Load task assets now that we have the sim instance
        print("Loading task-specific assets...")
        task.load_task_assets()
        print("Task assets loaded successfully")
        
        return env
        
    except Exception as e:
        print(f"ERROR in create_dex_env: {e}")
        import traceback
        traceback.print_exc()
        raise