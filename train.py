#!/usr/bin/env python3
"""
Training script for DexHand RL policies using rl_games.

This script provides training functionality for the DexHand environment
using PPO and other RL algorithms from the rl_games library.
"""

import sys
import argparse
import yaml
from datetime import datetime
from pathlib import Path

# Add parent directory to path to import dexhand_env
sys.path.append(str(Path(__file__).parent))

# Import factory first (it will handle isaacgym imports properly)
from dexhand_env.factory import make_env  # noqa: E402
from dexhand_env.utils.config_utils import load_config  # noqa: E402

# Now import torch and other modules
import torch  # noqa: E402
import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402

# Import RL registration after torch (it needs torch)
from dexhand_env.rl import register_rlgames_env  # noqa: E402

# Import rl_games components
from rl_games.common import env_configurations  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402


def set_seed(seed: int, torch_deterministic: bool = False):
    """Set random seed for reproducibility."""
    if seed == -1:
        seed = np.random.randint(0, 10000)

    logger.info(f"Setting seed: {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed


def load_train_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_env_fn(
    task_name: str,
    cfg: dict,
    num_envs: int,
    sim_device: str,
    rl_device: str,
    graphics_device_id: int,
    headless: bool,
):
    """Create environment function for rl_games."""

    def _create_env(**kwargs):
        env = make_env(
            task_name=task_name,
            num_envs=num_envs,
            sim_device=sim_device,
            rl_device=rl_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            cfg=cfg,
        )
        return env

    return _create_env


def build_runner(algo_config: dict, env_creator) -> Runner:
    """Build rl_games runner with configuration."""
    runner = Runner()
    runner.load(algo_config)
    return runner


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train DexHand RL policies")

    # Environment arguments
    parser.add_argument("--task", type=str, default="BaseTask", help="Task to train on")
    parser.add_argument(
        "--num-envs", type=int, default=1024, help="Number of parallel environments"
    )
    parser.add_argument(
        "--sim-device", type=str, default="cuda:0", help="Device for physics simulation"
    )
    parser.add_argument(
        "--rl-device", type=str, default="cuda:0", help="Device for RL algorithm"
    )
    parser.add_argument(
        "--graphics-device-id", type=int, default=0, help="Graphics device ID"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable rendering (default is headless mode)",
    )

    # Training arguments
    parser.add_argument(
        "--train-config",
        type=str,
        default="dexhand_env/cfg/train/BaseTaskPPO.yaml",
        help="Path to training configuration file",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (-1 for random)"
    )
    parser.add_argument(
        "--torch-deterministic",
        action="store_true",
        help="Use deterministic algorithms",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint to load"
    )
    parser.add_argument(
        "--test", action="store_true", help="Run in test mode (no training)"
    )
    parser.add_argument(
        "--max-iterations", type=int, default=10000, help="Maximum training iterations"
    )

    # Logging arguments
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name for experiment (auto-generated if not provided)",
    )
    parser.add_argument(
        "--log-interval", type=int, default=10, help="Logging interval (episodes)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Set logging level",
    )
    parser.add_argument(
        "--no-log-file", action="store_true", help="Disable logging to file"
    )

    args = parser.parse_args()

    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.experiment_name is None:
        args.experiment_name = f"{args.task}_{timestamp}"

    # Create output directory
    output_dir = Path("runs") / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging level
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        level=args.log_level.upper(),
        format="<green>{time:HH:mm:ss}</green> | <level>{level:8}</level> | <level>{message}</level>",
        colorize=True,
    )

    # Add file logging unless disabled
    if not args.no_log_file:
        logger.add(
            output_dir / "train.log",
            level=args.log_level.upper(),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level:8} | {message}",
        )

    # Save command line arguments
    with open(output_dir / "args.yaml", "w") as f:
        yaml.dump(vars(args), f)

    logger.info(f"Starting training with arguments: {args}")

    # Set seed
    args.seed = set_seed(args.seed, args.torch_deterministic)

    # Load task configuration
    task_cfg_path = f"dexhand_env/cfg/task/{args.task}.yaml"
    task_cfg = load_config(task_cfg_path)

    # Load training configuration
    train_cfg = load_train_config(args.train_config)

    # Update training config with runtime parameters
    train_cfg["params"]["seed"] = args.seed
    train_cfg["params"]["config"]["max_epochs"] = args.max_iterations
    train_cfg["params"]["config"]["num_actors"] = args.num_envs
    train_cfg["params"]["config"]["env_name"] = "rlgpu_dexhand"
    train_cfg["params"]["config"]["device"] = args.rl_device
    train_cfg["params"]["config"]["full_experiment_name"] = args.experiment_name

    # Configure TensorBoard logging
    if train_cfg["params"]["config"].get("use_tensorboard", False):
        train_cfg["params"]["config"]["tensorboard_logdir"] = str(output_dir)

    # Set checkpoint if provided
    if args.checkpoint:
        train_cfg["params"]["load_checkpoint"] = True
        train_cfg["params"]["load_path"] = args.checkpoint

    # Register DexHand environment with rl_games
    register_rlgames_env()

    # Create environment creator function
    env_creator = create_env_fn(
        task_name=args.task,
        cfg=task_cfg,
        num_envs=args.num_envs,
        sim_device=args.sim_device,
        rl_device=args.rl_device,
        graphics_device_id=args.graphics_device_id,
        headless=not args.render,  # Default to headless, use --render to enable viewer
    )

    # Update environment creator in config
    env_configurations.configurations["rlgpu_dexhand"]["env_creator"] = env_creator

    # Build and run trainer
    runner = build_runner(train_cfg, env_creator)

    # Save training config
    with open(output_dir / "train_config.yaml", "w") as f:
        yaml.dump(train_cfg, f)

    # Run training or testing
    if args.test:
        logger.info("Running in test mode")
        runner.run(
            {"train": False, "play": True, "checkpoint": args.checkpoint, "sigma": None}
        )
    else:
        logger.info("Starting training")
        runner.run(
            {"train": True, "play": False, "checkpoint": args.checkpoint, "sigma": None}
        )

    logger.info(f"Training completed. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
