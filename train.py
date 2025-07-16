#!/usr/bin/env python3
"""
Training script for DexHand RL policies using rl_games.

This script provides training functionality for the DexHand environment
using PPO and other RL algorithms from the rl_games library.
"""

import os
import sys
import yaml
from datetime import datetime
from pathlib import Path
from typing import List
import hydra
from omegaconf import DictConfig, OmegaConf

# Minimal custom resolvers needed for conditional logic
OmegaConf.register_new_resolver("if", lambda pred, a, b: a if pred else b)

# Add parent directory to path to import dexhand_env
sys.path.append(str(Path(__file__).parent))

# Import CLI utilities for preprocessing
from dexhand_env.utils.cli_utils import preprocess_cli_args, show_cli_help  # noqa: E402
from dexhand_env.utils.config_utils import (  # noqa: E402
    validate_config,
    get_experiment_name,
    resolve_config_safely,
    save_config,
    validate_checkpoint_exists,
)
from dexhand_env.utils.experiment_manager import create_experiment_manager  # noqa: E402

# Import factory first (it will handle isaacgym imports properly)
from dexhand_env.factory import make_env  # noqa: E402

# Now import torch and other modules
import torch  # noqa: E402
import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402

# Import RL registration after torch (it needs torch)
from dexhand_env.rl import register_rlgames_env, RewardComponentObserver  # noqa: E402
from dexhand_env.rl.rl_games_patches import apply_rl_games_patches  # noqa: E402

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
    virtual_screen_capture: bool = False,
    force_render: bool = False,
    video_config: dict = None,
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
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
            video_config=video_config,
        )
        return env

    return _create_env


def build_runner(
    algo_config: dict, env_creator, reward_log_interval: int = 10
) -> Runner:
    """Build rl_games runner with configuration."""
    # Create custom observer for reward component logging
    observer = RewardComponentObserver(log_interval=reward_log_interval)

    # Pass observer to runner constructor
    runner = Runner(algo_observer=observer)
    runner.load(algo_config)
    return runner


def get_config_overrides(cfg: DictConfig) -> List[str]:
    """Get list of non-default config overrides for reproducibility."""
    overrides = []

    # Check task override
    if cfg.task.name != "BaseTask":
        overrides.append(f"task={cfg.task.name}")

    # Check environment overrides
    if cfg.env.numEnvs != 1024:  # Default from config.yaml
        overrides.append(f"env.numEnvs={cfg.env.numEnvs}")
    if cfg.env.render is not None:
        overrides.append(f"env.render={cfg.env.render}")

    # Check training overrides
    if cfg.train.test:
        overrides.append("train.test=true")
    if cfg.train.checkpoint:
        overrides.append(f"train.checkpoint={cfg.train.checkpoint}")
    if cfg.train.seed != 42:  # Default from config.yaml
        overrides.append(f"train.seed={cfg.train.seed}")
    if cfg.train.maxIterations != 10000:  # Default from config.yaml
        overrides.append(f"train.maxIterations={cfg.train.maxIterations}")

    # Check logging overrides
    if cfg.train.logging.logLevel != "info":  # Default from config.yaml
        overrides.append(f"train.logging.logLevel={cfg.train.logging.logLevel}")

    return overrides


@hydra.main(config_path="dexhand_env/cfg", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main training function."""

    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = get_experiment_name(cfg, timestamp)

    # Create experiment manager and output directory
    experiment_manager = create_experiment_manager(cfg)
    output_dir = experiment_manager.create_experiment_directory(experiment_name)
    logger.debug(f"Experiment directory created: {output_dir}")

    # Configure logging level
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        level=cfg.train.logging.logLevel.upper(),
        format="<green>{time:HH:mm:ss}</green> | <level>{level:8}</level> | <level>{message}</level>",
        colorize=True,
    )

    # Add file logging unless disabled
    if not cfg.train.logging.noLogFile:
        logger.add(
            output_dir / "train.log",
            level=cfg.train.logging.logLevel.upper(),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level:8} | {message}",
        )

    # Save configuration
    save_config(cfg, output_dir)

    # Save git metadata and command line for reproducibility
    try:
        import subprocess

        git_info = []

        # Save command line information
        git_info.append("=== COMMAND LINE INFORMATION ===")
        git_info.append(f"Original command: {' '.join(sys.argv)}")

        # Reconstruct Hydra command with resolved config
        hydra_cmd_parts = ["python", "train.py"]
        overrides = get_config_overrides(cfg)
        hydra_cmd_parts.extend(overrides)

        git_info.append(f"Hydra equivalent: {' '.join(hydra_cmd_parts)}")
        git_info.append("")
        git_info.append("=== GIT INFORMATION ===")

        # Save to separate command info file as well
        with open(output_dir / "command_info.txt", "w") as f:
            f.write(f"Original command: {' '.join(sys.argv)}\n")
            f.write(f"Hydra equivalent: {' '.join(hydra_cmd_parts)}\n")
            f.write(f"Working directory: {Path.cwd()}\n")
            f.write(f"Timestamp: {timestamp}\n")

        # Get commit hash
        try:
            commit_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
            ).strip()
            git_info.append(f"Commit: {commit_hash}")
        except subprocess.CalledProcessError:
            git_info.append("Commit: Not available (not a git repository)")

        # Get branch name
        try:
            branch = subprocess.check_output(
                ["git", "branch", "--show-current"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            git_info.append(f"Branch: {branch}")
        except subprocess.CalledProcessError:
            git_info.append("Branch: Not available")

        # Get git status (uncommitted changes)
        try:
            status = subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
            )
            if status.strip():
                git_info.append("Status: Uncommitted changes detected")
                git_info.append("Uncommitted files:")
                git_info.append(status)
            else:
                git_info.append("Status: Working tree clean")
        except subprocess.CalledProcessError:
            git_info.append("Status: Not available")

        # Get git diff of uncommitted changes
        try:
            diff = subprocess.check_output(
                ["git", "diff", "HEAD"], stderr=subprocess.DEVNULL, text=True
            )
            if diff.strip():
                git_info.append("\nUncommitted changes diff:")
                git_info.append(diff)
        except subprocess.CalledProcessError:
            pass

        # Save git info to file
        with open(output_dir / "git_info.txt", "w") as f:
            f.write("\n".join(git_info))

        logger.info("Git metadata saved for reproducibility")
    except Exception as e:
        logger.warning(f"Could not save git metadata: {e}")

    logger.info(f"Starting training with config: {OmegaConf.to_yaml(cfg)}")

    # Validate configuration
    try:
        validate_config(cfg)
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return

    # Validate checkpoint exists if specified
    if not validate_checkpoint_exists(cfg.train.checkpoint):
        return

    # Set seed
    set_seed(cfg.train.seed, cfg.train.torchDeterministic)

    # Resolve entire config with full context first, then extract sections
    # This allows interpolations like ${...env.numEnvs} to access the complete config hierarchy
    resolved_cfg = resolve_config_safely(cfg)
    task_cfg = resolved_cfg["task"]
    train_cfg = resolved_cfg["train"]

    # Ensure task_cfg has env section for RL framework compatibility
    if "env" not in task_cfg:
        task_cfg["env"] = resolved_cfg["env"].copy()

    # Ensure task_cfg has sim section for DexHandBase compatibility
    if "sim" not in task_cfg:
        task_cfg["sim"] = resolved_cfg["sim"].copy()

    # Add episodeLength to env section for VecTask compatibility
    task_cfg["env"]["episodeLength"] = task_cfg["episodeLength"]

    # Add physics_engine to root level for VecTask compatibility
    task_cfg["physics_engine"] = task_cfg["sim"]["physics_engine"]

    # Pass logging level to environment (merge logLevel configurations)
    task_cfg["env"]["logLevel"] = cfg.train.logging.logLevel

    # Update training config with runtime parameters (only truly dynamic values)
    train_cfg["config"]["full_experiment_name"] = experiment_name

    # Configure TensorBoard logging
    if train_cfg["config"].get("use_tensorboard", False):
        train_cfg["config"]["tensorboard_logdir"] = str(output_dir)

    # Set checkpoint loading dynamically based on whether checkpoint is provided
    if cfg.train.checkpoint:
        train_cfg["load_checkpoint"] = True
        train_cfg["load_path"] = cfg.train.checkpoint
    else:
        train_cfg["load_checkpoint"] = False
        train_cfg["load_path"] = ""

    # Register DexHand environment with rl_games
    register_rlgames_env()

    # Apply RL Games compatibility patches (hot-reload + device compatibility)
    apply_rl_games_patches()

    # Determine rendering mode with new defaults:
    # - Test mode defaults to render enabled
    # - Train mode defaults to headless
    # - Explicit render setting overrides defaults
    if hasattr(cfg.env, "render") and cfg.env.render is not None:
        # Explicit render setting overrides defaults
        should_render = cfg.env.render
    else:
        # Use new default logic: test mode renders by default, train mode is headless
        should_render = cfg.train.test

    # Handle video recording in headless mode
    record_video = getattr(cfg.env, "recordVideo", False)
    stream_video = getattr(cfg.env, "streamVideo", False)
    virtual_screen_capture = (record_video or stream_video) and not should_render
    force_render = (
        record_video or stream_video
    )  # Force rendering for video capture even in headless

    # Setup video recording and streaming configuration
    video_config = None

    if record_video or stream_video:
        logger.info("Setting up video features:")
        logger.info(f"  - Video recording: {'ENABLED' if record_video else 'DISABLED'}")
        logger.info(f"  - HTTP streaming: {'ENABLED' if stream_video else 'DISABLED'}")

        # Validate dependencies before building config
        video_dependencies_ok = True

        if record_video:
            try:
                import cv2  # noqa: F401

                logger.info("  ✓ OpenCV available for video recording")
            except ImportError:
                logger.error(
                    "  ✗ OpenCV required for video recording. Install with: pip install opencv-python"
                )
                video_dependencies_ok = False

        if stream_video:
            try:
                import flask  # noqa: F401

                logger.info("  ✓ Flask available for HTTP streaming")
            except ImportError:
                logger.error(
                    "  ✗ Flask required for HTTP streaming. Install with: pip install flask"
                )
                video_dependencies_ok = False

        if not video_dependencies_ok:
            logger.error(
                "Video feature dependencies missing. Continuing without video features."
            )
            record_video = False
            stream_video = False
            virtual_screen_capture = False
            force_render = False
        else:
            # Base video configuration - use config as single source of truth
            video_config = {
                "enabled": True,
                "fps": cfg.env.videoFps,
                "resolution": cfg.env.videoResolution,
                "codec": cfg.env.videoCodec,
                "maxDuration": cfg.env.videoMaxDuration,
            }

            # Add file recording config if enabled
            if record_video:
                video_output_dir = os.path.join(output_dir, "videos")
                os.makedirs(video_output_dir, exist_ok=True)
                video_config["output_dir"] = video_output_dir
                logger.info(f"  📁 Video output directory: {video_output_dir}")

            # Add HTTP streaming config if enabled
            if stream_video:
                video_config["stream_enabled"] = True
                video_config["stream_host"] = cfg.env.videoStreamHost
                video_config["stream_port"] = cfg.env.videoStreamPort
                video_config["stream_quality"] = cfg.env.videoStreamQuality
                video_config["stream_buffer_size"] = cfg.env.videoStreamBufferSize
                logger.info(
                    f"  🌐 HTTP stream will be available at: http://{video_config['stream_host']}:{video_config['stream_port']}"
                )

            logger.info(
                f"  ⚙️  Render config: virtual_screen_capture={virtual_screen_capture}, force_render={force_render}"
            )
    else:
        logger.info("Video features disabled (recordVideo=false, streamVideo=false)")

    # Create environment creator function
    env_creator = create_env_fn(
        task_name=cfg.task.name,
        cfg=task_cfg,
        num_envs=cfg.env.numEnvs,
        sim_device=cfg.env.device,
        rl_device=cfg.env.device,
        graphics_device_id=cfg.sim.graphicsDeviceId,
        headless=not should_render,
        virtual_screen_capture=virtual_screen_capture,
        force_render=force_render,
        video_config=video_config,
    )

    # Update environment creator in config
    env_configurations.configurations["rlgpu_dexhand"]["env_creator"] = env_creator

    # Build and run trainer
    # RL Games expects config with 'params' wrapper, but we flattened it
    # Create compatible structure for RL Games
    rl_games_train_cfg = {"params": train_cfg}
    runner = build_runner(
        rl_games_train_cfg,
        env_creator,
        reward_log_interval=cfg.train.logging.rewardLogInterval,
    )

    # Save training config
    with open(output_dir / "train_config.yaml", "w") as f:
        yaml.dump(train_cfg, f)

    # Run training or testing
    if cfg.train.test:
        logger.info("Running in test mode")

        # Configure hot-reload if requested
        if cfg.train.checkpoint and getattr(cfg.train, "reloadInterval", 0) > 0:
            reload_interval = cfg.train.reloadInterval
            runner.configure_hot_reload(interval=reload_interval)
            logger.info(
                f"Hot-reload configured: {cfg.train.checkpoint} (interval: {reload_interval}s)"
            )

        logger.info("About to call runner.run() in test mode")
        runner.run(
            {
                "train": False,
                "play": True,
                "checkpoint": cfg.train.checkpoint,
                "sigma": None,
            }
        )
        logger.info("runner.run() completed in test mode")
    else:
        logger.info("Starting training")
        logger.info("About to call runner.run() in train mode")
        runner.run(
            {
                "train": True,
                "play": False,
                "checkpoint": cfg.train.checkpoint,
                "sigma": None,
            }
        )
        logger.info("runner.run() completed in train mode")

    logger.info(f"Training completed. Results saved to: {output_dir}")


if __name__ == "__main__":
    # Handle help request
    if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h", "help"]:
        show_cli_help()
        sys.exit(0)

    # Preprocess CLI arguments for aliases and smart path resolution
    processed_args, preprocessor = preprocess_cli_args()

    # Run main function with processed arguments
    main()
