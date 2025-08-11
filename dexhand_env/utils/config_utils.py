"""
Essential configuration utilities for DexHand.

Provides minimal, fail-fast config validation and helper functions.
"""

import yaml
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Dict, Any, Optional, Union
from loguru import logger


def validate_config(cfg: DictConfig) -> None:
    """
    Validate configuration with fail-fast approach.

    Args:
        cfg: Configuration to validate

    Raises:
        AttributeError: If required fields are missing (fail fast)
        ValueError: If critical values are invalid
    """
    # Required fields - let AttributeError crash if missing (fail fast)
    cfg.task.name
    cfg.env.numEnvs
    cfg.env.device
    cfg.train.seed
    cfg.train.test

    # Basic sanity checks - crash on obviously bad values
    if cfg.env.numEnvs <= 0:
        raise ValueError(f"numEnvs must be positive, got {cfg.env.numEnvs}")

    if cfg.env.device not in ["cuda:0", "cpu"]:
        raise ValueError(f"device must be 'cuda:0' or 'cpu', got {cfg.env.device}")


def validate_checkpoint_exists(checkpoint_path: Optional[str]) -> bool:
    """Validate that checkpoint file exists if specified."""
    if checkpoint_path is None or checkpoint_path == "null":
        return True

    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        logger.error(f"Checkpoint file does not exist: {checkpoint_path}")
        return False

    return True


def get_experiment_name(cfg: DictConfig, timestamp: str) -> str:
    """Generate experiment name from config and timestamp."""
    # Check if experimentName is explicitly set (optional field)
    try:
        experiment_name = cfg.train.logging.experimentName
        if experiment_name is not None:
            return experiment_name
    except (AttributeError, KeyError):
        pass  # logging section or experimentName not present

    # Simple naming: task + mode + timestamp
    mode = "test" if cfg.train.test else "train"
    return f"{cfg.task.name}_{mode}_{timestamp}"


def resolve_config_safely(cfg: DictConfig) -> Dict[str, Any]:
    """Safely resolve configuration with better error handling."""
    try:
        return OmegaConf.to_container(cfg, resolve=True)
    except Exception as e:
        error_msg = str(e)
        if "InterpolationKeyError" in error_msg and "not found" in error_msg:
            missing_key = error_msg.split("'")[1] if "'" in error_msg else "unknown"
            raise ValueError(f"Config key '{missing_key}' not found") from e
        raise ValueError(f"Config resolution failed: {error_msg}") from e


def save_config(cfg: DictConfig, output_dir: Path) -> None:
    """Save configuration to output directory."""
    with open(output_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
