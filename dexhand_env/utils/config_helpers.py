"""
Configuration helper utilities for DexHand.

Provides utility functions for common configuration operations.
"""

from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger


def get_experiment_name(cfg: DictConfig, timestamp: str) -> str:
    """
    Generate experiment name from config and timestamp.

    Args:
        cfg: Configuration object
        timestamp: Timestamp string (format: YYYYMMDD_HHMMSS)

    Returns:
        Generated experiment name (format: {task}_{train|test}_{YYMMDD_HHMMSS})
    """
    if cfg.logging.experimentName is not None:
        return cfg.logging.experimentName

    # Simplify timestamp from YYYYMMDD_HHMMSS to YYMMDD_HHMMSS
    if len(timestamp) >= 8 and timestamp.startswith("20"):
        short_timestamp = timestamp[2:]  # Remove "20" prefix
    else:
        short_timestamp = timestamp

    # Simple naming: task + mode + timestamp
    mode = "test" if cfg.training.test else "train"
    return f"{cfg.task.name}_{mode}_{short_timestamp}"


def get_config_summary(cfg: DictConfig) -> Dict[str, Any]:
    """
    Get a summary of key configuration parameters.

    Args:
        cfg: Configuration object

    Returns:
        Dictionary with key configuration values
    """
    return {
        "task": cfg.task.name,
        "num_envs": cfg.env.numEnvs,
        "test_mode": cfg.training.test,
        "render": cfg.env.render,
        "checkpoint": cfg.training.checkpoint,
        "seed": cfg.training.seed,
        "max_iterations": cfg.training.maxIterations,
        "physics_dt": getattr(cfg.task.sim, "dt", None),
    }


def resolve_config_safely(cfg: DictConfig) -> Dict[str, Any]:
    """
    Safely resolve configuration with better error handling.

    Args:
        cfg: Configuration to resolve

    Returns:
        Resolved configuration dictionary

    Raises:
        ValueError: If interpolation fails with helpful error message
    """
    try:
        return OmegaConf.to_container(cfg, resolve=True)
    except Exception as e:
        # Try to provide better error message
        error_msg = str(e)
        if "InterpolationKeyError" in error_msg:
            # Extract the missing key from the error
            if "not found" in error_msg:
                missing_key = error_msg.split("'")[1] if "'" in error_msg else "unknown"
                raise ValueError(
                    f"Configuration interpolation failed: key '{missing_key}' not found. "
                    f"This usually means a required configuration section is missing."
                ) from e

        raise ValueError(f"Configuration resolution failed: {error_msg}") from e


def save_config_with_metadata(
    cfg: DictConfig, output_dir: Path, metadata: Optional[Dict[str, Any]] = None
):
    """
    Save configuration with additional metadata.

    Args:
        cfg: Configuration to save
        output_dir: Directory to save config in
        metadata: Additional metadata to include
    """
    # Save main config
    with open(output_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    # Save config summary
    summary = get_config_summary(cfg)
    if metadata:
        summary.update(metadata)

    with open(output_dir / "config_summary.yaml", "w") as f:
        OmegaConf.save(OmegaConf.create(summary), f)

    logger.debug(f"Configuration saved to {output_dir}")


def validate_checkpoint_exists(checkpoint_path: Optional[str]) -> bool:
    """
    Validate that checkpoint file exists if specified.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        True if checkpoint exists or is None, False otherwise
    """
    if checkpoint_path is None or checkpoint_path == "null":
        return True

    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        logger.error(f"Checkpoint file does not exist: {checkpoint_path}")
        return False

    if not checkpoint_file.suffix == ".pth":
        logger.warning(
            f"Checkpoint file does not have .pth extension: {checkpoint_path}"
        )

    return True


def check_config_compatibility(cfg: DictConfig) -> bool:
    """
    Check for potential compatibility issues in configuration.

    Args:
        cfg: Configuration to check

    Returns:
        True if no major compatibility issues found
    """
    issues = []

    # Check for known problematic combinations
    if cfg.env.numEnvs == 1 and not cfg.training.test:
        issues.append(
            "Single environment training is very slow - consider test=true for single env"
        )

    if cfg.env.render and cfg.env.numEnvs > 16:
        issues.append(f"Rendering with {cfg.env.numEnvs} environments may be very slow")

    if cfg.training.test and not cfg.training.checkpoint:
        issues.append("Test mode without checkpoint - will use random policy")

    # Log issues
    for issue in issues:
        logger.warning(f"Config compatibility: {issue}")

    return len(issues) == 0
