"""
Configuration utilities for DexHand environment.

This module provides utilities for loading and managing configuration files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Union
from loguru import logger


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Dictionary containing configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from: {config_path}")
        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing configuration file {config_path}: {e}")


def merge_configs(
    base_config: Dict[str, Any], override_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.

    Args:
        base_config: Base configuration dictionary
        override_config: Override configuration dictionary

    Returns:
        Merged configuration dictionary
    """
    result = base_config.copy()

    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def save_config(config: Dict[str, Any], save_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary to save
        save_path: Path where to save the configuration
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved configuration to: {save_path}")
