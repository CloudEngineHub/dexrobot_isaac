"""
Configuration validation utilities for DexHand.

Provides early validation of configuration files to catch issues before training starts.
"""

from omegaconf import DictConfig
from typing import Any
from loguru import logger


class ConfigValidator:
    """Validates DexHand configuration files for common issues."""

    # Required configuration fields
    REQUIRED_FIELDS = {
        "task.name": str,
        "env.numEnvs": int,
        "env.device": str,
        "train.seed": int,
        "train.test": bool,
        "train.logging.logLevel": str,
    }

    # Valid values for enum-like fields
    VALID_VALUES = {
        "env.device": ["cuda:0", "cpu"],
        "train.logging.logLevel": ["debug", "info", "warning", "error", "critical"],
        "task.controlMode": ["position", "position_delta"],
    }

    # Reasonable ranges for numeric fields
    NUMERIC_RANGES = {
        "env.numEnvs": (1, 16384),
        "train.seed": (0, 2**31 - 1),
        "train.maxIterations": (1, 1000000),
        "sim.dt": (0.001, 0.1),
        "task.episodeLength": (10, 10000),
        "sim.graphicsDeviceId": (0, 8),
    }

    def __init__(self):
        self.warnings = []
        self.errors = []

    def validate_config(self, cfg: DictConfig) -> bool:
        """
        Validate configuration and return True if valid.

        Args:
            cfg: Configuration to validate

        Returns:
            True if configuration is valid, False otherwise
        """
        self.warnings.clear()
        self.errors.clear()

        # Check required fields
        self._validate_required_fields(cfg)

        # Check valid values
        self._validate_enum_fields(cfg)

        # Check numeric ranges
        self._validate_numeric_ranges(cfg)

        # Check task-specific requirements
        self._validate_task_config(cfg)

        # Check consistency
        self._validate_consistency(cfg)

        # Log results
        for warning in self.warnings:
            logger.warning(f"Config validation warning: {warning}")

        for error in self.errors:
            logger.error(f"Config validation error: {error}")

        return len(self.errors) == 0

    def _get_nested_value(self, cfg: DictConfig, path: str) -> Any:
        """Get nested configuration value by dot-separated path."""
        keys = path.split(".")
        value = cfg
        try:
            for key in keys:
                value = getattr(value, key)
            return value
        except AttributeError:
            return None

    def _validate_required_fields(self, cfg: DictConfig):
        """Validate that all required fields are present and have correct types."""
        for field_path, expected_type in self.REQUIRED_FIELDS.items():
            value = self._get_nested_value(cfg, field_path)
            if value is None:
                self.errors.append(f"Required field missing: {field_path}")
            elif not isinstance(value, expected_type):
                self.errors.append(
                    f"Field {field_path} should be {expected_type.__name__}, got {type(value).__name__}"
                )

    def _validate_enum_fields(self, cfg: DictConfig):
        """Validate fields that should have specific values."""
        for field_path, valid_values in self.VALID_VALUES.items():
            value = self._get_nested_value(cfg, field_path)
            if value is not None and value not in valid_values:
                self.errors.append(
                    f"Field {field_path} has invalid value '{value}'. Valid values: {valid_values}"
                )

    def _validate_numeric_ranges(self, cfg: DictConfig):
        """Validate numeric fields are within reasonable ranges."""
        for field_path, (min_val, max_val) in self.NUMERIC_RANGES.items():
            value = self._get_nested_value(cfg, field_path)
            if value is not None:
                if not isinstance(value, (int, float)):
                    self.errors.append(
                        f"Field {field_path} should be numeric, got {type(value).__name__}"
                    )
                elif not (min_val <= value <= max_val):
                    self.warnings.append(
                        f"Field {field_path} value {value} is outside recommended range [{min_val}, {max_val}]"
                    )

    def _validate_task_config(self, cfg: DictConfig):
        """Validate task-specific configuration requirements."""
        task_name = self._get_nested_value(cfg, "task.name")

        if task_name == "BoxGrasping":
            # BoxGrasping specific validations
            box_size = self._get_nested_value(cfg, "env.box.size")
            if box_size is not None and box_size <= 0:
                self.errors.append("BoxGrasping task requires positive box size")

            # Check if required observation keys are present
            obs_keys = self._get_nested_value(cfg, "task.policyObservationKeys")
            if obs_keys is not None:
                required_obs = ["contact_binary", "hand_pose"]
                missing_obs = [key for key in required_obs if key not in obs_keys]
                if missing_obs:
                    self.warnings.append(
                        f"BoxGrasping task missing recommended observations: {missing_obs}"
                    )

    def _validate_consistency(self, cfg: DictConfig):
        """Validate consistency between related configuration values."""
        # Check test mode consistency
        is_test = self._get_nested_value(cfg, "train.test")
        checkpoint = self._get_nested_value(cfg, "train.checkpoint")

        if is_test and not checkpoint:
            self.warnings.append("Test mode enabled but no checkpoint specified")

        # Check environment consistency
        num_envs = self._get_nested_value(cfg, "env.numEnvs")
        render = self._get_nested_value(cfg, "env.render")

        if num_envs and num_envs > 1 and render:
            self.warnings.append(f"Rendering with {num_envs} environments may be slow")

        # Check physics consistency
        dt = self._get_nested_value(cfg, "sim.dt")
        substeps = self._get_nested_value(cfg, "sim.substeps")

        if dt and substeps:
            physics_dt = dt / substeps
            if physics_dt < 0.0001:  # 0.1ms
                self.warnings.append(
                    f"Very small physics timestep {physics_dt:.6f}s may cause instability"
                )


def validate_config(cfg: DictConfig) -> bool:
    """
    Validate configuration and log any issues.

    Args:
        cfg: Configuration to validate

    Returns:
        True if configuration is valid, False otherwise
    """
    validator = ConfigValidator()
    return validator.validate_config(cfg)
