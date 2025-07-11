"""
RL Games compatibility patches for DexHand environment.

This module provides patches for rl_games to support enhanced functionality including:
- Hot-reload functionality: Dynamic model reloading during testing when checkpoint files are updated
- Device compatibility patches: Automatic CPU/GPU device mismatch handling
- Additional rl_games integration improvements

These patches ensure seamless integration between DexHand and rl_games library.
"""

import os
import time
import threading
import torch
from loguru import logger
from pathlib import Path


class HotReloadManager:
    """Manages hot-reload functionality for rl_games models."""

    def __init__(self, checkpoint_path: str, reload_interval: float = 30.0):
        self.checkpoint_path = Path(checkpoint_path)
        self.reload_interval = reload_interval
        self.last_mtime = 0
        self.is_running = False
        self.monitor_thread = None
        self.runner = None

    def start_monitoring(self, runner):
        """Start monitoring the checkpoint file for changes."""
        self.runner = runner
        self.is_running = True

        # Get initial modification time
        if self.checkpoint_path.exists():
            self.last_mtime = os.path.getmtime(self.checkpoint_path)

        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        logger.info(
            f"Hot-reload monitoring started: {self.checkpoint_path} (interval: {self.reload_interval}s)"
        )

    def stop_monitoring(self):
        """Stop monitoring the checkpoint file."""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        logger.info("Hot-reload monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop that runs in a separate thread."""
        while self.is_running:
            try:
                if self.checkpoint_path.exists():
                    current_mtime = os.path.getmtime(self.checkpoint_path)

                    # Check if file was modified (and we have a baseline)
                    if current_mtime != self.last_mtime and self.last_mtime > 0:
                        logger.info(
                            f"Checkpoint update detected: {self.checkpoint_path}"
                        )

                        # Wait for player to be available if not yet created
                        max_wait_time = 10.0  # seconds
                        wait_interval = 0.5
                        waited_time = 0.0

                        while waited_time < max_wait_time:
                            if (
                                hasattr(self.runner, "player")
                                and self.runner.player is not None
                                and hasattr(self.runner.player, "model")
                            ):
                                break
                            time.sleep(wait_interval)
                            waited_time += wait_interval

                        self._reload_model()
                        self.last_mtime = current_mtime
                    elif self.last_mtime == 0:
                        # First time seeing the file
                        self.last_mtime = current_mtime

                time.sleep(self.reload_interval)

            except Exception as e:
                logger.error(f"Hot-reload monitoring error: {e}")
                time.sleep(self.reload_interval)

    def _reload_model(self):
        """Reload the model with new checkpoint weights."""
        try:
            logger.info("Reloading model weights...")

            # Load the new checkpoint (safer loading)
            checkpoint = torch.load(
                self.checkpoint_path, map_location="cpu", weights_only=False
            )

            # Extract model state dict (handle different checkpoint formats)
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                # Assume the checkpoint is the state dict itself
                state_dict = checkpoint

            # Get the model from the runner's stored player
            if hasattr(self.runner, "player") and self.runner.player is not None:
                player = self.runner.player
                if hasattr(player, "model"):
                    model = player.model

                    # Load the new weights (same pattern as player's restore method)
                    model.load_state_dict(state_dict, strict=False)

                    # Also load running_mean_std if present (for input normalization)
                    if hasattr(player, "normalize_input") and player.normalize_input:
                        if "running_mean_std" in checkpoint and hasattr(
                            model, "running_mean_std"
                        ):
                            model.running_mean_std.load_state_dict(
                                checkpoint["running_mean_std"]
                            )
                            logger.info("Running mean/std state reloaded")

                    logger.success("Model weights reloaded successfully!")

                    # Log some info about the reload
                    if "epoch" in checkpoint:
                        logger.info(
                            f"Loaded checkpoint from epoch: {checkpoint['epoch']}"
                        )
                    if "total_time" in checkpoint:
                        logger.info(
                            f"Checkpoint training time: {checkpoint['total_time']:.2f}s"
                        )

                else:
                    logger.error(
                        "Player exists but has no 'model' attribute - hot-reload failed"
                    )
            else:
                logger.error("Could not access player from runner - hot-reload failed")
                logger.debug(
                    f"Runner has player attr: {hasattr(self.runner, 'player')}"
                )
                if hasattr(self.runner, "player"):
                    logger.debug(f"Player is None: {self.runner.player is None}")

        except Exception as e:
            logger.error(f"Failed to reload model: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")


# Global hot-reload manager instance
_hot_reload_manager = None


def _apply_device_fix_patch():
    """Apply device fix patch to handle CPU mode device mismatch."""
    try:
        from rl_games.algos_torch.players import PpoPlayerContinuous

        # Store original method
        original_get_action = PpoPlayerContinuous.get_action

        def patched_get_action(self, obses, is_deterministic=False):
            """Patched get_action method with device fix."""
            try:
                return original_get_action(self, obses, is_deterministic)
            except RuntimeError as e:
                if "Expected all tensors to be on the same device" in str(e):
                    logger.info(
                        "Device mismatch detected, moving all player tensors to CPU"
                    )
                    # Move all player tensors to CPU
                    if hasattr(self, "model"):
                        self.model = self.model.to("cpu")
                    if hasattr(self, "actions_low"):
                        self.actions_low = self.actions_low.to("cpu")
                    if hasattr(self, "actions_high"):
                        self.actions_high = self.actions_high.to("cpu")
                    logger.info("All player tensors moved to CPU successfully")
                    # Try again
                    return original_get_action(self, obses, is_deterministic)
                else:
                    raise

        # Apply the patch
        PpoPlayerContinuous.get_action = patched_get_action

        logger.info("Device fix patch applied successfully!")
    except ImportError:
        logger.warning("Could not apply device fix patch - rl_games not available")
    except Exception as e:
        logger.warning(f"Could not apply device fix patch: {e}")


def apply_rl_games_patches():
    """Apply all rl_games compatibility patches."""
    try:
        from rl_games.torch_runner import Runner

        # Store original methods
        original_run = Runner.run

        def patched_run_play(self, args):
            """Patched run_play method that stores the player instance."""
            from rl_games.torch_runner import _restore, _override_sigma

            # Create player and store reference for hot-reload access
            self.player = self.create_player()

            # Restore and override as in original method
            _restore(self.player, args)
            _override_sigma(self.player, args)

            # Run the player
            self.player.run()

        def patched_run(self, args):
            """Patched run method with hot-reload support."""
            global _hot_reload_manager

            # Check if this is test mode with hot-reload enabled
            is_test_mode = args.get("play", False) and not args.get("train", True)
            checkpoint_path = args.get("checkpoint")

            if is_test_mode and checkpoint_path and hasattr(self, "_hot_reload_config"):
                reload_interval = getattr(self, "_hot_reload_config", {}).get(
                    "interval", 30.0
                )

                # Create and start hot-reload manager
                _hot_reload_manager = HotReloadManager(checkpoint_path, reload_interval)
                _hot_reload_manager.start_monitoring(self)

                try:
                    # Run the original method
                    return original_run(self, args)
                finally:
                    # Stop monitoring when done
                    if _hot_reload_manager:
                        _hot_reload_manager.stop_monitoring()
                        _hot_reload_manager = None
            else:
                # No hot-reload needed, run normally
                return original_run(self, args)

        # Apply the patches
        Runner.run_play = patched_run_play
        Runner.run = patched_run

        # Add method to configure hot-reload
        def configure_hot_reload(self, interval=30.0):
            """Configure hot-reload settings for this runner."""
            self._hot_reload_config = {"interval": interval}

        Runner.configure_hot_reload = configure_hot_reload

        # Apply device fix patch for CPU mode
        _apply_device_fix_patch()

        logger.info("RL Games patches applied successfully!")
        logger.info(
            "- Hot-reload functionality: Use runner.configure_hot_reload(interval=30.0) to enable"
        )
        logger.info(
            "- Device compatibility: Automatic CPU/GPU device mismatch handling enabled"
        )

    except Exception as e:
        logger.error(f"Failed to apply RL Games patches: {e}")


def is_hot_reload_active():
    """Check if hot-reload is currently active."""
    global _hot_reload_manager
    return _hot_reload_manager is not None and _hot_reload_manager.is_running


# Legacy function name for backward compatibility
def apply_hot_reload_patch():
    """Legacy function name. Use apply_rl_games_patches() instead."""
    logger.warning(
        "apply_hot_reload_patch() is deprecated. Use apply_rl_games_patches() instead."
    )
    return apply_rl_games_patches()
