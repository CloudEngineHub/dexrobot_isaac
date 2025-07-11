"""
Experiment directory management utilities for DexHand.

Provides intelligent experiment directory management with:
- Clean workspace (runs/) with only recent experiments
- Full archive (runs_all/) with all experiments
- User pinning (runs/pinned/) for favorite experiments
- Automatic symlink cleanup and management
"""

import shutil
from pathlib import Path
from typing import List, Optional
from loguru import logger


class ExperimentManager:
    """
    Manages experiment directories with clean workspace and archival system.

    Directory structure:
    - runs_all/: Permanent archive containing all experiments (real directories)
    - runs/: Clean workspace with recent experiments (symlinks to runs_all/) + pinned/
    - runs/pinned/: User favorites (real directories, manually managed)
    """

    def __init__(self, max_recent_runs: int = 10, use_clean_workspace: bool = True):
        """
        Initialize experiment manager.

        Args:
            max_recent_runs: Maximum number of recent runs to keep in workspace
            use_clean_workspace: Enable clean workspace management
        """
        self.max_recent_runs = max_recent_runs
        self.use_clean_workspace = use_clean_workspace

        # Define directory paths
        self.runs_all_dir = Path("runs_all")
        self.runs_dir = Path("runs")
        self.pinned_dir = Path("runs/pinned")

        # Initialize directory structure
        self._ensure_directories()

    def _ensure_directories(self):
        """Ensure all required directories exist."""
        # Create runs_all/ for permanent archive
        self.runs_all_dir.mkdir(exist_ok=True)

        # Create runs/ workspace
        self.runs_dir.mkdir(exist_ok=True)

        # Create runs/pinned/ for user favorites
        self.pinned_dir.mkdir(exist_ok=True)

        # Add .gitkeep to pinned directory
        gitkeep_file = self.pinned_dir / ".gitkeep"
        if not gitkeep_file.exists():
            gitkeep_file.touch()
            logger.debug("Created .gitkeep in runs/pinned/")

    def create_experiment_directory(self, experiment_name: str) -> Path:
        """
        Create new experiment directory with clean workspace management.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Path to the experiment directory (in runs_all/)
        """
        if not self.use_clean_workspace:
            # Legacy behavior: create directly in runs/
            experiment_dir = self.runs_dir / experiment_name
            experiment_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(
                f"Created experiment directory (legacy mode): {experiment_dir}"
            )
            return experiment_dir

        # New behavior: create in runs_all/ and symlink to runs/
        archive_dir = self.runs_all_dir / experiment_name
        workspace_symlink = self.runs_dir / experiment_name

        # Create the real directory in archive
        archive_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created experiment archive: {archive_dir}")

        # Create symlink in workspace (if not exists)
        if not workspace_symlink.exists():
            try:
                workspace_symlink.symlink_to(archive_dir.absolute())
                logger.debug(
                    f"Created workspace symlink: {workspace_symlink} -> {archive_dir}"
                )
            except OSError as e:
                logger.warning(f"Failed to create symlink {workspace_symlink}: {e}")

        # Clean up old symlinks to maintain workspace limit
        self._cleanup_workspace()

        return archive_dir

    def _cleanup_workspace(self):
        """Remove old symlinks from workspace to maintain recent runs limit."""
        if not self.use_clean_workspace:
            return

        # Get all symlinks in runs/ (excluding pinned/)
        symlinks = []
        for item in self.runs_dir.iterdir():
            if item.is_symlink() and item.name != "pinned":
                symlinks.append(item)

        # Sort by modification time (newest first)
        symlinks.sort(key=lambda p: p.lstat().st_mtime, reverse=True)

        # Remove oldest symlinks beyond the limit
        if len(symlinks) > self.max_recent_runs:
            for old_symlink in symlinks[self.max_recent_runs :]:
                try:
                    old_symlink.unlink()
                    logger.debug(f"Removed old workspace symlink: {old_symlink}")
                except OSError as e:
                    logger.warning(f"Failed to remove old symlink {old_symlink}: {e}")

    def find_experiment_directories(self, pattern: str = None) -> List[Path]:
        """
        Find experiment directories across both workspace and archive.

        Args:
            pattern: Optional pattern to filter directory names

        Returns:
            List of experiment directory paths (real paths, not symlinks)
        """
        directories = []

        # Search in runs_all/ (the source of truth)
        if self.runs_all_dir.exists():
            for item in self.runs_all_dir.iterdir():
                if item.is_dir():
                    if pattern is None or pattern in item.name:
                        directories.append(item)

        # Also search in runs/pinned/ for user favorites
        if self.pinned_dir.exists():
            for item in self.pinned_dir.iterdir():
                if item.is_dir() and item.name != ".gitkeep":
                    if pattern is None or pattern in item.name:
                        directories.append(item)

        # Fallback: search in runs/ for legacy directories (real dirs, not symlinks)
        if self.runs_dir.exists():
            for item in self.runs_dir.iterdir():
                if item.is_dir() and not item.is_symlink() and item.name != "pinned":
                    if pattern is None or pattern in item.name:
                        directories.append(item)

        # Remove duplicates and sort by modification time (newest first)
        unique_dirs = list(set(directories))
        unique_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        return unique_dirs

    def get_recent_experiments(self, limit: Optional[int] = None) -> List[Path]:
        """
        Get most recent experiment directories.

        Args:
            limit: Maximum number of experiments to return (default: max_recent_runs)

        Returns:
            List of recent experiment directory paths
        """
        if limit is None:
            limit = self.max_recent_runs

        all_experiments = self.find_experiment_directories()
        return all_experiments[:limit]

    def pin_experiment(self, experiment_name: str) -> bool:
        """
        Pin an experiment to runs/pinned/ by moving it from workspace.

        Args:
            experiment_name: Name of experiment to pin

        Returns:
            True if successfully pinned, False otherwise
        """
        workspace_path = self.runs_dir / experiment_name
        pinned_path = self.pinned_dir / experiment_name

        # Check if experiment exists in workspace
        if not workspace_path.exists():
            logger.error(f"Experiment not found in workspace: {experiment_name}")
            return False

        # Check if already pinned
        if pinned_path.exists():
            logger.warning(f"Experiment already pinned: {experiment_name}")
            return True

        try:
            # If it's a symlink, copy the real directory content
            if workspace_path.is_symlink():
                real_path = workspace_path.resolve()
                shutil.copytree(real_path, pinned_path)
                workspace_path.unlink()  # Remove the symlink
                logger.info(
                    f"Pinned experiment: {experiment_name} (copied from {real_path})"
                )
            else:
                # If it's a real directory, move it
                shutil.move(str(workspace_path), str(pinned_path))
                logger.info(f"Pinned experiment: {experiment_name} (moved)")

            return True

        except Exception as e:
            logger.error(f"Failed to pin experiment {experiment_name}: {e}")
            return False

    def migrate_existing_runs(self):
        """
        Migrate existing runs/ directory to new structure.

        This is a one-time migration helper for users upgrading to the new system.
        """
        if not self.use_clean_workspace:
            logger.info("Clean workspace disabled, skipping migration")
            return

        logger.info("Migrating existing experiment directories to new structure...")

        # Find real directories in runs/ (not symlinks or pinned/)
        legacy_dirs = []
        if self.runs_dir.exists():
            for item in self.runs_dir.iterdir():
                if item.is_dir() and not item.is_symlink() and item.name != "pinned":
                    legacy_dirs.append(item)

        if not legacy_dirs:
            logger.info("No legacy directories found to migrate")
            return

        migrated_count = 0
        for legacy_dir in legacy_dirs:
            archive_path = self.runs_all_dir / legacy_dir.name
            workspace_symlink = self.runs_dir / legacy_dir.name

            try:
                # Move real directory to archive
                if not archive_path.exists():
                    shutil.move(str(legacy_dir), str(archive_path))
                    logger.debug(f"Moved to archive: {legacy_dir} -> {archive_path}")

                    # Create symlink in workspace
                    workspace_symlink.symlink_to(archive_path.absolute())
                    logger.debug(
                        f"Created symlink: {workspace_symlink} -> {archive_path}"
                    )

                    migrated_count += 1
                else:
                    logger.warning(
                        f"Archive path already exists, skipping: {archive_path}"
                    )

            except Exception as e:
                logger.error(f"Failed to migrate {legacy_dir}: {e}")

        if migrated_count > 0:
            logger.info(
                f"Migrated {migrated_count} experiment directories to new structure"
            )
            # Clean up workspace after migration
            self._cleanup_workspace()

    def get_status_summary(self) -> dict:
        """
        Get summary of experiment directory status.

        Returns:
            Dictionary with status information
        """
        archive_count = (
            len(list(self.runs_all_dir.iterdir())) if self.runs_all_dir.exists() else 0
        )
        workspace_symlinks = (
            len(
                [
                    p
                    for p in self.runs_dir.iterdir()
                    if p.is_symlink() and p.name != "pinned"
                ]
            )
            if self.runs_dir.exists()
            else 0
        )
        pinned_count = (
            len(
                [
                    p
                    for p in self.pinned_dir.iterdir()
                    if p.is_dir() and p.name != ".gitkeep"
                ]
            )
            if self.pinned_dir.exists()
            else 0
        )

        return {
            "clean_workspace_enabled": self.use_clean_workspace,
            "max_recent_runs": self.max_recent_runs,
            "total_archived_experiments": archive_count,
            "recent_workspace_experiments": workspace_symlinks,
            "pinned_experiments": pinned_count,
            "runs_all_exists": self.runs_all_dir.exists(),
            "runs_pinned_exists": self.pinned_dir.exists(),
        }


def create_experiment_manager(cfg) -> ExperimentManager:
    """
    Create ExperimentManager from configuration.

    Args:
        cfg: Configuration object with experiment settings

    Returns:
        Configured ExperimentManager instance
    """
    experiment_cfg = getattr(cfg, "experiment", {})

    max_recent_runs = getattr(experiment_cfg, "maxRecentRuns", 10)
    use_clean_workspace = getattr(experiment_cfg, "useCleanWorkspace", True)

    return ExperimentManager(
        max_recent_runs=max_recent_runs, use_clean_workspace=use_clean_workspace
    )
