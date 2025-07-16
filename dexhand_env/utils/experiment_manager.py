"""
Experiment directory management utilities for DexHand.

Simple experiment management with train/test separation and latest symlinks.
"""

from pathlib import Path
from typing import Optional, List


def classify_experiment_type(experiment_name: str) -> str:
    """Classify experiment as 'train' or 'test' based on name."""
    return "test" if "_test_" in experiment_name.lower() else "train"


class ExperimentManager:
    """
    Manages experiment directories with workspace and archive.

    Structure:
    - runs_all/: Archive with all experiments
    - runs/: Workspace with recent experiments (symlinks)
    - runs/latest_train, runs/latest_test: Latest symlinks
    """

    def __init__(self, max_train_runs: int = 10, max_test_runs: int = 10):
        self.max_train_runs = max_train_runs
        self.max_test_runs = max_test_runs

        self.runs_all_dir = Path("runs_all")
        self.runs_dir = Path("runs")

        self._ensure_directories()

    def _ensure_directories(self):
        """Ensure required directories exist."""
        self.runs_all_dir.mkdir(exist_ok=True)
        self.runs_dir.mkdir(exist_ok=True)

    def create_experiment_directory(self, experiment_name: str) -> Path:
        """Create experiment directory and manage workspace."""
        # Create in archive
        archive_dir = self.runs_all_dir / experiment_name
        archive_dir.mkdir(parents=True, exist_ok=True)

        # Create workspace symlink
        workspace_symlink = self.runs_dir / experiment_name
        if not workspace_symlink.exists():
            workspace_symlink.symlink_to(archive_dir.absolute())

        # Cleanup and update symlinks
        self._cleanup_workspace()
        self._update_latest_symlinks()

        return archive_dir

    def _cleanup_workspace(self):
        """Remove old symlinks to maintain limits."""
        # Get all experiment symlinks (exclude latest_* symlinks)
        symlinks = [
            item
            for item in self.runs_dir.iterdir()
            if item.is_symlink() and not item.name.startswith("latest_")
        ]

        # Separate by type
        train_symlinks = [
            s for s in symlinks if classify_experiment_type(s.name) == "train"
        ]
        test_symlinks = [
            s for s in symlinks if classify_experiment_type(s.name) == "test"
        ]

        # Sort by modification time (newest first)
        train_symlinks.sort(key=lambda p: p.lstat().st_mtime, reverse=True)
        test_symlinks.sort(key=lambda p: p.lstat().st_mtime, reverse=True)

        # Remove old symlinks
        for old_symlink in train_symlinks[self.max_train_runs :]:
            old_symlink.unlink()
        for old_symlink in test_symlinks[self.max_test_runs :]:
            old_symlink.unlink()

    def _update_latest_symlinks(self):
        """Update latest_train and latest_test symlinks."""
        experiments = self.get_all_experiments()

        # Separate by type
        train_experiments = [
            e for e in experiments if classify_experiment_type(e.name) == "train"
        ]
        test_experiments = [
            e for e in experiments if classify_experiment_type(e.name) == "test"
        ]

        # Update latest_train
        if train_experiments:
            self._update_symlink("latest_train", train_experiments[0])

        # Update latest_test
        if test_experiments:
            self._update_symlink("latest_test", test_experiments[0])

    def _update_symlink(self, symlink_name: str, target: Path):
        """Update a symlink to point to target."""
        symlink_path = self.runs_dir / symlink_name
        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()
        symlink_path.symlink_to(target.absolute())

    def get_all_experiments(self) -> List[Path]:
        """Get all experiments sorted by modification time (newest first)."""
        experiments = []
        if self.runs_all_dir.exists():
            experiments.extend(d for d in self.runs_all_dir.iterdir() if d.is_dir())
        return sorted(experiments, key=lambda p: p.stat().st_mtime, reverse=True)

    def get_latest_experiment(self, run_type: str = "train") -> Optional[Path]:
        """Get latest experiment of specified type."""
        experiments = self.get_all_experiments()
        filtered = [
            e for e in experiments if classify_experiment_type(e.name) == run_type
        ]
        return filtered[0] if filtered else None


def create_experiment_manager(cfg) -> ExperimentManager:
    """Create ExperimentManager from configuration."""
    experiment_cfg = getattr(cfg, "experiment", {})
    max_train_runs = getattr(experiment_cfg, "maxTrainRuns", 10)
    max_test_runs = getattr(experiment_cfg, "maxTestRuns", 10)

    return ExperimentManager(max_train_runs=max_train_runs, max_test_runs=max_test_runs)
