"""
CLI utilities for DexHand training script.

Provides simplified command-line interface with aliases and smart path resolution.
"""

import sys
from pathlib import Path
from typing import List, Tuple, Optional


def find_latest_checkpoint_file(experiment_dir: Path) -> Optional[Path]:
    """Find the most recent checkpoint file in an experiment directory.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        Path to the most recent .pth file, or None if no checkpoints found
    """
    # Look for nn/*.pth files first (standard location)
    nn_dir = experiment_dir / "nn"
    if nn_dir.exists():
        pth_files = list(nn_dir.glob("*.pth"))
        if pth_files:
            # Always prioritize most recently modified checkpoint
            return max(pth_files, key=lambda p: p.stat().st_mtime)

    # Fall back to any .pth file in the directory
    pth_files = list(experiment_dir.glob("**/*.pth"))
    if pth_files:
        return max(pth_files, key=lambda p: p.stat().st_mtime)

    return None


class CLIPreprocessor:
    """Preprocesses command-line arguments to handle aliases and smart path resolution."""

    # Mapping from short aliases to full Hydra paths
    ALIASES = {
        "numEnvs": "env.numEnvs",
        "test": "train.test",
        "checkpoint": "train.checkpoint",
        "seed": "train.seed",
        "render": "env.render",
        "device": "env.device",
        "maxIterations": "train.maxIterations",
        "logLevel": "train.logging.logLevel",
    }

    def preprocess_args(self, args: List[str]) -> List[str]:
        """Preprocess command-line arguments to expand aliases and resolve paths."""
        processed = []
        task_name = self._extract_task_name(args)

        for arg in args:
            if "=" not in arg:
                processed.append(arg)
                continue

            key, value = arg.split("=", 1)

            # Handle config alias
            if key == "config":
                processed.append(f"--config-name={value}")
                continue

            # Handle aliases
            if key in self.ALIASES:
                key = self.ALIASES[key]

            # Handle checkpoint path resolution
            if key in ["train.checkpoint", "checkpoint"]:
                if key == "checkpoint":
                    key = "train.checkpoint"
                value = self._resolve_checkpoint_path(value, task_name)

            processed.append(f"{key}={value}")

        return processed

    def _extract_task_name(self, args: List[str]) -> str:
        """Extract task name from command-line arguments."""
        for arg in args:
            if "=" in arg:
                key, value = arg.split("=", 1)
                if key == "task":
                    return value
        return None

    def _resolve_checkpoint_path(self, path: str, task_name: str = None) -> str:
        """Resolve checkpoint path to a .pth file."""
        if path in ["null", "None", ""]:
            return path

        path_obj = Path(path)

        # Direct .pth file
        if path_obj.suffix == ".pth" and path_obj.exists():
            return str(path_obj)

        # Directory - auto-resolve to checkpoint file
        if path_obj.is_dir():
            return self._find_checkpoint_in_dir(path_obj)

        # Special symlinks
        if path in ["latest", "latest_train", "latest_test"]:
            return self._resolve_latest_checkpoint(path, task_name)

        # Partial path matching
        result = self._find_matching_directory(path)
        return result or path

    def _resolve_experiment_dir(self, path: str, task_name: str = None) -> str:
        """Resolve path to an experiment directory."""
        if path in ["null", "None", ""]:
            return path

        path_obj = Path(path)

        # Direct directory
        if path_obj.is_dir():
            return str(path_obj)

        # Direct .pth file - return parent directory
        if path_obj.suffix == ".pth" and path_obj.exists():
            # If file is in nn/ subdirectory, return parent of that
            if path_obj.parent.name == "nn":
                return str(path_obj.parent.parent)
            return str(path_obj.parent)

        # Special symlinks
        if path in ["latest", "latest_train", "latest_test"]:
            return self._resolve_latest_experiment_dir(path, task_name)

        # Partial path matching
        result = self._find_matching_experiment_dir(path)
        return result or path

    def _find_checkpoint_in_dir(self, dir_path: Path) -> str:
        """Find most recent checkpoint file in directory."""
        checkpoint_file = find_latest_checkpoint_file(dir_path)
        if checkpoint_file:
            return str(checkpoint_file)
        return str(dir_path)

    def _resolve_latest_checkpoint(
        self, symlink_name: str, task_name: str = None
    ) -> str:
        """Resolve latest checkpoint to a .pth file using symlinks."""
        runs_dir = Path("runs")

        # Try symlink first
        if symlink_name == "latest":
            symlink_name = "latest_train"  # Default to train

        symlink_path = runs_dir / symlink_name
        if symlink_path.exists() and symlink_path.is_symlink():
            resolved_dir = symlink_path.resolve()
            return self._find_checkpoint_in_dir(resolved_dir)

        # Fallback to manual search
        return self._find_latest_by_search(
            task_name, symlink_name.replace("latest_", "")
        )

    def _resolve_latest_experiment_dir(
        self, symlink_name: str, task_name: str = None
    ) -> str:
        """Resolve latest experiment directory using symlinks."""
        runs_dir = Path("runs")

        # Try symlink first
        if symlink_name == "latest":
            symlink_name = "latest_train"  # Default to train

        symlink_path = runs_dir / symlink_name
        if symlink_path.exists() and symlink_path.is_symlink():
            resolved_dir = symlink_path.resolve()
            return str(resolved_dir)

        # Fallback to manual search
        return self._find_latest_experiment_by_search(
            task_name, symlink_name.replace("latest_", "")
        )

    def _find_latest_by_search(self, task_name: str, run_type: str) -> str:
        """Find latest experiment checkpoint by searching directories."""
        experiments = self._find_experiments_with_checkpoints()

        if not experiments:
            return "latest"

        # Filter by type and task
        if run_type:
            experiments = [
                e for e in experiments if self._classify_run_type(e.name) == run_type
            ]
        if task_name:
            experiments = [e for e in experiments if e.name.startswith(task_name)]

        if experiments:
            latest = max(experiments, key=lambda d: d.stat().st_mtime)
            return self._find_checkpoint_in_dir(latest)

        return "latest"

    def _find_latest_experiment_by_search(self, task_name: str, run_type: str) -> str:
        """Find latest experiment directory by searching directories."""
        experiments = self._find_experiments_with_checkpoints()

        if not experiments:
            return "latest"

        # Filter by type and task
        if run_type:
            experiments = [
                e for e in experiments if self._classify_run_type(e.name) == run_type
            ]
        if task_name:
            experiments = [e for e in experiments if e.name.startswith(task_name)]

        if experiments:
            latest = max(experiments, key=lambda d: d.stat().st_mtime)
            return str(latest)

        return "latest"

    def _find_experiments_with_checkpoints(self) -> List[Path]:
        """Find all experiment directories that contain checkpoints."""
        experiments = []

        # Search runs_all first
        for search_dir in [Path("runs_all"), Path("runs")]:
            if search_dir.exists():
                for item in search_dir.iterdir():
                    if item.is_dir() and item.name != "pinned":
                        real_dir = item.resolve() if item.is_symlink() else item
                        if list(real_dir.glob("**/*.pth")):  # Has checkpoints
                            experiments.append(real_dir)

        return experiments

    def _classify_run_type(self, experiment_name: str) -> str:
        """Classify experiment type."""
        return "test" if "_test_" in experiment_name.lower() else "train"

    def _find_matching_directory(self, path: str) -> str:
        """Find directory with partial name match and return checkpoint file."""
        for search_dir in [Path("runs"), Path("runs_all")]:
            if search_dir.exists():
                for item in search_dir.iterdir():
                    if item.is_dir() and path in item.name and item.name != "pinned":
                        real_dir = item.resolve() if item.is_symlink() else item
                        return self._find_checkpoint_in_dir(real_dir)
        return None

    def _find_matching_experiment_dir(self, path: str) -> str:
        """Find directory with partial name match and return the directory."""
        for search_dir in [Path("runs"), Path("runs_all")]:
            if search_dir.exists():
                for item in search_dir.iterdir():
                    if item.is_dir() and path in item.name and item.name != "pinned":
                        real_dir = item.resolve() if item.is_symlink() else item
                        return str(real_dir)
        return None

    def get_usage_examples(self) -> str:
        """Get usage examples for the CLI aliases."""
        examples = [
            "# Simplified syntax:",
            "python train.py config=test_render numEnvs=1 test=true",
            "",
            "# Smart checkpoint resolution:",
            "python train.py checkpoint=latest          # Latest training run",
            "python train.py checkpoint=latest_train    # Explicit training run",
            "python train.py checkpoint=latest_test     # Explicit test run",
            "",
            "# Available aliases:",
        ]

        examples.append(f"  {'config':12} → --config-name")
        for alias, full_path in self.ALIASES.items():
            examples.append(f"  {alias:12} → {full_path}")

        return "\n".join(examples)


def preprocess_cli_args() -> Tuple[List[str], CLIPreprocessor]:
    """Preprocess sys.argv to handle aliases and smart path resolution."""
    preprocessor = CLIPreprocessor()
    original_args = sys.argv[1:]
    processed_args = preprocessor.preprocess_args(original_args)
    sys.argv[1:] = processed_args
    return processed_args, preprocessor


def show_cli_help():
    """Show CLI help with aliases and examples."""
    preprocessor = CLIPreprocessor()
    print("DexHand Training CLI - Simplified Syntax")
    print("=" * 50)
    print(preprocessor.get_usage_examples())
    print()
