"""
CLI utilities for DexHand training script.

Provides simplified command-line interface with aliases and smart path resolution.
"""

import sys
from pathlib import Path
from typing import List, Tuple
from loguru import logger


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
        "maxIter": "train.maxIterations",
        "logLevel": "train.logging.logLevel",
    }

    def __init__(self):
        self.original_args = None
        self.processed_args = None

    def preprocess_args(self, args: List[str]) -> List[str]:
        """
        Preprocess command-line arguments to expand aliases and resolve paths.

        Args:
            args: Original command-line arguments

        Returns:
            Processed arguments with aliases expanded and paths resolved
        """
        self.original_args = args.copy()
        processed = []

        # Extract task name for task-aware checkpoint resolution
        task_name = self._extract_task_name(args)

        for arg in args:
            # Skip if not a key=value override
            if "=" not in arg:
                processed.append(arg)
                continue

            key, value = arg.split("=", 1)

            # Handle special config alias (maps to --config-name)
            if key == "config":
                processed.append(f"--config-name={value}")
                logger.debug(
                    f"Expanded config alias: config={value} → --config-name={value}"
                )
                continue

            # Handle aliases
            if key in self.ALIASES:
                key = self.ALIASES[key]
                logger.debug(f"Expanded alias: {args[args.index(arg)]} → {key}={value}")

            # Handle special checkpoint path resolution
            if key in ["train.checkpoint", "checkpoint"]:
                if key == "checkpoint":
                    key = "train.checkpoint"
                value = self._resolve_checkpoint_path(value, task_name)

            processed.append(f"{key}={value}")

        self.processed_args = processed
        return processed

    def _extract_task_name(self, args: List[str]) -> str:
        """
        Extract task name from command-line arguments.

        Args:
            args: Command-line arguments

        Returns:
            Task name if found, None otherwise
        """
        for arg in args:
            if "=" in arg:
                key, value = arg.split("=", 1)
                if key == "task":
                    return value
        return None

    def _resolve_checkpoint_path(self, path: str, task_name: str = None) -> str:
        """
        Resolve checkpoint path with smart directory handling.

        Args:
            path: Original checkpoint path (can be file or directory)
            task_name: Task name for filtering latest checkpoints

        Returns:
            Resolved path to actual .pth file
        """
        if path in ["null", "None", ""]:
            return path

        path_obj = Path(path)

        # If it's already a .pth file and exists, return as-is
        if path_obj.suffix == ".pth":
            if path_obj.exists():
                logger.debug(f"Using checkpoint file: {path}")
                return str(path_obj)
            else:
                logger.warning(f"Checkpoint file not found: {path}")
                return path

        # If it's a directory, try to auto-resolve
        if path_obj.is_dir():
            return self._auto_resolve_checkpoint_dir(path_obj)

        # If path doesn't exist but looks like a directory path, try to find it
        if not path_obj.exists():
            # Handle special cases like 'latest'
            if path == "latest":
                return self._find_latest_checkpoint(task_name)

            # Try to find similar directory names in both runs/ and runs_all/
            search_dirs = [Path("runs"), Path("runs_all")]
            for search_dir in search_dirs:
                if search_dir.exists():
                    for run_dir in search_dir.iterdir():
                        if (
                            run_dir.is_dir()
                            and path in run_dir.name
                            and run_dir.name != "pinned"
                        ):
                            # Resolve symlinks to get the real directory
                            real_dir = (
                                run_dir.resolve() if run_dir.is_symlink() else run_dir
                            )
                            logger.info(f"Found similar run directory: {real_dir}")
                            return self._auto_resolve_checkpoint_dir(real_dir)

        # Return original path if no resolution possible
        logger.warning(f"Could not resolve checkpoint path: {path}")
        return path

    def _auto_resolve_checkpoint_dir(self, dir_path: Path) -> str:
        """
        Auto-resolve checkpoint file from experiment directory.

        Args:
            dir_path: Path to experiment directory

        Returns:
            Path to best checkpoint file
        """
        # Strategy 1: Look for nn/{task_name}.pth (best checkpoint)
        nn_dir = dir_path / "nn"
        if nn_dir.exists():
            # Try common task names
            for task_name in ["BoxGrasping", "BaseTask"]:
                best_checkpoint = nn_dir / f"{task_name}.pth"
                if best_checkpoint.exists():
                    logger.info(f"Found best checkpoint: {best_checkpoint}")
                    return str(best_checkpoint)

            # Fall back to any .pth file that doesn't start with 'last_'
            for pth_file in nn_dir.glob("*.pth"):
                if not pth_file.name.startswith("last_"):
                    logger.info(f"Found checkpoint: {pth_file}")
                    return str(pth_file)

            # Fall back to latest 'last_' checkpoint
            last_checkpoints = list(nn_dir.glob("last_*.pth"))
            if last_checkpoints:
                # Sort by modification time, newest first
                latest = max(last_checkpoints, key=lambda p: p.stat().st_mtime)
                logger.info(f"Found latest checkpoint: {latest}")
                return str(latest)

        # Strategy 2: Look for any .pth file in the directory
        pth_files = list(dir_path.glob("**/*.pth"))
        if pth_files:
            latest = max(pth_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Found checkpoint in directory: {latest}")
            return str(latest)

        logger.error(f"No checkpoint files found in directory: {dir_path}")
        return str(dir_path)

    def _find_latest_checkpoint(self, task_name: str = None) -> str:
        """
        Find the latest experiment directory and resolve its checkpoint.

        Args:
            task_name: Optional task name to filter directories by

        Returns:
            Path to latest checkpoint
        """
        # Search in both runs/ and runs_all/ directories
        search_dirs = []

        # Add runs_all/ if it exists (primary archive)
        runs_all_dir = Path("runs_all")
        if runs_all_dir.exists():
            search_dirs.append(runs_all_dir)

        # Add runs/ (workspace and legacy)
        runs_dir = Path("runs")
        if runs_dir.exists():
            search_dirs.append(runs_dir)

        if not search_dirs:
            logger.error("No 'runs' or 'runs_all' directories found")
            return "latest"

        # Find experiment directories across all search locations
        exp_dirs = []
        for search_dir in search_dirs:
            for item in search_dir.iterdir():
                # Skip pinned directory and only include real directories
                if item.is_dir() and item.name != "pinned":
                    # Resolve symlinks to get the real directory
                    real_dir = item.resolve() if item.is_symlink() else item
                    exp_dirs.append(real_dir)

        # Remove duplicates (in case of symlinks)
        exp_dirs = list(set(exp_dirs))

        if not exp_dirs:
            logger.error("No experiment directories found in search paths")
            return "latest"

        # Filter directories that actually contain checkpoint files
        dirs_with_checkpoints = []
        for d in exp_dirs:
            # Check if directory contains .pth files (in any subdirectory)
            pth_files = list(d.glob("**/*.pth"))
            if pth_files:
                dirs_with_checkpoints.append(d)

        if not dirs_with_checkpoints:
            # Fall back to old behavior if no directories have checkpoints
            latest_dir = max(exp_dirs, key=lambda d: d.stat().st_mtime)
            logger.warning(
                f"No checkpoint files found in any directory. Using latest by time: {latest_dir}"
            )
            return self._auto_resolve_checkpoint_dir(latest_dir)

        # If task name is specified, try to find directories matching that task first
        if task_name:
            task_dirs = [
                d for d in dirs_with_checkpoints if d.name.startswith(task_name)
            ]
            if task_dirs:
                latest_dir = max(task_dirs, key=lambda d: d.stat().st_mtime)
                logger.info(
                    f"Found latest {task_name} experiment directory: {latest_dir}"
                )
                return self._auto_resolve_checkpoint_dir(latest_dir)
            else:
                logger.warning(
                    f"No {task_name} directories found with checkpoints, falling back to any task"
                )

        # Pick the most recent directory that has checkpoint files (any task)
        latest_dir = max(dirs_with_checkpoints, key=lambda d: d.stat().st_mtime)
        logger.info(f"Found latest experiment directory with checkpoints: {latest_dir}")

        return self._auto_resolve_checkpoint_dir(latest_dir)

    def get_usage_examples(self) -> str:
        """Get usage examples for the CLI aliases."""
        examples = [
            "# Original syntax vs. simplified syntax:",
            "python train.py --config-name=test_render env.numEnvs=1 train.test=true",
            "python train.py config=test_render numEnvs=1 test=true",
            "",
            "# Smart checkpoint resolution:",
            "python train.py checkpoint=runs/BoxGrasping_20250707_183716  # Auto-finds .pth",
            "python train.py checkpoint=latest  # Auto-finds latest experiment (any task)",
            "python train.py task=BoxGrasping checkpoint=latest  # Auto-finds latest BoxGrasping",
            "",
            "# Available aliases:",
        ]

        # Add config alias first (special case)
        examples.append(f"  {'config':12} → --config-name")

        # Add regular aliases
        for alias, full_path in self.ALIASES.items():
            examples.append(f"  {alias:12} → {full_path}")

        return "\n".join(examples)


def preprocess_cli_args() -> Tuple[List[str], CLIPreprocessor]:
    """
    Preprocess sys.argv to handle aliases and smart path resolution.

    Returns:
        Tuple of (processed_args, preprocessor_instance)
    """
    preprocessor = CLIPreprocessor()

    # Skip script name (sys.argv[0])
    original_args = sys.argv[1:]
    processed_args = preprocessor.preprocess_args(original_args)

    # Replace sys.argv with processed args
    sys.argv[1:] = processed_args

    return processed_args, preprocessor


def show_cli_help():
    """Show CLI help with aliases and examples."""
    preprocessor = CLIPreprocessor()
    print("DexHand Training CLI - Simplified Syntax")
    print("=" * 50)
    print(preprocessor.get_usage_examples())
    print()
