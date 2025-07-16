"""
CLI utilities for DexHand training script.

Provides simplified command-line interface with aliases and smart path resolution.
"""

import sys
from pathlib import Path
from typing import List, Tuple


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
        """Resolve checkpoint path with smart directory handling."""
        if path in ["null", "None", ""]:
            return path

        path_obj = Path(path)

        # Direct .pth file
        if path_obj.suffix == ".pth" and path_obj.exists():
            return str(path_obj)

        # Directory - auto-resolve to checkpoint
        if path_obj.is_dir():
            return self._find_checkpoint_in_dir(path_obj)

        # Special symlinks
        if path in ["latest", "latest_train", "latest_test"]:
            return self._resolve_latest_checkpoint(path, task_name)

        # Partial path matching
        return self._find_matching_directory(path) or path

    def _find_checkpoint_in_dir(self, dir_path: Path) -> str:
        """Find best checkpoint in directory."""
        # Look for nn/*.pth files first
        nn_dir = dir_path / "nn"
        if nn_dir.exists():
            pth_files = list(nn_dir.glob("*.pth"))
            if pth_files:
                # Prefer non-last_ files, then most recent
                non_last = [f for f in pth_files if not f.name.startswith("last_")]
                best_file = (
                    non_last[0]
                    if non_last
                    else max(pth_files, key=lambda p: p.stat().st_mtime)
                )
                return str(best_file)

        # Fall back to any .pth file
        pth_files = list(dir_path.glob("**/*.pth"))
        if pth_files:
            return str(max(pth_files, key=lambda p: p.stat().st_mtime))

        return str(dir_path)

    def _resolve_latest_checkpoint(
        self, symlink_name: str, task_name: str = None
    ) -> str:
        """Resolve latest checkpoint using symlinks."""
        runs_dir = Path("runs")

        # Try symlink first
        if symlink_name == "latest":
            symlink_name = "latest_train"  # Default to train

        symlink_path = runs_dir / symlink_name
        if symlink_path.exists() and symlink_path.is_symlink():
            return self._find_checkpoint_in_dir(symlink_path.resolve())

        # Fallback to manual search
        return self._find_latest_by_search(
            task_name, symlink_name.replace("latest_", "")
        )

    def _find_latest_by_search(self, task_name: str, run_type: str) -> str:
        """Find latest experiment by searching directories."""
        experiments = []

        # Search runs_all first
        for search_dir in [Path("runs_all"), Path("runs")]:
            if search_dir.exists():
                for item in search_dir.iterdir():
                    if item.is_dir() and item.name != "pinned":
                        real_dir = item.resolve() if item.is_symlink() else item
                        if list(real_dir.glob("**/*.pth")):  # Has checkpoints
                            experiments.append(real_dir)

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

    def _classify_run_type(self, experiment_name: str) -> str:
        """Classify experiment type."""
        return "test" if "_test_" in experiment_name.lower() else "train"

    def _find_matching_directory(self, path: str) -> str:
        """Find directory with partial name match."""
        for search_dir in [Path("runs"), Path("runs_all")]:
            if search_dir.exists():
                for item in search_dir.iterdir():
                    if item.is_dir() and path in item.name and item.name != "pinned":
                        real_dir = item.resolve() if item.is_symlink() else item
                        return self._find_checkpoint_in_dir(real_dir)
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
