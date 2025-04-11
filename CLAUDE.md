# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands
- Install: `pip install -e .`
- Train: `python DexHandEnv/train.py task=DexGrasp num_envs=4096 headless=True`
- Test: `python DexHandEnv/train.py task=DexGrasp test=True num_envs=2 checkpoint=runs/DexGrasp_*/nn/DexGrasp.pth`
- Multi-GPU: `torchrun --standalone --nnodes=1 --nproc_per_node=2 DexHandEnv/train.py multi_gpu=True task=DexReorient`
- Record video: `python DexHandEnv/train.py task=DexReorient capture_video=True capture_video_freq=1500 capture_video_len=100`

## Code Style Guidelines
- Imports: Standard library → third-party → local application imports
- Formatting: 4-space indentation, ~80 char line length
- Types: Use PEP 484 type hints where possible
- Naming: snake_case for functions/variables, PascalCase for classes, UPPER_SNAKE_CASE for constants
- Documentation: Include docstrings for functions and classes
- Configuration: Use Hydra with YAML files in cfg/ directory
- Error handling: Use try/except with specific exceptions and informative messages