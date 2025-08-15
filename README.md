# DexHand Isaac Gym Environments

This repository provides reinforcement learning environments for training dexterous manipulation policies with robotic hands using NVIDIA's Isaac Gym simulator.

**📖 Quick Links:**
- **[Getting Started](docs/GETTING_STARTED.md)** - 10-minute setup to first trained policy
- **[Training Guide](TRAINING.md)** - Training, testing, and experiment management
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Solutions to common issues
- **[Full Documentation](docs/)** - Complete guides and API reference

## Project Philosophy

This is a research-focused project built with specific architectural principles:

- **Component-Based Architecture**: Clean separation of concerns with modular components (PhysicsManager, ActionProcessor, RewardCalculator, etc.)
- **Fail-Fast Principle**: The system crashes immediately on errors rather than hiding them with fallbacks - this exposes bugs at their source for faster debugging
- **Scientific Computing Mindset**: Vectorized operations, tensor-based computation, and mathematical elegance over defensive business logic
- **Configuration Management**: Uses Hydra for flexible, hierarchical configuration with command-line overrides

## Installation

1. **Install Isaac Gym**
   - Download [Isaac Gym Preview 4](https://developer.nvidia.com/isaac-gym) from NVIDIA
   - Follow their installation instructions
   - Verify installation:
     ```bash
     cd isaacgym/python/examples
     python joint_monkey.py
     ```

2. **Clone Repository with Submodules**
   ```bash
   git clone --recursive https://github.com/dexrobot/dexrobot_isaac
   cd dexrobot_isaac
   ```

   If you already cloned without `--recursive`, initialize the submodules:
   ```bash
   git submodule update --init --recursive
   ```

   **Important**: The `dexrobot_mujoco` submodule contains essential hand models and assets. Always use `git submodule update` to ensure version consistency with the main repository. See [Git Submodule Setup Guide](docs/git-submodule-setup.md) for detailed instructions and troubleshooting.

3. **Install DexHand Environment**
   ```bash
   pip install -e .
   ```

4. **Verify Installation**
   ```bash
   # Test basic environment creation
   python examples/dexhand_test.py

   # Check submodule status
   git submodule status
   ```

   If you encounter issues, see the [Troubleshooting Guide](docs/TROUBLESHOOTING.md) or [Git Submodule Setup Guide](docs/git-submodule-setup.md).

## System Requirements

See [Isaac Gym system requirements](https://developer.nvidia.com/isaac-gym) for hardware and software dependencies. This project requires the same environment as Isaac Gym Preview 4.

### Submodule Troubleshooting

The `assets/dexrobot_mujoco` directory contains robot models as a git submodule. If you encounter issues:

1. **Update submodule to latest version:**
   ```bash
   cd assets/dexrobot_mujoco
   git fetch origin
   git checkout main  # or specific branch/commit
   cd ../..
   git add assets/dexrobot_mujoco
   git commit -m "Update submodule"
   ```

2. **If specific commit is not available on default remote:**
   ```bash
   cd assets/dexrobot_mujoco
   git remote -v  # Check available remotes
   git fetch origin  # or other remote name
   git checkout <requested_commit_hash>
   ```

## Usage Examples

### Basic Testing
```bash
# Visualize the dexterous hand with keyboard controls
python examples/dexhand_test.py

# Run with specific options
python examples/dexhand_test.py --num-envs 4 --episode-length 500
```

> **Note**: `dexhand_test.py` is a low-level testing script with its own command-line arguments. For training and testing RL policies, use `python train.py` with Hydra configuration syntax.

### Advanced Usage
```bash
# Test different control modes
python examples/dexhand_test.py --control-mode position_delta --policy-controls-fingers true

# Run headless for performance testing
python examples/dexhand_test.py --headless --num-envs 16

# Test with movement speed control
python examples/dexhand_test.py --movement-speed 0.5 --debug
```

### Train a Policy
```bash
# Basic training (headless mode for faster training)
python train.py task=BaseTask numEnvs=1024

# Training with visualization
python train.py task=BaseTask numEnvs=64 viewer=true
```

## Available Tasks

- **BaseTask**: Basic environment for testing and development
- **BlindGrasping**: Grasping and manipulation task with box objects

## Training

### Quick Training Commands

```bash
# Basic training with Hydra
python train.py

# Custom task and environment size
python train.py task=BlindGrasping numEnvs=2048

# Test mode with rendering (automatically enabled)
python train.py test=true checkpoint=path/to/checkpoint.pth

# Override configurations
python train.py config=debug  # Use debug.yaml config
```

## Documentation

### Essential Guides
- **[Getting Started](docs/GETTING_STARTED.md)** - 10-minute setup to first trained policy
- **[Training Guide](TRAINING.md)** - Quick commands for training, testing, and customization
- **[Configuration System](docs/guide-configuration-system.md)** - Full config reference and physics tuning
- **[Task Creation](docs/guide-task-creation.md)** - Build custom manipulation tasks

### Practical Workflows
- **[Indefinite Testing](docs/guide-indefinite-testing.md)** - Live policy monitoring during training
- **[HTTP Video Streaming](docs/guide-http-video-streaming.md)** - Remote visualization setup
- **[Physics Tuning](docs/guide-physics-tuning.md)** - Optimize simulation settings
- **[Debugging Guide](docs/guide-debugging.md)** - Component troubleshooting

### Technical Reference
- **[System Architecture](docs/ARCHITECTURE.md)** - Design principles and component structure
- **[Design Decisions](docs/DESIGN_DECISIONS.md)** - Critical caveats and quirks
- **[DOF Control API](docs/reference-dof-control-api.md)** - Action spaces and control modes
- **[Observation System](docs/guide-observation-system.md)** - Observation encoding
- **[Glossary](docs/GLOSSARY.md)** - Project terminology

### Development
- **[CLAUDE.md](CLAUDE.md)** - Development guidelines and architectural principles
- **[ROADMAP.md](ROADMAP.md)** - Project status and future plans

## Configuration

The project uses Hydra for configuration management:
- Main config: `dexhand_env/cfg/config.yaml`
- Task configs: `dexhand_env/cfg/task/`
- Training configs: `dexhand_env/cfg/train/`
- Override configs: `dexhand_env/cfg/{train_headless,test_render,debug}.yaml`

## Troubleshooting

Encountering an issue? Check our comprehensive **[Troubleshooting Guide](docs/TROUBLESHOOTING.md)** for solutions to common setup, execution, training, and API problems.

For Isaac Gym installation and graphics issues, see [Isaac Gym troubleshooting](https://developer.nvidia.com/isaac-gym).

## License

This project is licensed under the Apache License.

## Acknowledgements

This work builds upon the Isaac Gym framework developed by NVIDIA.
