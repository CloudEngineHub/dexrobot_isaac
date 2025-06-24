# DexHand Isaac Gym Environments

This repository provides reinforcement learning environments for training dexterous manipulation policies with robotic hands using NVIDIA's Isaac Gym simulator.

## Installation

1. **Install Isaac Gym**
   - Download [Isaac Gym Preview 4](https://developer.nvidia.com/isaac-gym) from NVIDIA
   - Follow their installation instructions
   - Verify installation:
     ```bash
     cd isaacgym/python/examples
     python joint_monkey.py
     ```

2. **Install DexHand Environment**
   ```bash
   git clone https://github.com/dexrobot/dexrobot_isaac
   cd dexrobot_isaac
   pip install -e .
   ```

## Quick Start

### Test the Environment
```bash
# Visualize the dexterous hand with keyboard controls
python examples/dexhand_test.py

# Run with specific options
python examples/dexhand_test.py --num-envs 4 --episode-length 500
```

### Train a Policy
```bash
# Basic training (headless mode for faster training)
python train.py --task BaseTask --num-envs 1024 --headless

# Training with visualization
python train.py --task BaseTask --num-envs 64
```

## Available Tasks

- **BaseTask**: Basic environment for testing and development
- **DexGrasp**: Grasping and lifting objects

## Training

For detailed training instructions, hyperparameter tuning, and advanced usage, see [TRAINING.md](TRAINING.md).

## Configuration

Environment parameters can be customized through YAML configuration files:
- Task configs: `dexhand_env/cfg/task/`
- Training configs: `dexhand_env/cfg/train/`

## License

This project is licensed under the Apache License.

## Acknowledgements

This work builds upon the Isaac Gym framework developed by NVIDIA.
