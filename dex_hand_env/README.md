# DexHandEnv: Dexterous Manipulation Environment

This package provides a reinforcement learning environment for dexterous manipulation tasks with robotic hands, built on top of NVIDIA's IsaacGym.

## Overview

DexHandEnv is a modular, component-based framework for dexterous manipulation research. It provides:

- A unified framework for creating dexterous manipulation tasks
- A component-based architecture for better code organization and reusability
- A configurable environment with various observation and action spaces
- Pre-built tasks like grasping and reorientation

## Installation

```bash
# Clone the repository
git clone https://github.com/dexrobot/dexrobot_isaac.git
cd dexrobot_isaac

# Install the package
pip install -e .
```

## Usage

### Running Example Tasks

```bash
# Run the DexGrasp task
python examples/run_dex_grasp.py

# Run headless
python examples/run_dex_grasp.py --headless

# Specify number of environments
python examples/run_dex_grasp.py --num_envs 4
```

### Creating Custom Tasks

To create a custom task, extend the `BaseTask` class:

```python
from dex_hand_env.tasks.base_task import BaseTask

class MyCustomTask(BaseTask):
    def __init__(self, sim, gym, device, num_envs, cfg):
        super().__init__(sim, gym, device, num_envs, cfg)
        # Initialize task-specific parameters
        
    def compute_task_reward_terms(self, obs_dict):
        # Compute task-specific rewards
        return {"my_reward": torch.ones(self.num_envs, device=self.device)}
    
    # Implement other required methods...
```

Then register your task in the factory:

```python
# In factory.py
from custom_tasks import MyCustomTask

def create_dex_env(...):
    # ...
    elif task_name == "MyCustomTask":
        task = MyCustomTask(None, None, torch.device(sim_device), cfg["env"]["numEnvs"], cfg)
    # ...
```

## Architecture

The DexHandEnv package uses a component-based architecture:

- `DexHandBase`: Main environment class that implements common functionality
- `DexTask`: Interface for task-specific behavior
- Components:
  - `CameraController`: Handles camera control and keyboard shortcuts
  - `FingertipVisualizer`: Visualizes fingertip contacts with color
  - `SuccessFailureTracker`: Tracks success and failure criteria
  - `RewardCalculator`: Calculates rewards

## Configuration

The environment is configured using a dictionary-like structure:

```python
cfg = {
    "env": {
        "numEnvs": 2,
        "episodeLength": 1000,
        "controlFrequencyInv": 2,
        # ... more environment parameters
    },
    "sim": {
        "dt": 0.01,
        "substeps": 2,
        "gravity": [0.0, 0.0, -9.81],
        # ... more simulation parameters
    },
    "task": {
        # Task-specific parameters
    },
    "reward": {
        # Reward-specific parameters
    }
}
```

## License

See the LICENSE file for licensing information.

## Acknowledgements

This package is developed by DexRobot Inc. It builds upon NVIDIA's IsaacGym and leverages ideas from various reinforcement learning frameworks.