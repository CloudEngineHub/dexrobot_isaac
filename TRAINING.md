# Training DexHand RL Policies

This document describes how to train reinforcement learning policies for the DexHand environment using the rl_games library.

**Related Documentation:**
- [Getting Started](docs/GETTING_STARTED.md) - Quick setup and first training run
- [Configuration System Guide](docs/guide-configuration-system.md) - 4-section hierarchy and physics configs
- [System Architecture](docs/ARCHITECTURE.md) - Design principles and component structure
- [Task Creation Guide](docs/guide-task-creation.md) - Creating custom manipulation tasks
- [DOF and Action Control API](docs/reference-dof-control-api.md) - Understanding action spaces
- [Terminology Glossary](docs/GLOSSARY.md) - Definitions of key concepts

## Quick Start

### Basic Training

To start training a policy with default settings:

```bash
python train.py
```

### Training with Custom Configuration

The project uses Hydra for configuration management. You can override any configuration parameter using either full paths or simplified aliases:

```bash
# Train with different task and environment count
python train.py task=BoxGrasping numEnvs=2048

# Training with visualization (fewer environments for better performance)
python train.py render=true numEnvs=64

# Use predefined configuration
python train.py config=train_headless
```

> **CLI Aliases:** The training script supports convenient aliases that map to full Hydra configuration paths. See the alias reference table below for the complete mapping.

### Test Mode with Automatic Rendering

Test mode automatically enables rendering by default:

```bash
# Test mode (renders by default) with smart checkpoint resolution
python train.py test=true checkpoint=latest

# Test specific task with directory auto-resolution
python train.py task=BoxGrasping test=true checkpoint=runs/BoxGrasping_20250707_183716

# Force headless in test mode
python train.py test=true checkpoint=latest render=false

# Hot-reload test mode (reloads checkpoint every 30 seconds)
python train.py test=true checkpoint=path/to/checkpoint.pth testing.reloadInterval=30
```

### CLI Alias Reference

The training script supports convenient aliases that map to full Hydra configuration paths:

| Alias | Full Path | Description |
|-------|-----------|-------------|
| `config` | `--config-name` | Configuration file to use |
| `numEnvs` | `env.numEnvs` | Number of parallel environments |
| `test` | `training.test` | Enable test mode |
| `checkpoint` | `training.checkpoint` | Checkpoint path for testing |
| `seed` | `training.seed` | Random seed |
| `render` | `env.render` | Enable visualization |
| `device` | `env.device` | CUDA device (e.g., "cuda:0") |
| `maxIter` | `training.maxIterations` | Maximum training iterations |
| `logLevel` | `logging.logLevel` | Logging level for entire system (debug, info, warning) |

> **Note**: The `logLevel` setting controls logging for both the training script and the environment/simulation. This provides a unified logging experience across the entire system.

**Usage Examples:**
```bash
# Using aliases (recommended for simplicity)
python train.py task=BoxGrasping numEnvs=1024 render=true

# Control logging level for entire system
python train.py logLevel=debug  # Enable debug logging everywhere
python train.py logLevel=warning  # Only show warnings and errors

# Using full paths (equivalent)
python train.py task=BoxGrasping env.numEnvs=1024 env.render=true

# Smart checkpoint resolution:
checkpoint=latest                    # Auto-finds latest training experiment
checkpoint=latest_train              # Explicit latest training experiment
checkpoint=latest_test               # Explicit latest testing experiment
checkpoint=runs/experiment_dir       # Auto-finds .pth file in directory
```

## Configuration Structure

The DexHand system uses a clean 4-section configuration hierarchy. See the [Configuration System Guide](docs/guide-configuration-system.md) for detailed information.

### Main Configuration Sections

- **`sim`**: Physics simulation parameters (dt, substeps, PhysX settings)
- **`env`**: Environment setup (numEnvs, device, render, task objects)
- **`task`**: RL task definition (episodes, observations, rewards, termination)
- **`train`**: Training algorithm configuration (rl_games parameters, logging)

### Physics Configuration Selection

Choose physics configuration based on your use case:

```bash
# High precision training (slower but more accurate)
python train.py task=BoxGrasping  # Uses /physics/accurate automatically

# Fast physics for visualization (faster but less precise)
python train.py -cn test_render   # Uses /physics/fast automatically

# Custom physics selection
python train.py +defaults=[config,/physics/accurate]
python train.py +defaults=[config,/physics/fast] render=true
```

**Available Physics Configs:**
- `physics/default`: Balanced quality/performance for BaseTask
- `physics/fast`: ~2-3x faster for real-time visualization
- `physics/accurate`: ~2-3x slower but higher precision for training

### Available Tasks

- `BaseTask`: Basic environment for testing and development
- `BoxGrasping`: Grasping and manipulation task with box objects

### Environment Parameters

- `env.numEnvs`: Number of parallel environments (default: 1024)
- `env.device`: Device for simulation and RL algorithm (default: "cuda:0")
- `env.graphicsDeviceId`: Graphics device ID (default: 0)
- `env.render`: Rendering mode (null=auto, true=force, false=headless)
- `env.recordVideo`: Enable video recording in headless mode

### Training Parameters

- `training.seed`: Random seed, use -1 for random (default: 42)
- `training.torch_deterministic`: Use deterministic algorithms for reproducibility
- `training.checkpoint`: Path to checkpoint to resume from
- `training.test`: Run in test mode (no training)
- `training.max_iterations`: Maximum training iterations (default: 10000)
- `testing.reloadInterval`: Checkpoint hot-reload interval in test mode (seconds)

### Experiment Management Parameters

- `experiment.maxRecentRuns`: Maximum recent experiments to show in workspace (default: 10)
- `experiment.useCleanWorkspace`: Enable workspace management system (default: true)

### Logging Parameters

- `logging.experiment_name`: Name for the experiment (auto-generated if null)
- `logging.log_interval`: Logging interval in episodes (default: 10)

## Configuration Examples

### Physics Configuration Override

```bash
# Training examples with different physics configs
python train.py task=BoxGrasping                    # Uses accurate physics (default)
python train.py task=BaseTask                       # Uses default physics
python train.py +defaults=[config,/physics/fast]   # Override with fast physics

# Test mode with visualization
python train.py test=true render=true -cn test_render  # Uses fast physics automatically

# Custom physics + other overrides
python train.py task=BoxGrasping numEnvs=512 +defaults=[config,/physics/accurate]
```

### Configuration Hierarchy Overrides

```bash
# Override sim section (physics)
python train.py sim.dt=0.01 sim.substeps=8

# Override env section (environment)
python train.py env.numEnvs=2048 env.render=true

# Override task section (RL parameters)
python train.py task.episodeLength=300 task.rewardWeights.object_height=2.0

# Override train section (training algorithm)
python train.py train.seed=123 train.maxIterations=5000

# Complex override example
python train.py task=BoxGrasping env.numEnvs=1024 sim.dt=0.005 task.episodeLength=400
```

## Training Examples

### Resume Training from Checkpoint

```bash
# Using simplified syntax with smart checkpoint resolution
python train.py task=BaseTask checkpoint=runs/BaseTask_20240101_120000

# Using full path
python train.py task=BaseTask training.checkpoint=runs/BaseTask_20240101_120000/nn/checkpoint_1000.pth
```

### Test a Trained Policy

```bash
# Using simplified syntax
python train.py task=BaseTask test=true checkpoint=latest

# Test with specific checkpoint
python train.py task=BaseTask test=true checkpoint=runs/BaseTask_20240101_120000

# Test with hot-reload (checkpoint reloads every 30 seconds)
python train.py task=BaseTask test=true checkpoint=runs/BaseTask_20240101_120000 testing.reloadInterval=30
```

### Video Recording

```bash
# Record video in headless mode
python train.py task=BoxGrasping training.test=true training.checkpoint=latest env.render=false env.recordVideo=true

# Record video with rendering
python train.py task=BoxGrasping training.test=true training.checkpoint=latest env.render=true env.recordVideo=true
```

> **Video Output:** Videos are saved in MP4 format to the Hydra output directory, typically `outputs/<YYYY-MM-DD>/<HH-MM-SS>/videos/`. Video recording may have a minor impact on performance but is useful for debugging and visualization.

### Multi-GPU Training

For multi-GPU training, use PyTorch's distributed launch:

```bash
python -m torch.distributed.launch --nproc_per_node=2 train.py task=BaseTask env.numEnvs=2048 env.render=false
```

## Training Configuration

Training hyperparameters are defined in YAML configuration files located in `dexhand_env/cfg/train/`. See the [rl_games documentation](https://github.com/Denys88/rl_games) for parameter descriptions.

### Creating Custom Training Configs

To create a custom training configuration:

1. Copy an existing config file:
   ```bash
   cp dexhand_env/cfg/train/BaseTaskPPO.yaml dexhand_env/cfg/train/MyTaskPPO.yaml
   ```

2. Modify the hyperparameters as needed

3. Use it for training:
   ```bash
   python train.py task=MyTask train=MyTaskPPO
   ```

## Output Structure

Training outputs use an intelligent workspace management system:

```
runs_all/                   # Permanent archive (all experiments)
├── BaseTask_20240101_120000/
│   ├── config.yaml        # Complete configuration
│   ├── train.log         # Training logs
│   └── nn/               # Neural network checkpoints
│       ├── checkpoint_100.pth
│       └── checkpoint_best.pth
└── BoxGrasping_20240102_150000/
    └── ...

runs/                      # Clean workspace (recent experiments + latest symlinks)
├── BaseTask_train_20240101_120000/     # symlink → runs_all/BaseTask_train_20240101_120000/
├── BoxGrasping_test_20240102_150000/   # symlink → runs_all/BoxGrasping_test_20240102_150000/
├── ... (more recent symlinks)
├── latest_train                        # symlink → latest training experiment
└── latest_test                         # symlink → latest testing experiment
```

### Workspace Management Features

- **Clean workspace**: `runs/` shows recent training and testing experiments as symlinks
- **Full archive**: `runs_all/` contains all experiments permanently (never auto-deleted)
- **Separate limits**: Configurable limits for training runs (`maxTrainRuns`) and testing runs (`maxTestRuns`)
- **Latest symlinks**: `runs/latest_train` and `runs/latest_test` always point to most recent experiments
- **Automatic cleanup**: Old symlinks are removed automatically to maintain configured limits
- **Smart checkpoint resolution**: CLI tools find checkpoints in both locations automatically

## Monitoring Training

### TensorBoard

Training metrics are automatically logged to TensorBoard:

```bash
tensorboard --logdir runs/
```

### Console Output

The training script prints statistics every `log_interval` episodes:
- Average reward
- Episode length
- Learning rate
- Policy loss
- Value loss
- Entropy

## Using Custom Reward Functions

To use custom reward functions in your training:

1. **Configure reward weights** in your task config file:
   ```yaml
   # In your custom task config: my_custom_task.yaml
   reward:
     # Standard rewards
     alive: 1.0
     height_safety: -10.0
     # Custom task-specific rewards
     my_custom_reward: 2.0
     distance_penalty: -0.5
   ```

2. **Implement the reward logic** in your task class - see the [Task Creation Guide](docs/guide-task-creation.md) for detailed implementation examples and [System Architecture](docs/ARCHITECTURE.md) for component interaction patterns.

> **Fail-Fast Training:** The training script performs extensive configuration checks before launching Isaac Sim. If it detects an error, it will exit immediately. This 'fail-fast' approach saves you from waiting for the simulation to load only to have it crash seconds later due to a simple typo in a config file.

## Troubleshooting Training Issues

**Enable Debug Logging:** For detailed troubleshooting, enable debug logging across the entire system:
```bash
python train.py logLevel=debug
```
This will show detailed information from both the training script and the environment/simulation.

For comprehensive troubleshooting including component initialization, reward system problems, action space mismatches, and performance issues, see the **[Troubleshooting Guide](docs/TROUBLESHOOTING.md)**.

For general rl_games troubleshooting, see [rl_games documentation](https://github.com/Denys88/rl_games).

## Organizing Your Experiments

### Pinning Important Experiments

When you find an experiment worth keeping permanently visible:

```bash
# After training completes, pin your favorite experiment
mv runs/BoxGrasping_excellent_results_20240101_120000 runs/pinned/

# Or give it a meaningful name when pinning
mv runs/BoxGrasping_train_20240101_120000 runs/pinned/best_grasping_model
```

**Benefits of pinning:**
- Pinned experiments remain in `runs/pinned/` permanently
- They don't get moved to archive or cleaned up automatically
- Easy access to your best models for testing and comparison
- Survives workspace cleanup when you exceed the recent runs limit

### Workspace Cleanup

The system automatically maintains a clean workspace:

- Only 10 most recent experiments shown in `runs/` (configurable via `experiment.maxRecentRuns`)
- Older experiments automatically moved to archive symlinks
- All experiment data preserved permanently in `runs_all/`
- Pinned experiments in `runs/pinned/` never auto-cleaned

### Finding Old Experiments

All experiments are always preserved and accessible:

```bash
# Search in full archive
ls runs_all/ | grep BoxGrasping

# Smart checkpoint resolution works with archived experiments
python train.py test=true checkpoint=runs_all/old_experiment_20231201_100000

# Latest checkpoint resolution searches both locations automatically
python train.py test=true checkpoint=latest
```

### Disabling Workspace Management

To use legacy behavior (all experiments directly in `runs/`):

```bash
python train.py experiment.useCleanWorkspace=false
```
