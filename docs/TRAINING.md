# Training and Testing Guide

Get your RL policies training in 30 seconds. This guide covers the essentials - for deep configuration details, see [guide-configuration-system.md](guide-configuration-system.md).

## Quick Start

### 1. Basic Training

```bash
python train.py config=train_headless task=BlindGrasping
```

**What this does:**
- `config=train_headless` → Loads `train_headless.yaml` which sets:
  - Training mode with PPO algorithm
  - 8192 parallel environments for fast training
  - No viewer (headless) for maximum performance
- `task=BlindGrasping` → Loads `BlindGrasping.yaml` which defines:
  - Task-specific rewards, observations, and termination criteria
  - Episode length, control mode, and physics settings

### 2. Basic Testing

```bash
python train.py config=test_viewer task=BlindGrasping checkpoint=latest
```

**What this does:**
- `config=test_viewer` → Loads `test_viewer.yaml` which sets:
  - Test mode (`train.test=true`)
  - 4 environments for smooth visualization
  - Isaac Gym viewer enabled
- `task=BlindGrasping` → Same task configuration as training
- `checkpoint=latest` → Automatically finds your most recent training run:
  - Reads `runs/latest_train` symlink
  - Loads newest checkpoint from `nn/` subdirectory
  - Hot-reloads new checkpoints every 30 seconds during training

**Alternative test configs:**
- `config=test_stream` - HTTP video streaming (access at `http://localhost:58080`)
- `config=test_record` - Save videos to disk

### 3. Training with Simple Customization

```bash
python train.py config=train_headless task=BlindGrasping numEnvs=1024
```

**CLI overrides:** Any config value can be overridden directly. Common aliases:

| Alias | Full Path | Example |
|-------|-----------|---------|
| `numEnvs` | `env.numEnvs` | `numEnvs=2048` |
| `viewer` | `env.viewer` | `viewer=true` |
| `maxIterations` | `train.maxIterations` | `maxIterations=5000` |
| `seed` | `train.seed` | `seed=123` |
| `device` | `env.device` | `device=cuda:1` |
| `testGamesNum` | `train.testGamesNum` | `testGamesNum=0` (indefinite) |
| `checkpoint` | `train.checkpoint` | `checkpoint=runs/my_experiment/nn/last.pth` |

### 4. Full Customization

```bash
python train.py config=train_headless task=BlindGrasping task.reward_weights.object_height=0.5
```

**Nested overrides:** Use dot notation to override any nested config value:
- `task.reward_weights.object_height=0.5` - Adjust specific reward weight
- `sim.physx.num_position_iterations=32` - Increase physics accuracy
- `task.episodeLength=1000` - Longer episodes
- `env.box.size=0.08` - Larger grasping object

This enables controlled experiments - change one parameter at a time to measure impact.

## How the Config System Works

The configuration uses [Hydra](https://hydra.cc/) with a clean hierarchy:

```
config=train_headless  →  Selects base configuration preset
task=BlindGrasping     →  Loads task-specific settings
numEnvs=1024          →  CLI override (alias expanded to env.numEnvs)
```

**Config composition order:**
1. Base `config.yaml` (defaults)
2. Selected config file (e.g., `train_headless.yaml`)
3. Task file (e.g., `task/BlindGrasping.yaml`)
4. CLI overrides (highest priority)

**The checkpoint=latest resolution process:**
1. `checkpoint=latest` → defaults to `checkpoint=latest_train`
2. Follows `runs/latest_train` symlink to actual experiment directory
3. Searches `nn/` subdirectory for newest `.pth` file
4. Returns full path to that checkpoint file

**Other checkpoint shortcuts:**
- `checkpoint=latest_test` - finds most recent test run
- `checkpoint=runs/BlindGrasping_train_20240101` - auto-finds `.pth` in directory
- `checkpoint=runs/experiment/nn/specific.pth` - direct path to file

**With hot-reload enabled:** The system monitors the resolved directory every 30 seconds (configurable via `reloadInterval`) and automatically loads newer checkpoints as they appear

See [guide-indefinite-testing.md](guide-indefinite-testing.md) for continuous monitoring setup.

## Monitoring Training Progress

### TensorBoard Visualization
```bash
# Start TensorBoard in another terminal
tensorboard --logdir runs/
```

Navigate to `http://localhost:6006` to see training curves. The system logs:
- Episode rewards and individual reward components
- Training loss and learning rate
- Success/failure rates (task-specific)

## Common Training Scenarios

### Debug Training
```bash
# Few environments with visualization
python train.py config=train_headless task=BlindGrasping numEnvs=4 viewer=true
```

### Fast Iteration
```bash
# Quick training runs for testing changes
python train.py config=train_headless task=BlindGrasping maxIterations=100 numEnvs=256
```

### Production Training
```bash
# Maximum performance with accurate physics
python train.py config=train_headless task=BlindGrasping numEnvs=8192 +defaults=[config,/physics/accurate]
```

### Continuous Monitoring
```bash
# Terminal 1: Training
python train.py config=train_headless task=BlindGrasping

# Terminal 2: Live testing with auto-reload
python train.py config=test_viewer task=BlindGrasping checkpoint=latest testGamesNum=0
```

### Experiment Comparison
```bash
# Run 1: Baseline
python train.py config=train_headless task=BlindGrasping

# Run 2: Higher reward weight
python train.py config=train_headless task=BlindGrasping task.reward_weights.object_height=2.0

# Run 3: Different physics
python train.py config=train_headless task=BlindGrasping +defaults=[config,/physics/accurate]
```

## Test Mode Options

### Interactive Viewer
```bash
python train.py config=test_viewer task=BlindGrasping checkpoint=latest
# Keyboard controls available, 4 environments, local display
```

### HTTP Streaming
```bash
python train.py config=test_stream task=BlindGrasping checkpoint=latest streamBindAll=true
# Access at http://server-ip:58080, no local display needed
```

### Video Recording
```bash
python train.py config=test_record task=BlindGrasping checkpoint=latest
# Saves MP4 files to runs/{experiment}/videos/
```

## Experiment Management

The system uses intelligent workspace management to keep experiments organized:

```
runs/                          # Clean workspace (recent runs + symlinks)
├── BlindGrasping_train_...    # → symlink to runs_all/
├── latest_train               # → always points to newest training
└── latest_test                # → always points to newest test

runs_all/                      # Permanent archive (never deleted)
└── BlindGrasping_train_...    # Actual experiment data
    ├── config.yaml
    ├── train.log
    └── nn/
        └── checkpoint_*.pth
```

**Key features:**
- **Auto-cleanup**: Only keeps 10 recent runs visible in `runs/` (configurable)
- **Permanent archive**: All experiments preserved in `runs_all/`
- **Latest symlinks**: `checkpoint=latest` uses `runs/latest_train`
- **Smart resolution**: CLI finds checkpoints in both locations automatically

### Pinning Important Experiments

Keep your best models easily accessible:

```bash
# Pin a successful experiment
mv runs/BlindGrasping_excellent_20240101_120000 runs/pinned/best_model

# Create shortcut without moving (using cp -P to copy symlink)
cp -P runs/BlindGrasping_train_20240101_120000 runs/pinned/paper_results
```

### Finding Old Experiments

```bash
# Search archive
ls runs_all/ | grep BlindGrasping

# Use archived checkpoints directly
python train.py test=true checkpoint=runs_all/old_experiment_20231201/nn/checkpoint.pth
```

## Repository-Specific Features

### CLI Aliases System
This repository defines custom CLI aliases in `CLIPreprocessor.ALIASES` for convenience:
- `numEnvs` → expands to `env.numEnvs`
- `checkpoint` → expands to `train.checkpoint` with smart resolution
- `config` → converted to `--config-name` for Hydra

### Checkpoint Resolution Logic
The `checkpoint=latest` uses intelligent resolution:
1. Follows `runs/latest_train` symlink
2. Finds newest `.pth` file in `nn/` subdirectory
3. Supports hot-reload for continuous monitoring

### Experiment Management Structure
Unique dual-directory system:
- `runs/` - Clean workspace with symlinks (auto-cleanup after 10 runs)
- `runs_all/` - Permanent archive of all experiments
- Latest symlinks always point to newest runs

## Tips

1. **Use task-specific configs**: Each task has optimized defaults in `task/*.yaml`
2. **Monitor with hot-reload**: `testGamesNum=0` with `reloadInterval=30` for live updates
3. **Check resolved config**: Run with `--cfg job` to see final configuration
4. **Use physics presets**: `/physics/fast` for debugging, `/physics/accurate` for training

## Related Documentation

- [Configuration System](guide-configuration-system.md) - Full configuration reference
- [Indefinite Testing](guide-indefinite-testing.md) - Continuous policy monitoring
- [Task Creation](guide-task-creation.md) - Create custom tasks
- [Physics Tuning](guide-physics-tuning.md) - Optimize simulation settings
