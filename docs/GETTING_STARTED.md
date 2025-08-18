# Getting Started with DexHand

This guide will get you from zero to your first trained RL policy in under 10 minutes.

## Prerequisites

- NVIDIA GPU with CUDA support
- Python 3.8+
- Isaac Gym Preview 4 (see [installation instructions](https://developer.nvidia.com/isaac-gym))

## Quick Setup (5 minutes)

### 1. Install Isaac Gym
```bash
# Download Isaac Gym Preview 4 from NVIDIA
# Follow their installation instructions, then verify:
cd isaacgym/python/examples
python joint_monkey.py  # Should show a working simulation
```

### 2. Clone and Install DexHand
```bash
git clone --recursive https://github.com/dexrobot/dexrobot_isaac
cd dexrobot_isaac
pip install -e .
```

> **Missing submodules?** Run `git submodule update --init --recursive` to fetch required robot models.

### 3. Verify Installation
```bash
# Quick test (should show hand visualization)
python examples/dexhand_test.py --num-envs 1 --episode-length 100
```

You should see an Isaac Gym window with a dexterous hand in the simulation.

## Your First Training (3 minutes)

### 1. Start Training
```bash
# Train a basic policy
python train.py task=BaseTask numEnvs=512
```

This creates a new experiment in `runs/BaseTask_train_YYMMDD_HHMMSS/` and begins training.

### 2. Test Your Trained Policy
```bash
# Test with visualization
python train.py task=BaseTask test=true checkpoint=latest viewer=true numEnvs=4
```

The system automatically finds your latest training checkpoint and visualizes the learned policy.

## Next Steps

- **[Training Guide](TRAINING.md)** - Comprehensive training workflows, testing options, and experiment management
- **[Task Creation Guide](guide-task-creation.md)** - Create custom manipulation tasks
- **[Troubleshooting](TROUBLESHOOTING.md)** - Solutions for common setup and runtime issues
- **[System Architecture](ARCHITECTURE.md)** - Understanding the component-based design

## Quick Troubleshooting

- **ImportError: isaacgym** → Isaac Gym not installed correctly
- **CUDA out of memory** → Reduce `numEnvs` (try 256 or 128)
- **Missing assets/dexrobot_mujoco** → Run `git submodule update --init --recursive`
- **No checkpoints found** → Training hasn't saved checkpoints yet (wait longer)

For detailed troubleshooting, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
