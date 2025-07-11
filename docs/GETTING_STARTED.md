# Getting Started with DexHand

This guide will get you from zero to your first trained RL policy in under 10 minutes.

**Next Steps:**
- [TRAINING.md](../TRAINING.md) - Complete training and testing guide
- [Component Initialization](guide-component-initialization.md) - Understanding the system architecture
- [Task Creation Guide](guide-task-creation.md) - Creating custom manipulation tasks
- [Observation System](guide-observation-system.md) - Customizing observations for your task

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

> **Already cloned without `--recursive`?** Run `git submodule update --init --recursive` before `pip install -e .` to fetch the required robot models.

### 3. Verify Installation
```bash
# Quick test (should show hand visualization)
python examples/dexhand_test.py --num-envs 1 --episode-length 100
```

**✅ Success Criteria:**
- Isaac Gym window opens showing a single dexterous hand
- Hand responds to keyboard controls (WASD for movement, Arrow keys for fingers)
- No error messages in terminal
- Window remains stable for 30+ seconds

**❌ Troubleshooting:**
- **"ImportError: isaacgym"** → Isaac Gym not installed correctly
- **"CUDA error"** → Check GPU drivers and CUDA installation
- **"No such file or directory"** → Run `git submodule update --init --recursive`

## Your First Training (3 minutes)

### 1. Start Training
```bash
# Train a basic policy (runs headless for speed)
python train.py task=BaseTask numEnvs=512

# Or use the default configuration
python train.py
```

This will:
- Create a new experiment in `runs/BaseTask_train_YYMMDD_HHMMSS/`
- Train a policy to control the dexterous hand
- Save checkpoints every few iterations

**✅ Success Criteria:**
- Training starts without errors
- Console shows "Starting training" message
- Reward values begin updating (may start negative)
- Files appear in `runs/BaseTask_train_YYMMDD_HHMMSS/`

**❌ Troubleshooting:**
- **"CUDA out of memory"** → Reduce `numEnvs` (try 256 or 128)
- **"RuntimeError: physics_manager"** → Initialization error, check Isaac Gym installation
- **"Config not found"** → Check you're in the correct directory
- Display training progress

### 2. Monitor Progress
```bash
# In another terminal, start TensorBoard
tensorboard --logdir runs/
```

Navigate to `http://localhost:6006` to see training curves.

**✅ Success Criteria:**
- TensorBoard opens in browser
- Training curves show improvement over time
- Episode reward generally increases (may be noisy)

### 3. Test Your Trained Policy
```bash
# Test with visualization (finds latest checkpoint automatically)
python train.py task=BaseTask test=true checkpoint=latest render=true numEnvs=4
```

**✅ Success Criteria:**
- Isaac Gym window opens showing 4 hands
- Hands move smoothly and maintain stability
- No erratic or unstable behavior
- Policy runs for full episode length

**❌ Troubleshooting:**
- **"No checkpoints found"** → Training hasn't created checkpoints yet (wait longer)
- **"Checkpoint not found"** → Use specific path: `checkpoint=runs/BaseTask_train_YYMMDD_HHMMSS/nn/BaseTask.pth`
- **Unstable behavior** → Normal for early training; train longer for better policy

## What Just Happened?

You've successfully:
1. ✅ **Trained an RL policy** to control a 23-DOF dexterous hand
2. ✅ **Monitored training progress** with TensorBoard
3. ✅ **Visualized the trained policy** in Isaac Gym

The BaseTask teaches the hand basic movement and stability without a specific manipulation goal.

## Next Steps

### Try Different Tasks
```bash
# Train on box grasping (more complex)
python train.py task=BoxGrasping numEnvs=1024

# Test the grasping policy
python train.py task=BoxGrasping test=true checkpoint=latest render=true
```

**✅ Success Criteria (BoxGrasping):**
- Hand approaches and grasps the box
- Box is lifted above the table
- Stable grasp maintained for several seconds
- Success rate > 70% after sufficient training

### Customize Training
```bash
# Train with more environments for faster learning
python train.py task=BaseTask numEnvs=2048

# Train with different random seed
python train.py task=BaseTask seed=123

# Train for longer
python train.py task=BaseTask maxIter=20000
```

### Create Your Own Task
See the [Task Creation Guide](guide-task-creation.md) for step-by-step instructions on implementing custom manipulation tasks.

## Common Issues and Solutions

### Installation Problems

**Problem**: `ModuleNotFoundError: No module named 'isaacgym'`
**Solution**: Isaac Gym not installed or not in Python path
```bash
# Verify Isaac Gym installation
cd isaacgym/python
pip install -e .
```

**Problem**: `RuntimeError: CUDA out of memory`
**Solution**: Reduce number of environments or use smaller batch sizes
```bash
# Try with fewer environments
python train.py numEnvs=256

# Or use CPU for testing (very slow)
python train.py device=cpu numEnvs=16
```

**Problem**: `FileNotFoundError: [Errno 2] No such file or directory: 'assets/dexrobot_mujoco'`
**Solution**: Initialize git submodules
```bash
git submodule update --init --recursive
```

### Training Issues

**Problem**: Training reward stays negative or doesn't improve
**Solution**: This is normal for early training - be patient
```bash
# Monitor progress with TensorBoard
tensorboard --logdir runs/

# Train for longer (default is 10000 iterations)
python train.py maxIter=20000
```

**Problem**: Training is very slow
**Solution**: Increase parallelization or use headless mode
```bash
# More environments for faster training
python train.py numEnvs=2048

# Ensure headless mode for speed (default in train mode)
python train.py render=false
```

### Testing Issues

**Problem**: Policy behaves erratically during testing
**Solution**: Train longer or check checkpoint
```bash
# Train for more iterations
python train.py maxIter=15000

# Test with specific checkpoint
python train.py test=true checkpoint=runs/BaseTask_train_YYMMDD_HHMMSS/nn/BaseTask.pth
```

**Problem**: `"latest" checkpoint not found`
**Solution**: Use explicit checkpoint path
```bash
# List available checkpoints
ls runs/*/nn/*.pth

# Use specific checkpoint
python train.py test=true checkpoint=runs/BaseTask_train_20250709_123456/nn/BaseTask.pth
```

### Performance Optimization

**For faster training:**
- Use more environments: `numEnvs=2048`
- Ensure headless mode: `render=false`
- Use multiple GPUs if available
- Close unnecessary applications

**For better visualization:**
- Use fewer environments: `numEnvs=64`
- Enable rendering: `render=true`
- Reduce episode length for faster cycles

### Getting Help

If you encounter issues not covered here:
1. Check the [Training Guide](../TRAINING.md) for advanced configuration
2. Review the [System Architecture](ARCHITECTURE.md) for deeper understanding
3. Consult the [API Reference](reference-dof-control-api.md) for technical details
4. Create an issue on the GitHub repository with your error message and system details

## Summary

In just a few commands, you've:
1. Set up a complete RL training environment
2. Trained your first dexterous manipulation policy
3. Visualized the results

The DexHand environment provides a powerful foundation for researching dexterous manipulation with reinforcement learning. The component-based architecture makes it easy to extend with custom tasks, rewards, and observations.

Ready to build something more complex? Check out the [Task Creation Guide](guide-task-creation.md) next!
