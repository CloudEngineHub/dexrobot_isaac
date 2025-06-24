# Training DexHand RL Policies

This document describes how to train reinforcement learning policies for the DexHand environment using the rl_games library.

## Quick Start

### Basic Training

To start training a policy with default settings:

```bash
python train.py --task BaseTask --num-envs 1024 --headless
```

### Training with Visualization

To train while viewing the simulation:

```bash
python train.py --task BaseTask --num-envs 64
```

Note: Using fewer environments when visualizing improves performance.

## Command Line Arguments

### Environment Arguments

- `--task`: Task to train on (default: "BaseTask")
  - Available tasks: `BaseTask`, `DexGrasp`
- `--num-envs`: Number of parallel environments (default: 1024)
- `--sim-device`: Device for physics simulation (default: "cuda:0")
- `--rl-device`: Device for RL algorithm (default: "cuda:0")
- `--graphics-device-id`: Graphics device ID (default: 0)
- `--headless`: Run without visualization

### Training Arguments

- `--train-config`: Path to training configuration file (default: "dexhand_env/cfg/train/BaseTaskPPO.yaml")
- `--seed`: Random seed, use -1 for random (default: 42)
- `--torch-deterministic`: Use deterministic algorithms for reproducibility
- `--checkpoint`: Path to checkpoint to resume from
- `--test`: Run in test mode (no training)
- `--max-iterations`: Maximum training iterations (default: 10000)

### Logging Arguments

- `--experiment-name`: Name for the experiment (auto-generated if not provided)
- `--log-interval`: Logging interval in episodes (default: 10)

## Examples

### Resume Training from Checkpoint

```bash
python train.py --task BaseTask --checkpoint runs/BaseTask_20240101_120000/nn/checkpoint_1000.pth
```

### Test a Trained Policy

```bash
python train.py --task BaseTask --test --checkpoint runs/BaseTask_20240101_120000/nn/checkpoint_best.pth
```

### Multi-GPU Training

For multi-GPU training, use PyTorch's distributed launch:

```bash
python -m torch.distributed.launch --nproc_per_node=2 train.py --task BaseTask --num-envs 2048 --headless
```

## Training Configuration

The PPO hyperparameters are defined in YAML configuration files located in `dexhand_env/cfg/train/`.

### Key Hyperparameters

- `learning_rate`: Learning rate for the optimizer (default: 3e-4)
- `gamma`: Discount factor (default: 0.99)
- `horizon_length`: Number of steps before update (default: 16)
- `minibatch_size`: Minibatch size for PPO updates (default: 8192)
- `mini_epochs`: Number of PPO epochs per update (default: 8)

### Creating Custom Training Configs

To create a custom training configuration:

1. Copy an existing config file:
   ```bash
   cp dexhand_env/cfg/train/BaseTaskPPO.yaml dexhand_env/cfg/train/MyTaskPPO.yaml
   ```

2. Modify the hyperparameters as needed

3. Use it for training:
   ```bash
   python train.py --task MyTask --train-config dexhand_env/cfg/train/MyTaskPPO.yaml
   ```

## Output Structure

Training outputs are saved in the `runs/` directory:

```
runs/
└── BaseTask_20240101_120000/
    ├── args.yaml           # Command line arguments
    ├── train_config.yaml   # Training configuration
    ├── train.log          # Training logs
    └── nn/                # Neural network checkpoints
        ├── checkpoint_100.pth
        ├── checkpoint_200.pth
        └── checkpoint_best.pth
```

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

## Customizing Rewards

To customize the reward function for your task:

1. Edit your task class (e.g., `BaseTask` or `DexGraspTask`)
2. Implement the `compute_task_rewards()` method
3. Configure reward weights in the task config file

Example reward implementation:

```python
def compute_task_rewards(self):
    # Example: reward for keeping hand at a certain height
    hand_height = self.hand_positions[:, 2]  # Z coordinate
    height_reward = torch.exp(-torch.abs(hand_height - 0.5))

    return {"height": height_reward}
```

## Troubleshooting

### Out of Memory

If you encounter GPU memory issues:
- Reduce `--num-envs`
- Reduce `minibatch_size` in the training config
- Use `--headless` mode

### Slow Training

To improve training speed:
- Use `--headless` mode
- Increase `--num-envs` (if GPU memory allows)
- Use GPU simulation: `--sim-device cuda:0`

### Diverging Training

If training becomes unstable:
- Reduce `learning_rate`
- Reduce `horizon_length`
- Check reward scaling in your task

## Next Steps

1. Implement task-specific rewards in your task class
2. Tune hyperparameters for your specific task
3. Add domain randomization for sim-to-real transfer
4. Implement curriculum learning for complex tasks
