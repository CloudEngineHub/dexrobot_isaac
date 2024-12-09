# GALAG-DexHand

This repository provides a framework for training dexterous grasping policies for robotic hands using NVIDIA's Isaac Gym simulator.

### Installation

1. Download and install Isaac Gym Preview 4 from NVIDIA's website

2. Verify Isaac Gym installation:

```bash
cd isaac-gym/python/examples
python joint_monkey.py
```

3. Clone and install this repository:

```bash
git clone https://github.com/lei00764/GALAG-DexHand
cd GALAG-DexHand
pip install -e .
```

## Running

### Training

```bash
python DexHandEnv/train.py task=DexHand num_envs=4096 headless=True
```
- `num_envs`: Number of parallel environments (default: 4096)
- `headless`: Run without visualization for faster training

### Testing

To test a trained model:

```bash
python DexHandEnv/train.py task=DexHand test=True num_envs=2 checkpoint=runs/DexHand_*/nn/DexHand.pth
python DexHandEnv/train.py task=DexHand test=True num_envs=2 checkpoint=runs/DexHand_02-17-39-23/nn/DexHand.pth
```

### Configuration

The environment and training parameters can be customized through config files:

- Environment config: `DexHandEnv/config/task/DexHand.yaml`
- Training config: `DexHandEnv/config/train/DexHandPPO.yaml`

### Video Recording

To capture training videos:

```bash
python DexHandEnv/train.py task=DexHand capture_video=True capture_video_freq=1500 capture_video_len=100
```

### Multi-GPU Training

For distributed training across multiple GPUs:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=2 DexHandEnv/train.py multi_gpu=True task=DexHand
```

## License

This project is licensed under the MIT License.

## Acknowledgements

This work builds upon the Isaac Gym framework developed by NVIDIA.
