# Isaac Gym Environments for Dexterous Manipulation

This repository provides a framework for training dexterous manipulation policies for robotic hands using NVIDIA's Isaac Gym simulator. It includes two main tasks:

- **DexGrasp**: A task focused on grasping and lifting objects from a surface
- **DexReorient**: A task focused on manipulating objects to match a target orientation

### Installation

1. Download and install [Isaac Gym Preview 4](https://developer.nvidia.com/isaac-gym) from NVIDIA's website

2. Verify Isaac Gym installation:

```bash
cd isaac-gym/python/examples
python joint_monkey.py
```

3. Clone and install this repository:

```bash
git clone https://github.com/dexrobot/dexrobot_isaac
cd dexrobot_isaac
pip install -e .
```

## Running

### Training

```bash
python DexHandEnv/train.py task=DexGrasp num_envs=4096 headless=True
```
- `num_envs`: Number of parallel environments (default: 4096)
- `headless`: Run without visualization for faster training

### Testing

To test a trained model:

```bash
python DexHandEnv/train.py task=DexGrasp test=True num_envs=2 checkpoint=runs/DexGrasp_*/nn/DexGrasp.pth
python DexHandEnv/train.py task=DexGrasp test=True num_envs=2 checkpoint=runs/DexGrasp_2023-04-11-14-42-29/nn/DexGrasp.pth
```

### Configuration

The environment and training parameters can be customized through config files:

- Environment configs: `DexHandEnv/cfg/task/DexGrasp.yaml` and `DexHandEnv/cfg/task/DexReorient.yaml`
- Training configs: `DexHandEnv/cfg/train/DexGraspPPO.yaml` and `DexHandEnv/cfg/train/DexReorientPPO.yaml`

### Video Recording

To capture training videos:

```bash
python DexHandEnv/train.py task=DexReorient capture_video=True capture_video_freq=1500 capture_video_len=100
```

### Multi-GPU Training

For distributed training across multiple GPUs:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=2 DexHandEnv/train.py multi_gpu=True task=DexReorient
```

## License

This project is licensed under the Apache License.

## Acknowledgements

This work builds upon the Isaac Gym framework developed by NVIDIA.
