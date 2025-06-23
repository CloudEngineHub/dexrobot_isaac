# Isaac Gym Environments for Dexterous Manipulation

This repository provides a framework for training dexterous manipulation policies for robotic hands using NVIDIA's Isaac Gym simulator. It includes two main tasks:

- **DexGrasp**: A task focused on grasping and lifting objects from a surface
- **DexReorient**: A task focused on manipulating objects to match a target orientation

## New Component-Based Architecture

The codebase has been refactored to use a component-based architecture, offering:

- Improved modularity and maintainability
- Clear separation of concerns through specialized components
- A standardized task interface for easier creation of new tasks
- Better code organization and reusability

The new architecture is available in the `dexhand_env` package. For details, see [dexhand_env/README.md](dexhand_env/README.md).

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

### Using the New Architecture

To run a simple test of the dexterous hand visualization:

```bash
# Run the simple dexterous hand visualization
python examples/simple_dexhand_test.py
```

For more options, see [examples/README_simple_dexhand_test.md](examples/README_simple_dexhand_test.md).

### Training with the New Architecture

Refer to the documentation in the [dexhand_env](dexhand_env/README.md) directory for instructions on training with the new component-based architecture.

### Legacy Implementation

The original implementation has been moved to the `legacy` directory and is kept for reference purposes only. We recommend using the new component-based architecture for all new development.

### Configuration

The environment and training parameters can be customized through config files in the new architecture:

- See the configuration files in the `dexhand_env/cfg` directory

### Video Recording and Multi-GPU Training

The new architecture supports video recording and multi-GPU training. Refer to the documentation in the [dexhand_env](dexhand_env/README.md) directory for details.

## Code Structure

The repository contains two implementations of the dexterous manipulation environment:

1. **Legacy Implementation**: The original monolithic implementation
   - Located in the `legacy/DexHandEnv_obsolete` directory
   - Kept for reference purposes only

2. **Current Implementation (dexhand_env)**: New component-based architecture
   - Located in the `dexhand_env` directory
   - Features modular components and a standardized task interface
   - Provides improved functionality with better code organization

3. **Simple Visualization Example**:
   - Located in the `examples` directory
   - Provides a minimal example to visualize the dexterous hand

## License

This project is licensed under the Apache License.

## Acknowledgements

This work builds upon the Isaac Gym framework developed by NVIDIA. The refactored architecture was developed by DexRobot Inc.
