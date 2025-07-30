# DexHand Examples

This directory contains example scripts for testing and demonstrating the DexHand environment.

## dexhand_test.py

Comprehensive test script for the DexHand environment using BaseTask only.

### Usage

The test script is hardcoded to use BaseTask and supports Hydra configuration overrides:

```bash
# Basic test (BaseTask only)
python examples/dexhand_test.py

# Headless test with custom parameters
python examples/dexhand_test.py headless=true steps=100 env.numEnvs=16

# Test with different control modes
python examples/dexhand_test.py env.controlMode=position_delta env.policyControlsHandBase=false

# Debug mode with verbose logging
python examples/dexhand_test.py debug=true log_level=debug

# Enable video recording and plotting
python examples/dexhand_test.py env.videoRecord=true enablePlotting=true
```

### Configuration Parameters

All configuration parameters can be overridden via command line:

**Test Settings:**
- `steps` (1200) - Total number of test steps to run
- `sleep` (0.01) - Sleep time between steps in seconds
- `device` ("cuda:0") - Device for simulation and RL
- `headless` (false) - Run without GUI visualization
- `debug` (false) - Enable debug output and additional logging
- `log_level` ("info") - Logging level (debug/info/warning/error)

**Environment Settings:**
- `env.numEnvs` (1024) - Number of parallel environments
- `env.controlMode` ("position") - Control mode (position/position_delta)
- `env.policyControlsHandBase` (true) - Include hand base in policy action space
- `env.policyControlsFingers` (true) - Include fingers in policy action space

**Recording & Visualization:**
- `env.videoRecord` (false) - Enable video recording (works in headless mode)
- `enablePlotting` (false) - Enable real-time plotting with Rerun
- `plotEnvIdx` (0) - Environment index to plot

**Task Selection:**
- Script is hardcoded to use BaseTask only (basic task with contact test boxes)

### Key Features

1. **BaseTask Focus**: Hardcoded to BaseTask for reliable and consistent testing of core functionality

2. **Action Verification**: Tests all DOF mappings with sequential action patterns to verify control modes

3. **Performance Profiling**: Optional timing analysis for step processing components

4. **Real-time Plotting**: Integration with Rerun for visualization (when available)

5. **Video Recording**: Supports video capture in both windowed and headless modes

6. **Comprehensive Logging**: Detailed information about environment setup, action mappings, and system state

### Keyboard Controls (Non-headless mode)

- `SPACE` - Toggle random actions mode
- `E` - Reset current environment
- `G` - Toggle between single robot and global view
- `UP/DOWN` - Navigate between robots
- `ENTER` - Toggle camera view mode
- `C` - Toggle contact force visualization

### Configuration System

The test script inherits configuration from the main Hydra config system:

- Base configuration: `dexhand_env/cfg/config.yaml`
- Task configuration: `dexhand_env/cfg/task/BaseTask.yaml` (hardcoded)
- Physics configurations: `dexhand_env/cfg/physics/default.yaml`

For testing other tasks, use the training script (`train.py`) instead.
