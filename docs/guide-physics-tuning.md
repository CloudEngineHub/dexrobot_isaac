# Physics Parameter Tuning Guide

This guide explains how to tune physics parameters for stable simulation, particularly for GPU pipeline with rendering.

**Related Documentation:**
- [Configuration System Guide](guide-configuration-system.md) - Modular physics configuration system
- [System Architecture](ARCHITECTURE.md) - Overall system design principles

## Problem Symptoms

### Drift Issues
- Fingers drift when no actions applied
- Hand position changes without commanded motion
- Joints don't hold position under zero action

### Performance Issues
- Simulation runs slowly
- CUDA memory errors
- Unstable contact behavior

## Modular Physics Configuration System

The DexHand system provides three physics configurations optimized for different scenarios:

### Available Physics Configurations

**Location:** `dexhand_env/cfg/physics/`

#### `default.yaml` - Balanced Performance
```yaml
# @package sim
substeps: 4                     # Standard physics substeps
gravity: [0.0, 0.0, -9.81]
physx:
  solver_type: 1                # TGS solver
  num_position_iterations: 16   # Balanced precision
  contact_offset: 0.001         # High precision detection
  rest_offset: 0.0005          # Stability maintenance
  # ... additional parameters
```

**Use cases:** BaseTask, general development, balanced quality/performance

#### `fast.yaml` - Real-time Visualization
```yaml
# Inherits from default.yaml and overrides for speed
defaults: [default, _self_]

substeps: 2                     # Reduced substeps for speed
physx:
  num_position_iterations: 8    # Fewer iterations for speed
  contact_offset: 0.002         # Slightly relaxed precision
```

**Use cases:** `test_viewer`, `test_stream`, interactive debugging (~2-3x faster)

#### `accurate.yaml` - High Precision Training
```yaml
# Inherits from default.yaml and overrides for precision
defaults: [default, _self_]

substeps: 16                    # Many substeps for stability
physx:
  num_position_iterations: 32   # High iteration count for precision
```

**Use cases:** Complex contact scenarios requiring high precision (~2-3x slower but higher quality)

### Selecting Physics Configurations

#### In Task Files
```yaml
# Example: Custom task with high-precision physics
defaults:
  - BaseTask
  - /physics/accurate      # Use high-precision physics
  - _self_

# Override timing for task-specific needs
sim:
  dt: 0.01                 # Task-specific control frequency
```

#### In Test Configurations
```yaml
# dexhand_env/cfg/test_viewer.yaml
defaults:
  - config
  - base/test
  - /physics/fast          # Fast physics for smooth rendering
  - _self_
```

#### Via CLI Override
```bash
# Select physics configuration at runtime
python train.py +defaults=[config,/physics/accurate]
python train.py +defaults=[config,/physics/fast] viewer=true
```

See the [Configuration System Guide](guide-configuration-system.md) for detailed customization patterns.

## Key Parameters Explained

### Solver Iterations
**`num_position_iterations: 6`**
- Controls constraint solving accuracy
- Reduced from previous default (32) due to smaller timestep
- 200 Hz physics allows fewer iterations for same quality
- Trade-off between accuracy and computational cost

### Timestep and Substeps
**`dt: 0.005` (200Hz)**
- High-fidelity simulation timestep for dexterous manipulation
- Smaller timesteps improve numerical stability and contact resolution
- Excellent for capturing rapid finger-object interactions
- Control frequency auto-detected after physics measurement

**`substeps: 1`**
- Reduced from previous default (4) due to higher frequency
- Higher physics frequency reduces need for substeps
- Optimal balance for 200 Hz simulation

### Contact Parameters

**`contact_offset` and `rest_offset`** (Critical for Stability)
- **contact_offset**: Distance at which contact constraints are generated
- **rest_offset**: Distance at which objects come to rest
- **IMPORTANT**: `rest_offset` must be smaller than `contact_offset`
- If `rest_offset >= contact_offset`, objects oscillate as contacts are repeatedly lost and regenerated

**Example (Box Oscillation Fix):**
```yaml
contact_offset: 0.001    # Contacts detected at 1mm
rest_offset: 0.0005     # Objects rest at 0.5mm (maintains contact)
```

**`contact_collection: 1`**
- CC_LAST_SUBSTEP mode required for GPU pipeline
- Different modes affect contact behavior
- Keep at 1 when using GPU pipeline

**`gpu_contact_pairs_per_env: 1024`**
- Contact pairs allocated per environment
- Automatically scaled by number of environments
- Increase if you see contact buffer overflow warnings

## Tuning Process

### 1. Start with Default Configuration
Use the settings above as your baseline.

### 2. Test Configuration
```bash
python examples/dexhand_test.py --episode-length 50
```

Observe:
- Joint position stability over time
- Computational performance (FPS)
- Any error messages or warnings

### 3. Iterative Tuning

**For stability issues:**
```yaml
# Increase solver iterations (default: 16 → 32)
physx:
  num_position_iterations: 32

# Decrease timestep (default: 0.01 → 0.005)
sim:
  dt: 0.005

# Increase substeps (default: 4 → 8)
sim:
  substeps: 8
```

**For performance issues:**
```yaml
# Use fast physics configuration
defaults: [/physics/fast, _self_]

# Or manually reduce solver quality
physx:
  num_position_iterations: 8  # Reduced from 16
```

**For memory issues:**
```bash
# Monitor GPU memory during execution
watch -n 1 nvidia-smi

# Reduce environments if memory constrained
python train.py env.numEnvs=512  # Instead of 8192
```

### 4. Validate with Repository Test Commands
```bash
# Test stability with BaseTask
python examples/dexhand_test.py task=BaseTask --episode-length 100

# Test with BlindGrasping task (more complex contacts)
python train.py task=BlindGrasping test=true numEnvs=4 viewer=true

# Benchmark performance
python train.py task=BaseTask numEnvs=8192 headless=true maxIterations=100
```

## Physics Pipeline Specifics

### GPU vs CPU Pipeline Differences
The repository uses GPU pipeline by default with specific requirements:
- `contact_collection: 1` (CC_LAST_SUBSTEP) required for GPU
- Contact buffer automatically scaled: `gpu_contact_pairs_per_env × numEnvs`
- GPU pipeline enforces specific tensor refresh order

### Viewer Impact on Physics
When `viewer=true`, Isaac Gym automatically:
- Calls `gym.sync_frame_time()` for real-time synchronization
- Limits simulation speed to match real-time for smooth visualization
- Shows high rendering % in profiler when waiting for real-time sync (expected behavior)
- For training without visualization, use `headless=true` to run at maximum speed

## Example Parameter Ranges

### For Higher Stability (if needed)
Consider trying:
- Smaller `dt` (e.g., 0.0025, 0.002) - Ultra-high frequency
- More `substeps` (e.g., 2, 3) - Add substeps if needed
- Higher `num_position_iterations` (e.g., 8, 12) - Increase solver quality

### For Better Performance (trading stability)
Consider trying:
- Larger `dt` (e.g., 0.01, 0.0167) - Lower frequency
- Keep `substeps: 1` - Already minimal
- Lower `num_position_iterations` (e.g., 4, 3) - Reduce solver quality


## Verification Commands

### Basic Stability Test
```bash
python examples/dexhand_test.py --episode-length 50
```

### GPU Pipeline Test
```bash
python examples/dexhand_test.py --use-gpu-pipeline --steps 100
```

### Performance Benchmark
```bash
python examples/dexhand_test.py --episode-length 1000 --num-envs 128
```

## Interpreting Results

### Metrics to Monitor
- DOF position changes over time (watch for drift)
- Physics step timing (check FPS counter)
- Contact buffer warnings in console output
- Joint position stability

## Common Issues and Solutions

### Box/Object Oscillation on Ground
**Symptom**: Objects oscillate or jitter when resting on surfaces

**Solution**: Ensure `rest_offset < contact_offset`
```yaml
# Correct configuration
contact_offset: 0.001    # Larger value
rest_offset: 0.0005     # Smaller value

# Incorrect (causes oscillation)
contact_offset: 0.001
rest_offset: 0.003      # ERROR: Larger than contact_offset!
```

**Explanation**: When `rest_offset >= contact_offset`, objects rest at a distance where contacts aren't detected, causing them to fall slightly. This creates a cycle of contact detection → separation → contact loss → falling, resulting in oscillation.

### Joint Drift
**Symptom**: Joints slowly drift from commanded positions

**Solution**:
- Increase `num_position_iterations` (try 8-16)
- Reduce `dt` for finer time resolution
- Check joint damping values in MJCF

### Contact Instability
**Symptom**: Unstable or bouncy contacts

**Solution**:
- Reduce `max_depenetration_velocity` (try 0.1-0.2)
- Lower `bounce_threshold_velocity` (try 0.1-0.15)
- Ensure proper mass ratios between objects
