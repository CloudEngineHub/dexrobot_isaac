# Physics Configuration System

This directory contains modular physics configurations that can be mixed and matched with different tasks for optimal performance vs. quality trade-offs.

## Configuration Files

### `default.yaml`
- **Purpose**: Balanced quality/performance for standard development
- **Settings**: 4 substeps, 16 position iterations, 0.001 contact_offset
- **Use case**: BaseTask and general development work
- **Performance**: Standard baseline

### `fast.yaml`
- **Purpose**: Optimized for real-time visualization and testing
- **Settings**: 2 substeps, 8 position iterations, 0.002 contact_offset
- **Use case**: test_render, test_stream configs for smooth visualization
- **Performance**: ~2-3x faster than default
- **Trade-off**: Slightly reduced physics accuracy for speed

### `accurate.yaml`
- **Purpose**: Maximum precision for training and research
- **Settings**: 16 substeps, 32 position iterations, 0.001 contact_offset
- **Use case**: Training and research requiring high precision
- **Performance**: ~2-3x slower than default
- **Trade-off**: Highest physics quality for computational cost

## Usage

### In Task Configurations
Add physics config to task defaults:
```yaml
defaults:
  - BaseTask
  - /physics/accurate    # Override physics settings
  - _self_
```

### In Test Configurations
Add physics config to test defaults:
```yaml
defaults:
  - config
  - base/test
  - /physics/fast        # Fast physics for smooth rendering
  - _self_
```

## Parameter Comparison

| Parameter | fast | default | accurate |
|-----------|------|---------|----------|
| **substeps** | 2 | 4 | 16 |
| **num_position_iterations** | 8 | 16 | 32 |
| **contact_offset** | 0.002 | 0.001 | 0.001 |
| **gpu_contact_pairs_per_env** | 256 | 512 | 512 |
| **default_buffer_size_multiplier** | 2.0 | 4.0 | 4.0 |

## Performance Impact

- **fast → default**: ~50% performance cost for better accuracy
- **default → accurate**: ~100% performance cost for maximum precision
- **fast → accurate**: ~300% performance cost for maximum quality

## Design Principles

1. **Task-specific dt**: Each task controls its own timestep (`dt`) for RL environment consistency
2. **Physics-specific substeps**: Physics configs control simulation parameters only
3. **Modular inheritance**: Mix and match physics profiles with any task
4. **Clear separation**: Physics simulation vs. RL environment timing are independent

## Migration from Monolithic Configs

Before:
```yaml
# MyTask.yaml
sim:
  dt: 0.01
  substeps: 16
  physx:
    num_position_iterations: 32
    # ... other physics params
```

After:
```yaml
# Example: Custom task with accurate physics
defaults:
  - BaseTask
  - /physics/accurate    # Inherits all physics params
  - _self_

sim:
  dt: 0.01              # Only RL timing control
```

This provides clean separation and reusable physics profiles across different tasks and test scenarios.
