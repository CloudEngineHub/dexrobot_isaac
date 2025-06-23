# Physics Parameter Tuning Guide

This guide explains how to tune physics parameters for stable simulation, particularly for GPU pipeline with rendering.

## Problem Symptoms

### Drift Issues
- Fingers drift when no actions applied
- Hand position changes without commanded motion
- Joints don't hold position under zero action

### Performance Issues
- Simulation runs slowly
- CUDA memory errors
- Unstable contact behavior

## Starting Configuration

### Current Default Settings (Optimized for Performance)
Location: `dexhand_env/cfg/task/BaseTask.yaml`

```yaml
sim:
  dt: 0.005                       # 200 Hz - high-fidelity physics for optimal performance
  substeps: 1                     # Reduced due to higher frequency
  physx:
    solver_type: 1                # TGS solver
    num_position_iterations: 6    # Optimized for smaller timestep
    num_velocity_iterations: 0    # NVIDIA recommended
    contact_offset: 0.008         # Balanced value for 200 Hz physics
    rest_offset: 0.003            # Appropriate for smaller timestep
    contact_collection: 1         # CC_LAST_SUBSTEP (GPU critical)
    default_buffer_size_multiplier: 4.0  # Optimized for performance
    gpu_contact_pairs_per_env: 512  # Auto-scaled by num_envs
    always_use_articulations: true
    bounce_threshold_velocity: 0.15  # Moderate value for 200 Hz
    max_depenetration_velocity: 8.0  # Optimized for faster corrections
```

**Performance**: These settings provide **6x performance improvement** over previous defaults while maintaining excellent stability for dexterous manipulation tasks.

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
- Auto-detected control frequency typically 100-200 Hz

**`substeps: 1`**
- Reduced from previous default (4) due to higher frequency
- Higher physics frequency reduces need for substeps
- Optimal balance for 200 Hz simulation

### Contact Parameters
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

### 2. Test Your Specific Use Case
```bash
python examples/dexhand_test.py --episode-length 50
```

Observe:
- Joint position stability over time
- Computational performance (FPS)
- Any error messages or warnings

### 3. Iterative Tuning

**For stability issues:**
- Try increasing `num_position_iterations`
- Try decreasing `dt` for finer time resolution
- Try increasing `substeps`
- Each change affects both stability and performance

**For performance issues:**
- Try reducing solver iterations
- Try increasing `dt` (larger timesteps)
- Try reducing `substeps`
- Consider trade-offs for your application

**For memory issues:**
- Monitor GPU memory usage
- Adjust contact buffer sizes if needed
- Consider reducing number of environments

### 4. Validate for Your Task
- Test with your specific manipulation tasks
- Ensure parameters work across different scenarios
- Document what works best for your use case

## General Principles

### Joint Damping vs Solver Parameters
- Increasing joint damping may not always improve stability
- Solver parameters often have more impact
- Experiment to find what works for your case

### Finding Balance
- Very high solver iterations increase computation time
- Very small timesteps can impact real-time performance
- Each application has different requirements

### Environment Differences
- GPU and CPU pipelines may behave differently
- Rendering vs headless modes can affect physics
- Always test in your target deployment configuration

### Viewer Synchronization
- IsaacGym automatically handles real-time synchronization via `gym.sync_frame_time()`
- Viewer always syncs to real-time when rendering for smooth visualization
- Performance profiling will show high rendering % when waiting for real-time sync
- For training without visualization, use headless mode to run as fast as possible

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

### Modern Recommended Ranges
**Dexterous Manipulation**: 200-500 Hz (dt: 0.005-0.002)
**General Robotics**: 100-200 Hz (dt: 0.01-0.005)
**Simple Tasks**: 50-100 Hz (dt: 0.02-0.01)

### Finding Your Balance
- Start with defaults
- Adjust one parameter at a time
- Test with your specific tasks
- Document what works for your application

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

### What to Look For
- Monitor DOF position changes over time
- Check if changes stabilize or continue drifting
- Note any oscillations or sudden jumps
- Observe computational performance (FPS)

### Making Decisions
- Small position drift may be acceptable for some tasks
- High-precision tasks need tighter tolerances
- Consider your real-time requirements
- Balance accuracy with computational budget

Your specific application will determine what constitutes acceptable performance.
