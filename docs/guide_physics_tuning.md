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

### Current Default Settings
Location: `dex_hand_env/cfg/task/BaseTask.yaml`

```yaml
sim:
  dt: 0.0083                      # 120 Hz - balanced for rendering
  substeps: 4                     # Better convergence with rendering
  physx:
    solver_type: 1                # TGS solver
    num_position_iterations: 32   # Critical for stability
    num_velocity_iterations: 0    # NVIDIA recommended
    contact_offset: 0.005         
    rest_offset: 0.002            
    contact_collection: 1         # CC_LAST_SUBSTEP (GPU critical)
    default_buffer_size_multiplier: 8.0
    gpu_contact_pairs_per_env: 1024  # Auto-scaled by num_envs
    always_use_articulations: true
    bounce_threshold_velocity: 0.1
    max_depenetration_velocity: 5.0
```

## Key Parameters Explained

### Solver Iterations
**`num_position_iterations: 32`**
- Controls constraint solving accuracy
- Higher values generally improve stability
- Trade-off between accuracy and computational cost
- Start with default and adjust based on your needs

### Timestep and Substeps
**`dt: 0.0083` (120Hz)**
- Simulation timestep in seconds
- Smaller values may improve stability
- Larger values improve performance
- Consider your control frequency needs

**`substeps: 4`**
- Internal physics substeps per simulation step
- More substeps can improve convergence
- Affects computational cost proportionally

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

## Example Parameter Ranges

### For Higher Stability
Consider trying:
- Smaller `dt` (e.g., 0.005, 0.001)
- More `substeps` (e.g., 6, 8)
- Higher `num_position_iterations` (e.g., 48, 64)

### For Better Performance
Consider trying:
- Larger `dt` (e.g., 0.01, 0.0167)
- Fewer `substeps` (e.g., 2, 1)
- Lower `num_position_iterations` (e.g., 16, 8)

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