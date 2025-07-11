# DexHand Troubleshooting Guide

This guide provides comprehensive solutions to common issues encountered when working with the DexHand environment.

> **Quick Links:** For Isaac Gym installation and graphics issues, see [Isaac Gym troubleshooting](https://developer.nvidia.com/isaac-gym). For general training troubleshooting, see [rl_games documentation](https://github.com/Denys88/rl_games).

## Setup Issues

### Git Submodule Issues

#### Missing Submodule Assets
**Symptom**: `FileNotFoundError` or `ModuleNotFoundError` related to `dexrobot_mujoco`
**Root Cause**: Git submodule not initialized or corrupted
**Solution**:
```bash
# Initialize submodules
git submodule update --init --recursive

# Verify submodule status
git submodule status

# Check assets exist
ls -la assets/dexrobot_mujoco/
```

#### Submodule Update Failures
**Symptom**: `fatal: remote error: access denied` or network timeouts during submodule updates
**Solutions**:
- Check network connectivity to repository sources
- Try alternative repository sources (GitHub vs Gitee)
- Verify SSH keys for authentication
- Use internal repository if available

**See Also**: [Git Submodule Setup Guide](git-submodule-setup.md) for comprehensive troubleshooting

### Episode Length Parameter Ignored
**Symptom**: The `--episode-length` parameter in test script does not properly limit episode duration
**Solution**: Ensure you're using `--episode-length` not `--episodeLength` (correct hyphenation)

### Hand Not Visible on Startup
**Symptom**: Hand model doesn't appear in the simulation viewer
**Solution**: Check that `hand_rigid_body_index` is properly initialized during component setup

### Isaac Gym Installation Problems
**Symptom**: Import errors or simulation fails to start
**Solution**:
1. Verify Isaac Gym Preview 4 installation following [official instructions](https://developer.nvidia.com/isaac-gym)
2. Test Isaac Gym with their examples: `cd isaacgym/python/examples && python joint_monkey.py`
3. Ensure CUDA compatibility with your GPU drivers

## Execution Errors

### Component Initialization Errors

#### `control_dt is None`
**Symptom**: Error during component initialization mentioning `control_dt` is None
**Root Cause**: Component trying to access `control_dt` before physics measurement is complete
**Solution**: Ensure `finalize_setup()` is called after physics measurement completes
**See Also**: [Component Initialization Guide](guide-component-initialization.md) for two-stage initialization details

#### Property Decorator Errors
**Symptom**: AttributeError when accessing component properties
**Root Cause**: Incorrect component parent relationships or initialization order
**Solution**:
1. Check component parent relationships are correctly established
2. Verify components are initialized in proper dependency order
3. Use property decorators for accessing sibling components
**See Also**: [Component Debugging Guide](guide-debugging.md) for detailed troubleshooting

### Actions Not Applied
**Symptom**: Robot doesn't respond to policy actions or moves incorrectly
**Root Cause**: Control mode mismatch or action space configuration error
**Solution**:
1. Verify control mode matches your action space (see [DOF Control API](reference-dof-control-api.md))
2. Check `policy_controls_fingers` and `policy_controls_base` settings in task config
3. Ensure action dimensions match expected input size
4. Debug action processing pipeline step by step

### Coordinate System Confusion
**Symptom**: Robot moves in unexpected directions or positions
**Root Cause**: Misunderstanding of fixed base coordinate system
**Solution**: Remember that ALL motion is relative to spawn position, not absolute world coordinates
**See Also**: [Coordinate Systems Guide](reference-coordinate-systems.md) for fixed base motion details

## Training Issues

### Reward System Problems

#### Zero Rewards
**Symptom**: Policy receives no reward signal during training
**Root Cause**: Reward component weights set to zero in task configuration
**Solution**:
1. Check reward weights in task config are non-zero
2. Verify reward components are properly computed in task implementation
3. Use reward logging to debug individual reward components

#### NaN Rewards
**Symptom**: Training fails with NaN (Not a Number) reward values
**Root Cause**: Mathematical operations producing invalid results in reward computation
**Solution**:
1. Add tensor validation in task reward computation methods
2. Check for division by zero in distance/ratio calculations
3. Validate input tensors for finite values before mathematical operations
4. Use reward clamping to prevent extreme values

#### Reward Scale Mismatch
**Symptom**: Policy doesn't learn or learns very slowly
**Root Cause**: Reward component ranges don't match expected policy learning scale
**Solution**:
1. Analyze reward component ranges and scales
2. Adjust reward weights to balance different components
3. Consider reward normalization or scaling
4. Verify reward signal provides sufficient learning gradient

### Action Space Mismatches

#### Wrong Action Dimensions
**Symptom**: Model expects different action size than environment provides
**Root Cause**: Mismatch between policy output and environment action space
**Solution**:
1. Verify `policy_controls_fingers` and `policy_controls_base` settings match model
2. Check action space configuration in task config
3. Ensure policy architecture matches expected action dimensions

#### Control Mode Errors
**Symptom**: Actions produce unexpected robot behavior
**Root Cause**: Action processor configuration doesn't match policy output expectations
**Solution**:
1. Ensure action processor configuration matches policy output type
2. Verify control mode (position, velocity, effort) matches training setup
3. Check action scaling and limits are appropriate

## Performance Issues

### Slow Training
**Symptom**: Training runs much slower than expected
**Common Causes**:
1. **Too many environments with rendering**: Reduce `env.numEnvs` when using `env.render=true`
2. **Single environment training**: Use `training.test=true` for single environment debugging
3. **GPU memory issues**: Reduce batch size or number of environments
**Solution**: Profile training to identify bottlenecks and adjust configuration accordingly

### Memory Issues
**Symptom**: Out of memory errors during training or simulation
**Solution**:
1. Reduce number of parallel environments (`env.numEnvs`)
2. Check for memory leaks in custom task implementations
3. Monitor GPU memory usage and adjust batch sizes
4. Use gradient checkpointing if available

## API Failures

### Configuration Errors
**Symptom**: Hydra configuration errors or parameter validation failures
**Solution**:
1. Use full configuration paths (e.g., `env.numEnvs`) for command-line overrides
2. Verify all required configuration sections are present
3. Check for typos in configuration parameter names
4. Validate configuration compatibility with current codebase version

### Checkpoint Loading Errors
**Symptom**: Cannot load saved model checkpoints
**Solution**:
1. Verify checkpoint file exists and has `.pth` extension
2. Check checkpoint compatibility with current model architecture
3. Use correct checkpoint path resolution (relative vs absolute)
4. Ensure checkpoint was saved with compatible rl_games version

## Getting Additional Help

If you're still experiencing issues:

1. **Check component logs**: Use loguru logging to get detailed debugging information
2. **Consult specialized guides**: See [Component Debugging Guide](guide-debugging.md) for component-specific issues
3. **Review examples**: Study the BoxGrasping task implementation for working patterns
4. **Validate setup**: Ensure your installation matches the requirements in [README.md](../README.md)

## Related Documentation

- **[Component Debugging Guide](guide-debugging.md)** - Detailed component troubleshooting and real-world case studies
- **[Coordinate Systems Guide](reference-coordinate-systems.md)** - Understanding fixed base motion and coordinate quirks
- **[DOF Control API](reference-dof-control-api.md)** - Action space and control mode reference
- **[Component Initialization Guide](guide-component-initialization.md)** - Two-stage initialization deep dive
- **[Task Creation Guide](guide-task-creation.md)** - Creating custom tasks and avoiding common pitfalls
