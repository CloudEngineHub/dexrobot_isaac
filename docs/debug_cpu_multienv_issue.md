# CPU Multi-Environment DOF Control Issue

## Problem Summary

Isaac Gym's CPU pipeline has a bug where `gym.set_dof_position_target_tensor()` only applies to environment 0 in multi-environment simulations. All other environments are ignored, despite receiving correct target tensors.

## Evidence

### Test Results

**CPU Pipeline (Broken):**
```bash
python examples/dexhand_test.py --device cpu --policy-controls-base true --num-envs 2
```
- Environment 0: Moves correctly (`[-0.051, -0.000, -0.001, ...]`)
- Environment 1: Does NOT move (`[-0.000, -0.000, -0.001, ...]`)

**GPU Pipeline (Works):**
```bash
python examples/dexhand_test.py --device cuda:0 --policy-controls-base true --num-envs 2
```
- Environment 0: Moves correctly (`[-0.082, -0.000, -0.001, ...]`)
- Environment 1: Moves correctly (`[-0.082, -0.000, -0.001, ...]`)

### Debug Logs Confirm Our Implementation is Correct

1. **Actions applied correctly to all environments:**
   ```
   Env 0: base_action[0]=0.378, finger_action[6]=0.265
   Env 1: base_action[0]=0.378, finger_action[6]=0.265
   ```

2. **Targets computed correctly for all environments:**
   ```
   Env 0 base targets: [-0.050, 0.000, 0.000, 0.000, 0.000, 0.000]
   Env 1 base targets: [-0.050, 0.000, 0.000, 0.000, 0.000, 0.000]
   ```

3. **Isaac Gym receives correct tensor with all environment targets:**
   ```
   Device: cpu, Targets tensor device: cpu
   Targets tensor shape: torch.Size([2, 26])
   ```

## Documentation Contradiction

Isaac Gym documentation states that **CPU pipeline has fewer restrictions** than GPU pipeline:

> "There are also some limitations when using the tensor API with the GPU pipeline. These don't apply with the CPU pipeline..."

However, our testing shows the opposite: GPU pipeline works correctly while CPU pipeline fails for multi-environment DOF control.

## Code Analysis

### Our Implementation (Verified Correct)

1. **Test Script:** Actions applied to all environments using `actions[:, ...]` syntax
2. **ActionProcessor:** Computes targets for all environments in tensor operations
3. **Tensor Application:** Calls `gym.set_dof_position_target_tensor()` with complete tensor

### Isaac Gym Bug Location

The bug appears to be in Isaac Gym's CPU pipeline implementation of `set_dof_position_target_tensor()`. The function:
- Receives the correct tensor with targets for all environments
- Only applies targets to environment 0
- Ignores environments 1+ (silent failure)

## Workaround

Use GPU pipeline for multi-environment simulations:
```bash
# Use this instead of --device cpu
python examples/dexhand_test.py --device cuda:0 --num-envs 2
```

## Debug Tracking

Debug logging has been added to track this issue:
- Test script logs multi-environment action application every 200 steps
- ActionProcessor logs target tensor verification every 200 steps
- Both use `logger.debug()` to avoid spam in normal usage

## Status

- **Bug Location:** Isaac Gym CPU pipeline
- **Workaround:** Use GPU pipeline
- **Impact:** Medium (GPU pipeline works fine)
- **Documented:** In roadmap.md section 5
