# Component Debugging Guide

This guide covers debugging issues specific to the DexHand component architecture.

## Two-Stage Initialization Issues

### `control_dt is None` Errors

**Symptom:** `AttributeError: 'NoneType' object has no attribute` when accessing `control_dt`

**Root Cause:** Component trying to access `control_dt` before physics measurement completes

**Solution:**
```python
# ❌ WRONG: Accessing control_dt in Stage 1
def initialize_from_config(self, config):
    dt = self.control_dt  # FAILS - not measured yet

# ✅ CORRECT: Access only in Stage 2
def finalize_setup(self):
    dt = self.control_dt  # WORKS - measurement complete
```

**Debugging Steps:**
1. Check if `finalize_setup()` was called on your component
2. Verify parent component has `physics_manager` with measured `control_dt`
3. Add logging to track initialization sequence

### Property Decorator Failures

**Symptom:** `AttributeError: Component has no attribute 'parent'`

**Root Cause:** Component instantiated without proper parent reference

**Solution:**
```python
# ❌ WRONG: Missing parent in constructor
class MyComponent:
    def __init__(self):
        pass

# ✅ CORRECT: Store parent reference
class MyComponent:
    def __init__(self, parent):
        self.parent = parent
```

## Component Interaction Issues

### Missing Component Dependencies

**Symptom:** `AttributeError: 'NoneType' object has no attribute` when accessing sibling components

**Root Cause:** Component created before its dependencies

**Debugging:**
```python
# Add validation in finalize_setup
def finalize_setup(self):
    assert self.parent.tensor_manager is not None, "TensorManager not initialized"
    assert self.parent.physics_manager.control_dt is not None, "control_dt not measured"
```

### Circular Component References

**Symptom:** Initialization hangs or fails with recursive calls

**Root Cause:** Components directly referencing each other instead of using property decorators

**Solution:**
```python
# ❌ WRONG: Direct reference creates coupling
def __init__(self, parent):
    self.tensor_manager = parent.tensor_manager  # Creates circular ref

# ✅ CORRECT: Property decorator for access
@property
def tensor_manager(self):
    return self.parent.tensor_manager
```

## Configuration Issues

### Hydra Override Failures

**Symptom:** Configuration overrides ignored or cause crashes

**Common Issues:**
- Using wrong parameter paths: `numEnvs` instead of `env.numEnvs`
- Missing required configuration sections
- Type mismatches in YAML

**Debugging:**
```bash
# Print resolved configuration
python train.py --cfg job --resolve

# Validate specific override
python train.py env.numEnvs=64 --cfg job
```

### Task Configuration Errors

**Symptom:** Task fails to load or missing required parameters

**Debugging Steps:**
1. Check task config inherits from `BaseTask`
2. Verify all required reward weights are defined
3. Validate observation space configuration

## Action Processing Issues

### Wrong Action Dimensions

**Symptom:** `RuntimeError: dimension mismatch` in action processing

**Root Cause:** Policy output doesn't match expected action space

**Debugging:**
```python
# Check action space configuration
print(f"Expected actions: {env.action_space.shape}")
print(f"Policy controls fingers: {cfg.policy_controls_fingers}")
print(f"Policy controls base: {cfg.policy_controls_base}")

# Validate action processor setup
print(f"Action processor expects: {action_processor.expected_action_dim}")
```

### Control Mode Mismatches

**Symptom:** Actions not applied or robot behaves unexpectedly

**Common Issues:**
- `control_mode` in config doesn't match policy expectations
- DOF limits not properly configured
- Action scaling incorrect for control mode

**Solution:**
```yaml
# Ensure control mode matches policy training
control_mode: "position_delta"  # or "position", "velocity", "effort"

# Verify DOF configuration
dof_properties:
  - stiffness: 1000.0
    damping: 50.0
    effort: 100.0
```

## Reward System Debugging

### Zero or NaN Rewards

**Symptom:** Training doesn't progress or crashes with NaN

**Debugging Steps:**
```python
# Add reward validation in task
def compute_task_reward_terms(self):
    rewards = {}
    # ... compute rewards ...

    # Validate rewards
    for name, reward in rewards.items():
        assert not torch.isnan(reward).any(), f"NaN in {name} reward"
        assert torch.isfinite(reward).all(), f"Infinite values in {name} reward"

    return rewards
```

### Reward Scale Issues

**Symptom:** Some reward components dominate training

**Debugging:**
```python
# Log reward component statistics
def compute_total_reward(self):
    # ... existing computation ...

    # Debug logging
    for name, component in reward_components.items():
        self.logger.debug(f"{name}: mean={component.mean():.4f}, std={component.std():.4f}")
```

## Performance Issues

### Slow Tensor Operations

**Symptom:** Training significantly slower than expected

**Common Causes:**
- CPU-GPU data transfers in tight loops
- Non-vectorized operations over environments
- Unnecessary tensor copying

**Profiling:**
```python
# Add timing to component methods
import time

def compute_observations(self):
    start = time.time()
    # ... existing code ...
    self.logger.debug(f"Observation computation: {time.time() - start:.4f}s")
```

### Memory Leaks

**Symptom:** GPU memory usage increases over time

**Common Causes:**
- Tensors not properly released
- Growing lists of tensor references
- Gradient accumulation not cleared

**Debugging:**
```python
import torch

# Monitor GPU memory
def log_memory_usage(self):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        self.logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
```

## Common Anti-Patterns

### Defensive Programming in Components

**❌ Wrong:**
```python
if self.tensor_manager is not None:
    tensors = self.tensor_manager.get_tensors()
else:
    tensors = None  # Silent failure hides bugs
```

**✅ Correct:**
```python
# Let it fail fast to expose initialization bugs
tensors = self.tensor_manager.get_tensors()
```

### Improper Tensor Sharing

**❌ Wrong:**
```python
# Storing tensors creates stale references
def __init__(self, parent):
    self.rigid_body_states = parent.tensor_manager.rigid_body_states
```

**✅ Correct:**
```python
# Access tensors through property for fresh data
@property
def rigid_body_states(self):
    return self.parent.tensor_manager.rigid_body_states
```

### Environment Index Confusion

**❌ Wrong:**
```python
# Using global indices when local expected
hand_pos = self.rigid_body_states[env_ids, self.hand_index]  # Wrong if env_ids are global
```

**✅ Correct:**
```python
# Use local environment indices
local_env_ids = self.global_to_local_env_ids(env_ids)
hand_pos = self.rigid_body_states[local_env_ids, self.hand_index]
```

## Debugging Tools

### Component State Inspection

```python
def debug_component_state(self):
    """Print component state for debugging"""
    print(f"Component: {self.__class__.__name__}")
    print(f"Initialized: {getattr(self, '_initialized', False)}")
    print(f"Parent: {self.parent.__class__.__name__ if self.parent else None}")
    if hasattr(self, 'control_dt'):
        print(f"control_dt: {self.control_dt}")
```

### Tensor Shape Validation

```python
def validate_tensor_shapes(self):
    """Validate tensor shapes match expectations"""
    expected_shapes = {
        'rigid_body_states': (self.num_envs, self.num_rigid_bodies, 13),
        'dof_states': (self.num_envs, self.num_dofs, 2),
    }

    for name, expected_shape in expected_shapes.items():
        tensor = getattr(self, name)
        assert tensor.shape == expected_shape, f"{name} shape {tensor.shape} != {expected_shape}"
```

## Case Study: CPU Multi-Environment DOF Control Issue

This is a real-world debugging case from the DexHand project that demonstrates systematic issue investigation.

### Problem Description
Isaac Gym's CPU pipeline was failing to apply DOF position targets to environments 1+ in multi-environment simulations, while environment 0 worked correctly.

### Investigation Process

**1. Observed Symptoms:**
```bash
# CPU Pipeline (Broken)
python examples/dexhand_test.py --device cpu --num-envs 2
# Env 0: Moves correctly, Env 1: Does NOT move
```

**2. Hypothesis Testing:**
- Tested GPU pipeline as control: `--device cuda:0` (worked correctly)
- Added debug logging to verify our implementation was correct
- Confirmed Isaac Gym received proper tensor with all environment targets

**3. Root Cause Analysis:**
```python
# Debug logs confirmed our tensor was correct:
# Targets tensor shape: torch.Size([2, 26])
# Env 0 targets: [-0.050, 0.000, ...]
# Env 1 targets: [-0.050, 0.000, ...]
```

**4. Contradiction Discovery:**
Isaac Gym documentation claims CPU pipeline has fewer restrictions, but testing showed the opposite.

### Resolution Strategy

**Workaround Implementation:**
```yaml
# Force GPU pipeline for multi-environment scenarios
env:
  device: "cuda:0"  # Instead of "cpu"
```

**Documentation:**
- Added issue to roadmap.md as "Won't Fix"
- Implemented debug logging for future reference
- Created this case study for team knowledge

### Key Lessons

1. **Always test control cases**: GPU vs CPU pipeline comparison isolated the issue
2. **Debug logging is crucial**: Proved our implementation was correct
3. **Document workarounds**: Even unfixable issues need clear guidance
4. **Challenge assumptions**: Official documentation can be incorrect

This case demonstrates the systematic debugging approach: hypothesis formation, controlled testing, and clear documentation of findings and workarounds.

This debugging guide helps identify and resolve issues specific to the DexHand architecture patterns.
