# Fix reset_idx Timing Consistency Bug

## Problem Statement

The `reset_idx()` implementation in `ResetManager` contains a critical timing consistency bug that violates the core assumption of deterministic `physics_steps_per_control_step`.

## Root Cause Analysis

### The Bug Location
**File**: `dexhand_env/components/reset/reset_manager.py`
**Lines**: 103-105

```python
def reset_idx(self, env_ids):
    try:
        if len(env_ids) == 0:
            # No environments to reset
            return True  # ← BUG: Early return skips step_physics()

        # ... reset logic ...

        # Run a physics step to integrate the DOF changes
        self.physics_manager.step_physics(refresh_tensors=True)  # ← This is skipped!
```

### Impact on Timing Consistency

**During Measurement Phase:**
- With resets: `physics_steps_per_control_step = 2` (main step + reset step)
- Without resets: `physics_steps_per_control_step = 1` (only main step, reset step skipped)

**During Normal Operation:**
- Control cycles WITH resets: 2 physics steps (correct)
- Control cycles WITHOUT resets: 1 physics step (inconsistent!)

This breaks the fundamental architectural assumption that `control_dt` is constant and deterministic.

### Why This Violates Core Architecture

1. **Parallel Simulation Constraint**: All environments must step together on GPU
2. **Deterministic Timing**: Every control cycle must take identical physics steps
3. **Action Scaling Dependency**: ActionProcessor calculates scaling coefficients assuming fixed control_dt

## Technical Analysis

### Parallel GPU Constraint
Isaac Gym runs all environments in parallel. The `step_physics()` call affects **ALL environments simultaneously**, not just the ones being reset. This step must be **unconditional** to maintain timing consistency.

### Measurement Impact
The auto-detection algorithm counts ALL `step_physics()` calls during the measurement cycle:

```python
# In _perform_control_cycle_measurement():
self.physics_manager.step_physics()      # Count = 1
self.reset_idx(all_env_ids)              # Should add 1 more → Count = 2
# But if env_ids is empty, reset_idx returns early → Count = 1 (BUG!)
```

### Action Scaling Corruption
The inconsistent `control_dt` corrupts action scaling in `position_delta` mode:

```python
# Calculated during finalize_setup() with measured control_dt
max_delta = control_dt × max_velocity

# With inconsistent control_dt:
# - Sometimes: max_delta = (physics_dt × 1) × max_velocity = smaller range
# - Sometimes: max_delta = (physics_dt × 2) × max_velocity = larger range
```

## Solution Design

### Core Principle
**The `step_physics()` call must be unconditional** - it executes regardless of whether any environments need resetting, because all environments must step together due to parallel simulation.

### Implementation Strategy

1. **Remove Early Return**: Eliminate the `len(env_ids) == 0` early return
2. **Conditional Reset Logic**: Apply reset operations only to environments that need it
3. **Unconditional Physics Step**: Always execute `step_physics()` for timing consistency

### Proposed Fix

```python
def reset_idx(self, env_ids):
    """Reset specified environments - physics step runs unconditionally for timing consistency."""
    try:
        # Reset logic only applied to environments that need it
        if len(env_ids) > 0:
            # Reset episode step count for reset environments
            self.episode_step_count[env_ids] = 0

            # Reset DOF states to default positions
            dof_pos = self.default_dof_pos.clone()
            self.dof_state[env_ids, :, 0] = dof_pos
            self.dof_state[env_ids, :, 1] = 0

            # Task-specific reset
            try:
                self.task.reset_task_state(env_ids)
            except AttributeError:
                pass

            # Apply root states and DOF states
            # ... (existing root state and DOF state application logic)

            # Reset action processor targets
            current_dof_positions = self.dof_state[env_ids, :, 0]
            self.action_processor.reset_targets(env_ids, current_dof_positions)

        # CRITICAL: Unconditional physics step for timing consistency
        # This ensures ALL control cycles take the same number of physics steps
        # regardless of whether any specific environments actually need reset
        self.physics_manager.step_physics(refresh_tensors=True)

        return True
```

## Testing Strategy

### Measurement Consistency Test
Verify that measurement produces consistent results:

```python
# Test 1: Measure with resets
physics_steps_with_resets = measure_control_cycle_with_resets()

# Test 2: Measure without resets
physics_steps_without_resets = measure_control_cycle_without_resets()

# Should be identical
assert physics_steps_with_resets == physics_steps_without_resets
```

### Action Scaling Validation
Verify action scaling coefficients remain consistent:

```python
# Test action scaling with both scenarios
action_scale_with_resets = action_processor.action_space_scale.clone()
action_scale_without_resets = action_processor.action_space_scale.clone()

# Should be identical
assert torch.allclose(action_scale_with_resets, action_scale_without_resets)
```

## Risk Assessment

### Low Risk Changes
- Logic restructuring maintains identical behavior for non-empty `env_ids`
- Physics step timing becomes more predictable, not less

### High Impact Benefits
- Eliminates timing inconsistency bugs
- Ensures deterministic control_dt measurement
- Maintains architectural invariants
- Fixes action scaling corruption

## Implementation Requirements

### Files to Modify
- `dexhand_env/components/reset/reset_manager.py` - Remove early return, restructure logic

### Testing Requirements
- Test both empty and non-empty `env_ids` scenarios
- Verify physics step counting consistency
- Validate action scaling coefficient stability
- Ensure no regression in reset functionality

### Documentation Updates
- Update component initialization guide with timing consistency notes
- Add architectural rationale comments in reset_manager.py

## Success Criteria

1. **Timing Consistency**: `physics_steps_per_control_step` identical regardless of reset scenarios
2. **Action Scaling Stability**: Consistent action scaling coefficients across all scenarios
3. **No Functional Regression**: Reset operations work identically for environments that need reset
4. **Architectural Compliance**: Maintains parallel simulation constraint and deterministic timing assumptions
