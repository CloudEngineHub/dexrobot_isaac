# fix-007-episode-length-of-grasping.md

Fix BlindGrasping task episode length inconsistencies and configuration inheritance issues.

## Context

BlindGrasping task episodes are terminating at inconsistent step counts (399-499 steps) instead of the expected 500 steps, and the physics timing configuration is not being applied correctly due to fundamental configuration inheritance architecture problems.

**Observed Symptoms:**
- Episodes ending at 399-499 steps instead of 500
- Physics running at 200Hz (dt=0.005) instead of expected 100Hz (dt=0.01)
- Control cycle requiring 2 physics steps instead of 1
- "Early failure seems not enforced" behavior

**Root Cause Analysis:**
The configuration inheritance order is architecturally wrong. Main config.yaml has `_self_` positioned last in defaults, causing its `sim.dt: 0.005` to override task-specific settings like BlindGrasping's `sim.dt: 0.01`.

## Current State

**Broken Configuration Hierarchy:**
1. Main config.yaml loads task (BlindGrasping: dt=0.01)
2. Main config.yaml applies `_self_` LAST (dt=0.005 overrides task)
3. Result: Wrong physics timing affects episode behavior

**Evidence:**
```
physics_dt: 0.005000s  # Should be 0.01 for BlindGrasping
physics_steps_per_control: 2  # Should be 1 with correct dt
control_dt: 0.010000s  # Correct result achieved wrong way
```

## Desired Outcome

**Correct Configuration Architecture:**
- Task-specific settings ALWAYS override base/global settings
- BlindGrasping runs with dt=0.01 (100Hz physics, 1 physics step per control)
- BaseTask continues with dt=0.005 (200Hz physics)
- Episodes run for full 500 steps when no early termination criteria are met

**Physics Timing Goals:**
- BlindGrasping: 100Hz physics (dt=0.01) with 1 physics step per control
- BaseTask: 200Hz physics (dt=0.005) with 2 physics steps per control
- Other tasks: Can specify their own optimal dt values

## Constraints

**Architectural Principles:**
- Follow fail-fast philosophy: task configs should not need defensive checks
- Maintain component responsibility separation
- Respect configuration inheritance: specialized overrides general
- No breaking changes to existing BaseTask behavior

**Configuration Design Rules:**
- Main config.yaml: Only global defaults that apply to ALL tasks
- Task configs: Task-specific overrides that should never be overridden
- CLI overrides: Should work as expected for task-specific parameters

## Implementation Notes

**Primary Fix:**
1. Fix Hydra inheritance order so task-specific configs properly override base config
2. Investigate moving `_self_` position in main config.yaml defaults list
3. Ensure BlindGrasping.yaml's `sim.dt: 0.01` overrides config.yaml's `sim.dt: 0.005`
4. Maintain base config.yaml as source of global defaults that tasks can override

**Testing Requirements:**
- Verify BlindGrasping shows `physics_dt: 0.010000s` and `physics_steps_per_control: 1`
- Verify BaseTask shows `physics_dt: 0.005000s` and `physics_steps_per_control: 2`
- Test episode lengths reach full 500 steps when no early termination occurs
- Validate termination criteria work correctly with proper timing

**Secondary Investigation:**
- Check if other parameters in main config.yaml have same inheritance problem
- Investigate if episode termination issues persist after physics timing fix
- Analyze whether early termination criteria need adjustment for new timing

## Dependencies

- Configuration system understanding (Hydra inheritance order)
- Physics timing measurement system (control_dt calculation)
- Episode termination logic in BlindGrasping task
