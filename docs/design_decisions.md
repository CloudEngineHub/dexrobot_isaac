# 🚨 Critical Design Decisions

**READ THIS FIRST** - Essential design caveats that affect physics, modeling, and control behavior.

## Core Architecture

### Fixed Hand Base with Internal DOF Control
- Hand uses `fix_base_link = True` - anchored to world, won't fall under gravity
- Movement via internal ARTx/y/z/Rx/y/z DOFs, not actor translation
- **CRITICAL**: All motion is relative to spawn position, not absolute world coordinates

### Relative Motion Control
- `ARTz = 0.0` → stay at spawn Z position  
- `ARTz = +0.1` → move +0.1 units from spawn position
- `ARTz = -0.1` → move -0.1 units from spawn position
- **NOT absolute world Z coordinates!**

## Isaac Gym Requirements

### Joint Limits Must Be Explicit
```xml
<!-- WRONG - limits ignored -->
<joint name="finger_joint" range="0 1.3" />

<!-- CORRECT - limits enforced -->  
<joint name="finger_joint" range="0 1.3" limited="true" />
```

### GPU Pipeline Physics Requirements
- `num_position_iterations: 32` minimum for joint stability
- `contact_collection: 1` (CC_LAST_SUBSTEP) critical for GPU
- Call order: create actors → acquire tensors → `gym.prepare_sim()`

## Configuration Sources

### Properties Come Directly from MJCF
- Actuator `kp` → PD stiffness, joint `damping` → damping  
- **No code overrides** - edit MJCF files to change properties
- Model regeneration required: `cd assets/dexrobot_mujoco/scripts && ./regenerate_all.sh`

## Related Documentation

- **DOF/Action Reference**: [`api_dof_control.md`](api_dof_control.md)
- **Observation System**: [`guide_observation_system.md`](guide_observation_system.md)
- **Physics Tuning**: [`guide_physics_tuning.md`](guide_physics_tuning.md)
- **Implementation Details**: [`reference_physics_implementation.md`](reference_physics_implementation.md)