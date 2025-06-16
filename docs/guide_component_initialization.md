# Component Initialization Architecture

This guide documents the initialization sequence and dependency management for the DexHand environment components, with special focus on the physics manager → action processor → control_dt → action scaling chain.

## Overview

The DexHand environment uses a component-based architecture where each component has clearly defined responsibilities and dependencies. The initialization follows a specific order to ensure all dependencies are available when needed.

## Initialization Sequence

### 1. Core Infrastructure Setup
```python
# In DexHandBase.__init__()
def _init_components(self):
    # Create simulation first
    self.sim = self.create_sim()

    # Create hand initializer
    self.hand_initializer = HandInitializer(...)

    # Load assets and create environments
    self.hand_asset = self.hand_initializer.load_hand_asset(...)
    self._create_envs()
    handles = self.hand_initializer.create_hands(...)
```

### 2. Tensor System Setup
```python
    # Create tensor manager (after actors exist)
    self.tensor_manager = TensorManager(...)

    # CRITICAL: Acquire tensors BEFORE prepare_sim for GPU pipeline
    self.tensor_manager.acquire_tensor_handles()

    # CRITICAL: Call prepare_sim after actors created and tensors acquired
    self.gym.prepare_sim(self.sim)

    # Set up tensor references
    tensors = self.tensor_manager.setup_tensors(...)
    self.dof_state = tensors["dof_state"]
    self.dof_props = tensors["dof_props"]
    # ... other tensors
```

### 3. Physics Manager Creation
```python
    # Create physics manager (foundation for timing)
    self.physics_manager = PhysicsManager(
        gym=self.gym,
        sim=self.sim,
        device=self.device,
        use_gpu_pipeline=self.use_gpu_pipeline,
    )

    # Set physics timestep
    self.physics_manager.set_dt(self.physics_dt)
```

## Critical Dependency Chain: control_dt

The `control_dt` parameter flows through the system following a specific dependency chain:

### Single Source of Truth Pattern

```python
# PhysicsManager stores the authoritative control_dt
class PhysicsManager:
    def __init__(self):
        self.physics_dt = 0.01
        self.physics_steps_per_control_step = 1  # Auto-detected

    @property
    def control_dt(self):
        return self.physics_dt * self.physics_steps_per_control_step

    def set_dt(self, physics_dt):
        self.physics_dt = physics_dt
        # control_dt automatically recalculated via property
```

### Dependent Components Access via Property Decorators

Both ActionProcessor and ObservationEncoder access `control_dt` using property decorators:

```python
class ActionProcessor:
    def __init__(self, ..., physics_manager):
        self.physics_manager = physics_manager

    @property
    def control_dt(self):
        """Access control_dt from physics manager (single source of truth)."""
        if self.physics_manager is None:
            raise RuntimeError("physics_manager not set. Cannot access control_dt.")
        return self.physics_manager.control_dt

class ObservationEncoder:
    def __init__(self, ..., physics_manager):
        self.physics_manager = physics_manager

    @property
    def control_dt(self):
        """Access control_dt from physics manager (single source of truth)."""
        if self.physics_manager is None:
            raise RuntimeError("physics_manager not set. Cannot access control_dt.")
        return self.physics_manager.control_dt
```

### 4. Action Processor Initialization

```python
    # Create action processor (depends on physics_manager for control_dt)
    self.action_processor = ActionProcessor(
        gym=self.gym,
        sim=self.sim,
        num_envs=self.num_envs,
        device=self.device,
        dof_props=self.dof_props,
        hand_asset=self.hand_asset,
        physics_manager=self.physics_manager,  # ← Key dependency
    )

    # Set control mode BEFORE setup (affects scaling calculations)
    self.action_processor.set_control_mode(self.cfg["env"]["controlMode"])

    # Basic setup (computes action scaling using control_dt)
    self.action_processor.setup(self.num_dof, self.dof_props)
```

### 5. Observation Encoder Initialization

```python
    # Create observation encoder (also depends on physics_manager)
    self.observation_encoder = ObservationEncoder(
        gym=self.gym,
        sim=self.sim,
        num_envs=self.num_envs,
        device=self.device,
        tensor_manager=self.tensor_manager,
        hand_asset=self.hand_asset,
        hand_initializer=self.hand_initializer,
        physics_manager=self.physics_manager,  # ← Key dependency
    )
```

## Action Scaling Dependency Chain

The action scaling system depends on `control_dt` being available during setup:

### Position Mode Scaling
```python
# In ActionProcessor._compute_base_scaling_coeffs()
for i in range(self.NUM_BASE_DOFS):
    if self.action_control_mode == "position":
        # Direct DOF limit mapping
        min_val = self.dof_lower_limits[i]
        max_val = self.dof_upper_limits[i]
    else:  # position_delta mode
        # Velocity-based scaling (requires control_dt)
        if i < 3:  # Linear DOFs
            max_delta = self.control_dt * self.policy_base_lin_velocity_limit
        else:  # Angular DOFs
            max_delta = self.control_dt * self.policy_base_ang_velocity_limit
        min_val = -max_delta
        max_val = max_delta

    # Pre-compute scaling coefficients
    self.action_space_scale[action_idx] = 0.5 * (max_val - min_val)
    self.action_space_bias[action_idx] = min_val
```

### Why Property Decorators Are Used

Property decorators provide several advantages over manual synchronization:

1. **Automatic Updates**: When `physics_steps_per_control_step` changes (via auto-detection), `control_dt` automatically reflects the new value
2. **Fail-Fast**: Components immediately fail if `physics_manager` is not properly initialized
3. **No State Duplication**: `control_dt` exists in only one place
4. **No Manual Synchronization**: No need to call update methods when physics step ratio changes

### Before (Manual Synchronization - Removed)
```python
# OLD APPROACH - prone to inconsistency
def some_physics_change():
    new_control_dt = self.physics_manager.control_dt
    self.action_processor.set_control_dt(new_control_dt)  # Manual update
    self.observation_encoder.set_control_dt(new_control_dt)  # Manual update
    # Risk: forgetting to update all components
```

### After (Property Decorator - Current)
```python
# NEW APPROACH - automatic consistency
def some_physics_change():
    # Physics manager updates its control_dt calculation
    self.physics_manager.physics_steps_per_control_step = new_value

    # All components automatically see the new control_dt
    # No manual synchronization needed!
```

## Auto-Detection and Dynamic Updates

### Physics Step Auto-Detection

The system can detect the required physics steps per control step during runtime:

```python
# In PhysicsManager.step_physics()
def step_physics(self, refresh_tensors=True):
    self.physics_step_count += 1

    # Auto-detect physics_steps_per_control_step
    if not self.auto_detected_physics_steps:
        measured_steps = self.physics_step_count - self.last_control_step_count

        if measured_steps > self.physics_steps_per_control_step:
            self.physics_steps_per_control_step = measured_steps
            self.auto_detected_physics_steps = True
            # control_dt automatically updates via property calculation
```

### Automatic Propagation

When auto-detection occurs, the new `control_dt` is automatically available to all dependent components:

```python
# In DexHandBase.post_physics_step()
def post_physics_step(self):
    # Mark control step for auto-detection
    self.physics_manager.mark_control_step()

    # If auto-detection updated physics_steps_per_control_step,
    # all components automatically see the new control_dt
    # No manual updates needed!
```

## Component Dependencies Summary

```
PhysicsManager (stores control_dt)
    ↓
ActionProcessor (accesses control_dt via @property)
    ↓
Action scaling coefficients
    ↓
Proper action space mapping

PhysicsManager (stores control_dt)
    ↓
ObservationEncoder (accesses control_dt via @property)
    ↓
Manual velocity computation
    ↓
Accurate velocity observations
```

## Best Practices

### 1. Pass Dependencies in Constructors
```python
# ✅ CORRECT - dependencies passed during initialization
def __init__(self, ..., physics_manager):
    self.physics_manager = physics_manager
```

### 2. Use Property Decorators for Computed Values
```python
# ✅ CORRECT - computed from single source
@property
def control_dt(self):
    return self.physics_manager.control_dt
```

### 3. Fail Fast on Missing Dependencies
```python
# ✅ CORRECT - explicit error messages
@property
def control_dt(self):
    if self.physics_manager is None:
        raise RuntimeError("physics_manager not set. Cannot access control_dt.")
    return self.physics_manager.control_dt
```

### 4. Avoid Manual Synchronization
```python
# ❌ WRONG - manual synchronization prone to errors
def update_control_dt(self, new_dt):
    self.control_dt = new_dt  # Duplicates state

# ✅ CORRECT - single source accessed via property
@property
def control_dt(self):
    return self.physics_manager.control_dt  # No duplication
```

## Initialization Order Requirements

The components must be initialized in this order to satisfy dependencies:

1. **Simulation and Assets**: Basic Isaac Gym setup
2. **TensorManager**: Acquire tensors after actors exist
3. **PhysicsManager**: Foundation for timing calculations
4. **ActionProcessor**: Depends on physics_manager for control_dt
5. **ObservationEncoder**: Depends on physics_manager for control_dt
6. **Other Components**: Reset manager, viewer controller, etc.

## Troubleshooting

### Common Initialization Errors

**Error**: `RuntimeError: physics_manager not set. Cannot access control_dt.`
- **Cause**: Component trying to access control_dt before physics_manager is set
- **Solution**: Ensure physics_manager is passed in constructor and stored

**Error**: `AttributeError: '_dof_name_to_idx_cache' not initialized`
- **Cause**: ActionProcessor setup() called before all instance variables initialized
- **Solution**: Ensure all instance variables set in `__init__` before any methods called

**Error**: Action scaling produces wrong values
- **Cause**: control_dt not available when scaling coefficients computed
- **Solution**: Ensure physics_manager initialized before ActionProcessor.setup()

## Related Documentation

- **Physics Implementation**: [`reference_physics_implementation.md`](reference_physics_implementation.md)
- **Design Decisions**: [`design_decisions.md`](design_decisions.md)
- **DOF/Action Reference**: [`api_dof_control.md`](api_dof_control.md)
- **Observation System**: [`guide_observation_system.md`](guide_observation_system.md)
