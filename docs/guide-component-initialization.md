# Component Initialization Architecture

This guide documents the initialization sequence and dependency management for the DexHand environment components, with special focus on the two-stage initialization pattern and the physics manager → action processor → control_dt → action scaling chain.

## Overview

The DexHand environment uses a component-based architecture where each component has clearly defined responsibilities and dependencies. The initialization follows a specific order to ensure all dependencies are available when needed.

## Two-Stage Initialization Pattern

### Why Two-Stage is Necessary

The DexHand environment uses a **two-stage initialization pattern** that is a core architectural principle. This pattern exists because `control_dt` can only be determined at runtime by measuring actual physics behavior:

```python
# control_dt = physics_dt × physics_steps_per_control
# where physics_steps_per_control is measured, not configured
```

**Why measurement is required:**
- Environment resets require variable physics steps to stabilize
- Isaac Gym may add internal physics steps during state changes
- GPU pipeline timing variations
- Multi-environment synchronization effects

**📊 Visual Explanation**: See [`control-dt-timing-diagram.md`](control-dt-timing-diagram.md) for a detailed explanation of the parallel simulation constraint and measurement process with timeline illustrations.

### The Two-Stage Lifecycle

**Stage 1: Construction + Basic Initialization**
```python
# Create components with known dependencies
action_processor = ActionProcessor(parent=self)
action_processor.initialize_from_config(config)
# Component is functional but cannot access control_dt yet
```

**Stage 2: Finalization After Measurement**
```python
# Measure control_dt by running dummy control cycle
self.physics_manager.start_control_cycle_measurement()
# ... run measurement cycle ...
self.physics_manager.finish_control_cycle_measurement()

# Now finalize all components
action_processor.finalize_setup()  # Can now access control_dt
task.finalize_setup()
```

### Component Development Pattern

All components follow this standardized pattern:

```python
class MyComponent:
    def __init__(self, parent):
        """Initialize with parent reference only."""
        self.parent = parent
        self.gym = parent.gym
        self.sim = parent.sim

    @property
    def control_dt(self):
        """Access control_dt from physics manager (single source of truth)."""
        return self.parent.physics_manager.control_dt

    def initialize_from_config(self, config):
        """Phase 1: Initialize with configuration (no control_dt access)."""
        # Set up everything that doesn't need control_dt
        pass

    def finalize_setup(self):
        """Phase 2: Complete initialization (control_dt now available)."""
        # Set up everything that needs control_dt
        pass
```

### Component Development Rules

1. **Split initialization logic**: Basic setup in `initialize_from_config()`, control_dt-dependent logic in `finalize_setup()`
2. **Use property decorators**: Always access `control_dt` via `self.parent.physics_manager.control_dt`
3. **Trust initialization order**: Don't check if `control_dt` exists - trust the two-stage pattern
4. **Single source of truth**: Access all shared state through property decorators

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
        physics_dt=self.physics_dt,  # Physics timestep from config
        use_gpu_pipeline=self.use_gpu_pipeline,
    )
    # Note: control_dt will be determined by active measurement on first step
```

## Critical Dependency Chain: control_dt

The `control_dt` parameter flows through the system following a specific dependency chain:

### Single Source of Truth Pattern

```python
# PhysicsManager stores the authoritative control_dt
class PhysicsManager:
    def __init__(self, ..., physics_dt):
        self.physics_dt = physics_dt  # From config
        self.physics_steps_per_control_step = 1  # Will be measured
        self.control_dt = None  # Set after measurement

    def start_control_cycle_measurement(self):
        # Active measurement on first step
        if self.control_dt is None:
            # Reset counters and start measuring
            return True
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

    # Initialize configuration
    self.action_processor.initialize_from_config(action_processor_config)

    # Note: finalize_setup() is called after control_dt is measured
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

### Active Control Cycle Measurement

The system actively measures the required physics steps per control cycle during initialization:

```python
# In DexHandBase._init_components()
def _init_components(self):
    # ... create all components ...

    # Perform control cycle measurement to determine control_dt
    self._perform_control_cycle_measurement()

def _perform_control_cycle_measurement(self):
    # Start measurement
    if not self.physics_manager.start_control_cycle_measurement():
        return  # Already measured

    # Process dummy actions
    dummy_actions = torch.zeros((self.num_envs, self.num_actions))
    self.action_processor.process_actions(dummy_actions)

    # Step physics
    self.physics_manager.step_physics()

    # Force reset on all environments to measure full cycle
    self.reset_idx(torch.arange(self.num_envs))

    # Finish measurement and set control_dt
    self.physics_manager.finish_control_cycle_measurement()

    # Now finalize action processor with control_dt available
    self.action_processor.finalize_setup()
```

This ensures control_dt is determined before any external calls to reset() or step().

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

## Actor Creation and Ordering

The DexHand environment follows a specific actor creation order to ensure consistent indexing and proper simulation setup. This is critical for multi-actor environments where tasks add objects like cubes, spheres, or other interactive elements.

### Actor Creation Sequence

The creation flow follows this sequence:

1. **Task Initialization** - Task is created with placeholder sim/gym references
2. **Environment Creation** - Basic environment setup
3. **Task Asset Loading** - Task loads its assets (meshes, primitives) early
4. **Hand Creation** - Hands are created FIRST (always actor index 0)
5. **Task Object Creation** - Task objects are created AFTER hands

### Implementation Pattern

```python
def _init_components(self):
    # Create simulation
    self.sim = self.create_sim()

    # Update task with real sim/gym instances
    self.task.sim = self.sim
    self.task.gym = self.gym

    # Load task assets BEFORE creating any actors
    self.task.load_task_assets()

    # Create environments
    self._create_envs()

    # Create hands FIRST (ensures hand is actor 0)
    handles = self.hand_initializer.create_hands(self.envs, self.hand_asset)

    # Create task objects AFTER hands
    for i in range(self.num_envs):
        if hasattr(self.task, "create_task_objects"):
            self.task.create_task_objects(self.gym, self.sim, self.envs[i], i)
```

### Task Implementation Example

```python
class MyGraspTask(DexTask):
    def load_task_assets(self):
        """Load assets early, before any actors are created."""
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False

        self.object_asset = self.gym.create_box(
            self.sim, 0.05, 0.05, 0.05, asset_options
        )

    def create_task_objects(self, gym, sim, env_ptr, env_id):
        """Create task objects AFTER hands."""
        # Hand is already created as actor 0
        # Object will be actor 1
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 0.1)

        object_handle = gym.create_actor(
            env_ptr, self.object_asset, pose,
            f"object_{env_id}", env_id, 0  # collision group
        )
        self.object_handles.append(object_handle)
```

### Actor Indexing Requirements

With the proper creation order:

- **Hand**: Always actor index 0 in each environment
- **Task Objects**: Start from actor index 1

This consistent indexing is critical for:
- DOF control (hand DOFs are accessed via actor 0)
- State tensors (root states indexed by actor)
- Reset operations

### Common Actor Creation Issues

**Issue**: Actions not applied in multi-environment setup
- **Cause**: Actor indices might be incorrect when objects are created before hands
- **Solution**: Ensure hands are created first using the pattern above

**Issue**: DOF control affects wrong actor
- **Cause**: DOF indices assume hand is actor 0
- **Solution**: Follow the creation order to ensure hand is always first

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
3. **PhysicsManager**: Foundation for timing calculations (physics_dt from config)
4. **ActionProcessor**: Created with physics_manager reference
5. **ObservationEncoder**: Created with physics_manager reference
6. **Other Components**: Reset manager, viewer controller, etc.
7. **Control Cycle Measurement**: Measure physics steps and finalize ActionProcessor

## Troubleshooting

### Common Initialization Errors

**Error**: `RuntimeError: physics_manager not set. Cannot access control_dt.`
- **Cause**: Component trying to access control_dt before physics_manager is set
- **Solution**: Ensure physics_manager is passed in constructor and stored

**Error**: `AttributeError: '_dof_name_to_idx_cache' not initialized`
- **Cause**: ActionProcessor setup() called before all instance variables initialized
- **Solution**: Ensure all instance variables set in `__init__` before any methods called

**Error**: `TypeError: unsupported operand type(s) for *: 'NoneType' and 'float'`
- **Cause**: ActionProcessor trying to use control_dt before measurement
- **Solution**: Measurement should happen during initialization

**Error**: Action scaling produces wrong values
- **Cause**: control_dt not measured yet or measurement incorrect
- **Solution**: Check auto-detection is running and detecting correct step count

## Rule-Based Control Architecture

### Control Flow Overview

The environment supports hybrid control where some DOFs are controlled by the RL policy while others use rule-based control:

1. **Policy Control**: DOFs controlled by the RL agent through actions
2. **Rule-Based Control**: DOFs controlled by programmatic functions

### Execution Order

The control flow follows this specific order:

```python
# In pre_physics_step():
1. process_actions()          # Process RL policy actions
2. _apply_rule_based_control() # Override with rule-based control
3. set_dof_position_target()   # Apply final targets to simulation
```

**Important**: Rule-based control is applied AFTER process_actions, allowing it to override any default values.

### Configuration

Control mode is determined by configuration flags:
- `policy_controls_hand_base`: If False, base uses rule-based control
- `policy_controls_fingers`: If False, fingers use rule-based control

### Rule-Based Controller Interface

Controllers are registered using `set_rule_based_controllers()`:

```python
def my_base_controller(env):
    """Return (num_envs, 6) tensor with base DOF targets in physical units."""
    t = env.progress_buf[0] * env.control_dt
    targets = torch.zeros((env.num_envs, 6), device=env.device)
    targets[:, 0] = 0.1 * torch.sin(t)  # X oscillation
    return targets

env.set_rule_based_controllers(base_controller=my_base_controller)
```

### Default Values

When neither policy nor rule-based control is active:
- Default values from configuration are used
- These serve as fallback positions to prevent undefined behavior

## Related Documentation

- **Physics Implementation**: [`reference-physics-implementation.md`](reference-physics-implementation.md) - Physics stepping and tensor management
- **Action Processing**: [`guide-action-pipeline.md`](guide-action-pipeline.md) - Action pipeline and rule-based control
- **DOF/Action Reference**: [`reference-dof-control-api.md`](reference-dof-control-api.md) - Complete DOF mapping and action spaces
- **Observation System**: [`guide-observation-system.md`](guide-observation-system.md) - Observation encoding and configuration
- **System Architecture**: [`ARCHITECTURE.md`](ARCHITECTURE.md) - Overall system design and principles
- **Task Creation**: [`guide-task-creation.md`](guide-task-creation.md) - Creating custom tasks with proper initialization
