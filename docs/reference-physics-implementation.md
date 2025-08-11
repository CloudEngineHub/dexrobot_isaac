# Physics Implementation Reference

Technical details about how physics and models are implemented in the DexHand environment. Physics simulation is managed by `PhysicsManager` (`dexhand_env/components/physics_manager.py`).

## MJCF Property Loading

### How Isaac Gym Reads Properties

Isaac Gym loads joint properties directly from MJCF files without code overrides:

**Stiffness** from actuator definitions:
```xml
<position name="act_ARTx" joint="ARTx" kp="10000" ... />
<!-- Results in: stiffness = 10000.0 -->
```

**Damping** from joint attributes:
```xml
<joint name="ARTx" damping="20" ... />
<default><joint damping="1"/></default>  <!-- Default for others -->
<!-- Results in: damping = 20.0 or 1.0 -->
```

### Current Property Values
- Base joints: `stiffness=10000.0, damping=20.0`
- Finger joints: `stiffness=20.0, damping=1.0`

## Model Generation Pipeline

### Automatic Joint Limit Processing
The model generation scripts ensure Isaac Gym compatibility:

```python
# dexrobot_mujoco/utils/mjcf_utils.py:add_joint_limits()
for joint in root.findall(".//joint"):
    if joint.get("range") is not None and joint.get("limited") is None:
        joint.set("limited", "true")
```

This automatic processing ensures all joints with ranges have `limited="true"` attribute required by Isaac Gym.

### Model File Structure
```
assets/dexrobot_mujoco/dexrobot_mujoco/models/
├── dexhand021_right_simplified_floating.xml  # Main model
├── floating_base.xml                          # Base DOF definitions
└── defaults.xml                               # Joint properties
```

### Regeneration Process
To regenerate models after MJCF changes:
```bash
cd assets/dexrobot_mujoco/scripts
./regenerate_all.sh
```

## Physics Step Management

**📊 Conceptual Overview**: For a visual explanation of the parallel simulation constraint and measurement process, see [`control-dt-timing-diagram.md`](control-dt-timing-diagram.md).

### Why Auto-Detection is Necessary

The physics step auto-detection system solves several critical challenges:

1. **Reset Penetration Resolution**
   - Resets may initialize objects in penetrating configurations
   - Physics engine needs multiple steps to resolve penetrations
   - Number of steps varies based on penetration severity

2. **GPU Parallelization Constraints**
   - All parallel environments must step together on GPU
   - If ANY environment resets, ALL environments must take extra physics steps
   - Cannot step individual environments independently

3. **Manual Conversion Errors**
   - Manual tracking of physics vs control steps is error-prone
   - Different tasks may need different step ratios
   - Reset complexity varies between scenarios

### Implementation Architecture

The `PhysicsManager` component provides a wrapper that tracks ALL physics steps:

```python
class PhysicsManager:
    def __init__(self):
        self.physics_steps_count = 0
        self.control_steps_count = 0
        self.physics_steps_per_control_step = None

    def step_physics(self):
        # Physics stepping implemented in PhysicsManager.step_physics()
        """Wrapper that MUST be used for ALL physics steps"""
        self.physics_steps_count += 1
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
```

### Auto-Detection Algorithm

The system automatically determines the step ratio during the first control cycle:

```python
def detect_physics_steps_per_control(self):
    # Reset step counter
    initial_physics_steps = self.physics_steps_count

    # Perform one control cycle (includes potential resets)
    self.reset_idx(torch.arange(self.num_envs))
    self.compute_observations()
    self.step(torch.zeros_like(self.actions))

    # Calculate how many physics steps occurred
    steps_taken = self.physics_steps_count - initial_physics_steps

    # Set the detected ratio
    self.physics_steps_per_control_step = steps_taken
    self.control_dt = self.physics_dt * self.physics_steps_per_control_step

    print(f"Auto-detected physics_steps_per_control_step: {steps_taken}")
```

### Critical Usage Requirements

1. **Always Use the Wrapper**
   ```python
   # ❌ WRONG - Bypasses tracking
   self.gym.simulate(self.sim)

   # ✅ CORRECT - Properly tracked
   self._step_physics()
   ```

2. **Use Wrapper in ALL Locations**
   - Main stepping loop
   - Reset operations
   - Any initialization requiring physics steps
   - Debug or test code

3. **Typical Detection Results**
   ```
   # Simple reset without objects
   Auto-detected physics_steps_per_control_step: 2

   # Complex reset with penetrations
   Auto-detected physics_steps_per_control_step: 5

   # Reset with multiple objects
   Auto-detected physics_steps_per_control_step: 8
   ```

### Example: Reset with Depenetration

```python
def reset_idx(self, env_ids):
    # Set initial poses (may penetrate)
    self._set_actor_root_state_tensor_indexed(env_ids)

    # Multiple physics steps to resolve penetrations
    for _ in range(self.reset_physics_steps):
        self._step_physics()  # Uses wrapper for tracking

    # Now objects are depenetrated and stable
```

### Timing Calculations

Once detected, the system maintains consistent timing:

```python
# Control frequency = physics frequency / steps per control
control_freq = 1.0 / self.control_dt
physics_freq = 1.0 / self.physics_dt

# Example with auto-detected ratio of 2:
# physics_dt = 0.0083s (120 Hz)
# physics_steps_per_control_step = 2
# control_dt = 0.0166s (60 Hz)
```

## Tensor Management

### GPU Pipeline Tensor Acquisition

The `TensorManager` component handles tensor setup:

```python
# Critical: Must be called AFTER actor creation
def acquire_tensor_handles(self):
    # Get global tensors
    self.actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
    self.dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
    self.rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

    # Wrap tensors for GPU access
    self.root_state_tensor = gymtorch.wrap_tensor(self.actor_root_state_tensor)
    self.dof_state_tensor = gymtorch.wrap_tensor(self.dof_state_tensor)
    self.rigid_body_states = gymtorch.wrap_tensor(self.rigid_body_state_tensor)
```

### Tensor Refresh Pattern
```python
# Must refresh before reading
self.gym.refresh_dof_state_tensor(self.sim)
self.gym.refresh_actor_root_state_tensor(self.sim)
self.gym.refresh_rigid_body_state_tensor(self.sim)
```

## DOF Property Handling

### GPU vs CPU Pipeline Differences

**CPU Pipeline**: Can use asset DOF properties
```python
dof_props = self.gym.get_asset_dof_properties(asset)
```

**GPU Pipeline**: Must use actor DOF properties
```python
dof_props = self.gym.get_actor_dof_properties(env, hand_handle)
```

This difference is critical for GPU pipeline compatibility.

## Method Signatures

### PhysicsManager Core Methods

#### `step_physics(refresh_tensors: bool = True) -> bool`
**File**: `dexhand_env/components/physics_manager.py`

Physics stepping wrapper that handles simulation stepping and tensor refresh.

**Parameters**:
- `refresh_tensors`: Whether to refresh tensor data after stepping (default: True)

**Returns**: Success flag

**Usage**:
```python
# Standard physics step with tensor refresh
success = self.physics_manager.step_physics()

# Step without tensor refresh (for performance)
success = self.physics_manager.step_physics(refresh_tensors=False)
```

#### `start_control_cycle_measurement() -> bool`
**File**: `dexhand_env/components/physics_manager.py`

Start measuring physics steps for control_dt auto-detection.

**Returns**: True if measurement started, False if already measured

**Usage**:
```python
# Begin measurement during initialization
if self.physics_manager.start_control_cycle_measurement():
    # Perform dummy control cycle
    self.reset_idx(torch.arange(self.num_envs))
```

#### `finish_control_cycle_measurement() -> None`
**File**: `dexhand_env/components/physics_manager.py`

Complete measurement and set control_dt based on counted physics steps.

**Usage**:
```python
# Complete measurement and set control_dt
self.physics_manager.finish_control_cycle_measurement()
# control_dt is now available for component finalization
```

#### `apply_dof_states(env_ids: torch.Tensor, dof_state: torch.Tensor, actor_index: int = 0) -> bool`
**File**: `dexhand_env/components/physics_manager.py`

Apply DOF states to specified actor in given environments.

**Parameters**:
- `env_ids`: Environment IDs to update `[num_envs_to_update]`
- `dof_state`: DOF states `[num_envs, num_dofs, 2]` (position, velocity)
- `actor_index`: Local actor index within each environment (default: 0)

**Returns**: Success flag

**Usage**:
```python
# Apply new DOF states to specific environments
success = self.physics_manager.apply_dof_states(
    env_ids=reset_env_ids,
    dof_state=new_dof_states,
    actor_index=0  # Hand actor
)
```

#### `apply_root_states(env_ids: torch.Tensor, root_state_tensor: torch.Tensor, actor_indices: torch.Tensor) -> bool`
**File**: `dexhand_env/components/physics_manager.py`

Apply root states to specified actors across environments.

**Parameters**:
- `env_ids`: Environment IDs `[num_envs_to_update]`
- `root_state_tensor`: Root states `[num_envs, num_bodies, 13]` (pos: 3, quat: 4, vel: 6)
- `actor_indices`: Actor indices for each environment `[num_envs]`

**Returns**: Success flag

**Usage**:
```python
# Apply root states for object reset
success = self.physics_manager.apply_root_states(
    env_ids=reset_env_ids,
    root_state_tensor=new_root_states,
    actor_indices=self.object_actor_indices
)
```

#### `refresh_tensors() -> bool`
**File**: `dexhand_env/components/physics_manager.py`

Refresh all simulation tensors to ensure data is current.

**Returns**: Success flag

**Usage**:
```python
# Manually refresh tensors after state changes
self.physics_manager.refresh_tensors()
```

### Physics Properties

#### `control_dt -> float`
**File**: `dexhand_env/components/physics_manager.py`

Get current control timestep (physics_dt × physics_steps_per_control_step).

**Returns**: Control timestep in seconds

**Usage**:
```python
# Access control_dt for velocity calculations
dt = self.physics_manager.control_dt
velocity = position_delta / dt
```

#### `physics_steps_per_control_step -> int`
**File**: `dexhand_env/components/physics_manager.py`

Get measured physics steps per control step.

**Returns**: Number of physics steps per control step

## Implementation Locations

### Key Components
- **Physics Manager**: `PhysicsManager.step_physics()`
- **Tensor Manager**: `TensorManager.refresh_tensors()`
- **Hand Initializer**: `HandInitializer.create_hands()`
- **Model Generation**: `assets/dexrobot_mujoco/utils/mjcf_utils.py`

### Configuration Files
- **Physics Config**: `dexhand_env/cfg/task/BaseTask.yaml`
- **Model Files**: `assets/dexrobot_mujoco/dexrobot_mujoco/models/`

## See Also

- [`DESIGN_DECISIONS.md`](DESIGN_DECISIONS.md) - Critical design caveats
- [`guide-component-initialization.md`](guide-component-initialization.md) - Component initialization and dependency management
- [`guide-physics-tuning.md`](guide-physics-tuning.md) - Parameter tuning guide
- [`reference-dof-control-api.md`](reference-dof-control-api.md) - DOF and action reference
