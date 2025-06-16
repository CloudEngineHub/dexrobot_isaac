# Observation System Guide

This guide explains the observation system design, implementation, and usage in the DexHand environment.

## Design Philosophy

### Two-Level Architecture

The observation system implements a clear separation between data availability and data usage:

1. **Observation Dictionary (`obs_dict`)**: Contains ALL computed observations
   - Complete data for debugging and analysis
   - Always available regardless of configuration
   - Accessible via `get_observations_dict()`

2. **Observation Buffer (`obs_buf`)**: Contains only selected observations
   - Configured via `observation_keys` in YAML
   - Used for RL training
   - Memory-efficient tensor for policy input

### Key Design Principles

- **Flexibility**: Any observation can be enabled/disabled without code changes
- **Debuggability**: All observations remain accessible for inspection
- **Efficiency**: Only selected observations are concatenated for RL
- **Extensibility**: New observations can be added easily

## Configuration System

### Observation Keys Configuration

Observations are selected via the `observation_keys` list in config:

```yaml
# dex_hand_env/cfg/task/BaseTask.yaml
env:
  observation_keys:
    - hand_pose           # 7D: position (3) + quaternion (4)
    - hand_vel            # 6D: linear (3) + angular (3) velocities
    - base_dof_pos        # 6D: base joint positions
    - base_dof_vel        # 6D: base joint velocities
    - finger_dof_pos      # 20D: finger joint positions
    - finger_dof_vel      # 20D: finger joint velocities
    # - contact_forces    # Disabled but still computed
```

### Automatic Dimension Calculation

The system automatically calculates observation space size:

```python
# During initialization
for key in self.observation_keys:
    obs_shape = self._get_observation_shape(key)
    total_obs_dim += obs_shape
```

## Implementation Architecture

### 1. Observation Computation

All observations are computed regardless of configuration:

```python
def compute_observations(self):
    # Compute ALL observations into dictionary
    obs_dict = {}

    # Hand state observations
    obs_dict["hand_pose"] = self._compute_hand_pose()
    obs_dict["hand_vel"] = self._compute_hand_velocity()

    # DOF observations
    obs_dict["base_dof_pos"] = self._compute_base_dof_pos()
    obs_dict["finger_dof_pos"] = self._compute_finger_dof_pos()

    # Contact observations
    obs_dict["contact_forces"] = self._compute_contact_forces()

    # Task-specific observations
    task_obs = self._compute_task_observations()
    obs_dict.update(task_obs)

    return obs_dict
```

### 2. Selective Concatenation

Only enabled observations are concatenated:

```python
def _concatenate_observations(self, obs_dict):
    obs_list = []

    # Only concatenate enabled keys
    for key in self.observation_keys:
        if key in obs_dict:
            obs_list.append(obs_dict[key])

    # Create final observation buffer
    self.obs_buf = torch.cat(obs_list, dim=-1)
```

### 3. Shape Management

Each observation component has a defined shape:

```python
OBSERVATION_SHAPES = {
    "hand_pose": 7,                # pos(3) + quat(4) - raw pose
    "hand_pose_arr_aligned": 7,    # pos(3) + quat(4) - aligned to ARR DOFs
    "hand_vel": 6,                 # lin(3) + ang(3)
    "base_dof_pos": 6,             # 6 base DOFs
    "base_dof_vel": 6,
    "finger_dof_pos": 20,          # 20 finger DOFs
    "finger_dof_vel": 20,
    "fingertip_poses": 35,         # 5 tips × 7D
    "contact_forces": 15,          # 5 tips × 3D
    # ... more observations
}
```

## Accessor System

### Dictionary Access for Debugging

Access any observation regardless of configuration:

```python
# Get full observation dictionary
obs_dict = env.get_observations_dict()

# Access specific observations
hand_pose = obs_dict["hand_pose"]        # Always available
contact_forces = obs_dict["contact_forces"]  # Even if disabled

# Useful for debugging
print(f"Hand position: {hand_pose[:, :3]}")
print(f"Contact on thumb: {contact_forces[:, 0:3]}")
```

### Buffer Access for RL

Get concatenated observations for policy:

```python
# Get observation buffer
obs = env.obs_buf  # Shape: (num_envs, obs_dim)

# obs contains only enabled observations
# in the order specified by observation_keys
```

### Direct Component Access

Access specific observation components:

```python
# Via observation encoder
encoder = env.observation_encoder

# Get specific observations
hand_pose = encoder.get_hand_pose()
fingertip_poses = encoder.get_fingertip_poses()
dof_positions = encoder.get_dof_positions()
```

## Adding New Observations

### 1. Define Computation Method

```python
def _compute_my_observation(self):
    # Compute your observation
    my_obs = some_computation()

    # Ensure correct shape: (num_envs, obs_dim)
    return my_obs.view(self.num_envs, -1)
```

### 2. Add to Dictionary

```python
def _compute_default_observations(self):
    obs_dict = {}

    # Existing observations...

    # Add your observation
    obs_dict["my_observation"] = self._compute_my_observation()

    return obs_dict
```

### 3. Define Shape

```python
# In observation_encoder.py
OBSERVATION_SHAPES = {
    # Existing shapes...
    "my_observation": 10,  # Your observation dimension
}
```

### 4. Enable in Config

```yaml
env:
  observation_keys:
    - hand_pose
    - my_observation  # Add to enable for RL
```

## Common Observation Types

### State Observations
- `hand_pose`: Hand base position and orientation (raw from rigid body state)
- `hand_pose_arr_aligned`: Hand pose with orientation aligned to ARR DOFs (compensates for built-in 90° Y rotation)
- `hand_vel`: Hand base linear and angular velocity
- `base_dof_pos/vel`: Base joint positions and velocities
- `finger_dof_pos/vel`: Finger joint positions and velocities

### ARR-Aligned Pose Details

The `hand_pose_arr_aligned` observation addresses a coordinate system issue in the floating hand model:

**Problem**: Due to the floating hand model design, the hand is mounted with a built-in 90° Y-axis rotation. When ARRx=ARRy=ARRz=0, the raw hand quaternion is approximately [0, 0.707, 0, 0.707] instead of identity [0, 0, 0, 1].

**Solution**: The ARR-aligned pose compensates for this rotation:
- Position: Same as raw hand pose
- Orientation: Multiplied by inverse of the built-in rotation
- Result: Quaternion values that directly correspond to ARRx, ARRy, ARRz DOF values

**Usage Example**:
```python
obs_dict = env.get_observations_dict()

# Raw pose - includes built-in 90° Y rotation
raw_quat = obs_dict["hand_pose"][:, 3:7]  # [0, 0.707, 0, 0.707] when ARR=0

# ARR-aligned pose - compensated orientation
aligned_quat = obs_dict["hand_pose_arr_aligned"][:, 3:7]  # [0, 0, 0, 1] when ARR=0

# Convert to Euler angles for intuitive interpretation
from scipy.spatial.transform import Rotation
euler = Rotation.from_quat(aligned_quat).as_euler('xyz', degrees=True)
# euler ≈ [0°, 0°, 0°] when ARRx=ARRy=ARRz=0
```

### Contact Observations
- `contact_forces`: 3D force vectors at fingertips
- `contact_binary`: Binary contact indicators
- `contact_locations`: Contact point positions

### Relative Observations
- `fingertip_poses`: Fingertip poses in world frame
- `fingertip_poses_hand_frame`: In hand frame
- `object_pose_hand_frame`: Object relative to hand

### Control Observations
- `dof_targets`: Current DOF position targets
- `last_actions`: Previous step's actions
- `action_deltas`: Change in actions

## Performance Considerations

### Memory Efficiency
- Only enabled observations are stored in `obs_buf`
- Dictionary overhead is minimal (references only)
- Batch operations for all environments

### Computation Efficiency
- Vectorized operations for all observations
- No loops over environments
- Efficient tensor reshaping

### Best Practices
1. Enable only necessary observations for RL
2. Use dictionary access for debugging/analysis
3. Batch compute related observations
4. Validate shapes during development

## Troubleshooting

### Zero or Constant Values
- Check tensor refresh before reading
- Verify correct tensor indexing
- Ensure shape matches expectation

### Missing Observations
- Verify key exists in `obs_dict`
- Check computation method called
- Ensure shape defined in `OBSERVATION_SHAPES`

### Performance Issues
- Profile observation computation
- Check for unnecessary loops
- Use vectorized operations

## Example Usage

### Basic RL Training
```python
# Config: only essential observations
observation_keys:
  - finger_dof_pos
  - finger_dof_vel
  - dof_targets

# Training loop
obs = env.reset()
while training:
    action = policy(obs)
    obs, reward, done, info = env.step(action)
```

### Debugging Session
```python
# Get all observations for analysis
obs_dict = env.get_observations_dict()

# Check specific components
print("Keys available:", obs_dict.keys())
print("Hand pose:", obs_dict["hand_pose"])
print("Contact forces:", obs_dict["contact_forces"])

# Plot observations over time
plot_observation_history(obs_dict["finger_dof_pos"])
```

### Custom Task Observations
```python
# In your task class
def _compute_task_observations(self):
    obs_dict = {}

    # Task-specific observations
    obs_dict["target_distance"] = self._compute_target_distance()
    obs_dict["grasp_quality"] = self._compute_grasp_quality()

    return obs_dict
```

## Related Documentation

- **Component Initialization**: [`guide_component_initialization.md`](guide_component_initialization.md) - Dependency management and initialization order
- **Design Decisions**: [`design_decisions.md`](design_decisions.md) - Critical design caveats
- **DOF/Action Reference**: [`api_dof_control.md`](api_dof_control.md) - DOF indices and action mapping
- **Physics Implementation**: [`reference_physics_implementation.md`](reference_physics_implementation.md) - Technical implementation details
