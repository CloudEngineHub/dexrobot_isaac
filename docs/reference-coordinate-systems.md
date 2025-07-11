# Coordinate Systems and Motion Control

This guide explains the unique coordinate system behavior in the DexHand environment.

## Fixed Base Coordinate System

### Key Concept: Relative Motion Only

The DexHand uses `fix_base_link = True`, meaning the hand **cannot fall under gravity** and all motion is **relative to the spawn position**.

```python
# CRITICAL UNDERSTANDING:
# ARTz = 0.0  →  Stay at spawn height (NOT world Z=0)
# ARTz = +0.1 →  Move 0.1m UP from spawn position
# ARTz = -0.1 →  Move 0.1m DOWN from spawn position
```

### Common Mistakes

**❌ Absolute Coordinate Thinking:**
```python
# WRONG: Thinking these are world coordinates
target_position = [0.0, 0.0, 0.5]  # NOT "0.5m above world origin"
actions[2] = 0.5  # NOT "move to world Z=0.5"
```

**✅ Relative Coordinate Thinking:**
```python
# CORRECT: These are offsets from spawn position
relative_offset = [0.0, 0.0, 0.1]  # 0.1m above spawn height
actions[2] = 0.1  # Move 0.1m up from current position (in position_delta mode)
```

### Spawn Position as Origin

When the hand spawns at `initialHandPos: [0.0, 0.0, 0.2]`:

| Action Value | Actual Position | Description |
|--------------|----------------|-------------|
| `ARTz = 0.0` | `world_z = 0.2` | Stay at spawn height |
| `ARTz = +0.1` | `world_z = 0.3` | 0.1m above spawn |
| `ARTz = -0.1` | `world_z = 0.1` | 0.1m below spawn |
| `ARTz = -0.2` | `world_z = 0.0` | At world ground level |

## Floating Hand Orientation Quirks

### 90° Y-Axis Rotation Issue

The hand model has a **built-in 90° Y-axis rotation**. When all ARR values are zero:

```python
# When ARRx = ARRy = ARRz = 0.0:
expected_quat = [0, 0, 0, 1]        # What you might expect
actual_quat = [0, 0.707, 0, 0.707]  # What you actually get
```

### Using Aligned Observations

For control algorithms expecting standard orientation representations:

```python
# ❌ WRONG: Using raw hand pose
hand_orientation = observations["hand_pose_arr"][3:7]  # Includes 90° offset

# ✅ CORRECT: Using aligned pose
hand_orientation = observations["hand_pose_arr_aligned"][3:7]  # Removes offset
```

### Coordinate Frame Visualization

```
Standard Robot Frame:     DexHand Frame (90° Y rotation):

    Z                         X (forward)
    |                         |
    |                         |
    o─── Y                    o─── Z (up)
   /                         /
  X                         Y (left)
```

## Control Mode Implications

### Position Control

In `position` mode, actions are **absolute positions relative to spawn**:

```python
# Spawn at [0, 0, 0.2], then:
action = [0.1, 0.0, 0.1]  # Results in world position [0.1, 0.0, 0.3]
```

### Position Delta Control

In `position_delta` mode, actions are **incremental offsets**:

```python
# Current relative position [0.05, 0.0, 0.1], then:
action = [0.02, 0.0, 0.05]  # New relative position [0.07, 0.0, 0.15]
```

### Velocity Control

In `velocity` mode, actions are **velocities in the relative frame**:

```python
action = [0.1, 0.0, 0.0]  # 0.1 m/s in +X direction from spawn origin
```

## Practical Examples

### Moving to Specific Heights

**Goal:** Move hand to 30cm above the table (assuming table at world Z=0)

**If spawned at [0, 0, 0.2]:**
```python
# Position mode
target_height_relative = 0.3 - 0.2  # 30cm world - 20cm spawn
action[2] = 0.1  # Move to 0.1m above spawn = 30cm world height

# Position delta mode (from current position)
current_world_z = 0.2 + current_relative_z
desired_delta = 0.3 - current_world_z
action[2] = desired_delta
```

### Collision Avoidance

**Goal:** Keep hand above table surface (world Z=0)

```python
# Calculate minimum safe relative height
spawn_height = 0.2  # From config
table_height = 0.0  # World coordinates
safety_margin = 0.05

min_relative_z = (table_height + safety_margin) - spawn_height
# min_relative_z = (0.0 + 0.05) - 0.2 = -0.15

# Ensure action doesn't go below this
action[2] = max(action[2], min_relative_z)
```

### Reward Engineering

**Goal:** Reward for keeping hand at target height

```python
def compute_height_reward(self):
    # Get current world position
    spawn_z = 0.2  # From config initialHandPos[2]
    current_world_z = spawn_z + self.hand_relative_pos[:, 2]

    # Target in world coordinates
    target_world_z = 0.25

    # Compute reward based on world coordinates
    height_error = torch.abs(current_world_z - target_world_z)
    reward = torch.exp(-height_error * 10.0)

    return reward
```

## Configuration Examples

### Task-Specific Spawn Heights

```yaml
# For table manipulation (table at world Z=0)
env:
  initialHandPos: [0.0, 0.0, 0.25]  # Start 25cm above table

# For aerial manipulation
env:
  initialHandPos: [0.0, 0.0, 1.0]   # Start 1m above ground
```

### Safe Operating Ranges

```yaml
# Limit relative motion to safe ranges
action_scale:
  # Relative position limits
  ART_x: [-0.2, 0.2]  # ±20cm from spawn
  ART_y: [-0.2, 0.2]  # ±20cm from spawn
  ART_z: [-0.15, 0.3] # -15cm to +30cm from spawn
```

## Debugging Coordinate Issues

### Position Validation

```python
def validate_hand_position(self, world_pos, spawn_pos):
    """Validate hand position makes sense"""
    relative_pos = world_pos - spawn_pos

    # Check if hand is in reasonable range
    assert relative_pos[2] > -0.5, f"Hand too low: {world_pos[2]:.3f}"
    assert relative_pos[2] < 1.0, f"Hand too high: {world_pos[2]:.3f}"

    print(f"World pos: {world_pos}")
    print(f"Spawn pos: {spawn_pos}")
    print(f"Relative: {relative_pos}")
```

### Visualization Helpers

```python
def log_coordinate_info(self):
    """Debug coordinate transformations"""
    spawn_pos = self.initial_hand_pos
    current_relative = self.hand_relative_pos[0]  # First env
    current_world = spawn_pos + current_relative

    self.logger.info(f"Spawn: {spawn_pos}")
    self.logger.info(f"Relative: {current_relative}")
    self.logger.info(f"World: {current_world}")
```

## Method Signatures

### Coordinate Transformation Utilities

#### `point_in_hand_frame(world_point: torch.Tensor, hand_pose: torch.Tensor) -> torch.Tensor`
**File**: `dexhand_env/utils/coordinate_transforms.py`

Transform world coordinates to hand-relative coordinates accounting for spawn offset.

**Parameters**:
- `world_point`: Points in world coordinates `[num_envs, 3]`
- `hand_pose`: Hand pose `[num_envs, 7]` (position + quaternion)

**Returns**: Points in hand-relative frame `[num_envs, 3]`

**Usage**:
```python
# Transform object position to hand frame
from dexhand_env.utils.coordinate_transforms import point_in_hand_frame
object_in_hand_frame = point_in_hand_frame(object_world_pos, hand_pose)
```

#### `lookat_quaternion(eye: torch.Tensor, target: torch.Tensor, up: torch.Tensor) -> torch.Tensor`
**File**: `dexhand_env/utils/torch_jit_utils.py`

Compute quaternion for look-at transformation.

**Parameters**:
- `eye`: Eye position `[num_envs, 3]`
- `target`: Target position `[num_envs, 3]`
- `up`: Up vector `[num_envs, 3]`

**Returns**: Look-at quaternion `[num_envs, 4]`

**Usage**:
```python
# Compute hand orientation to look at object
from dexhand_env.utils.torch_jit_utils import lookat_quaternion
look_quat = lookat_quaternion(hand_pos, object_pos, up_vector)
```

### Coordinate System Validation

#### `validate_hand_position(world_pos: torch.Tensor, spawn_pos: torch.Tensor) -> None`
**Usage Pattern**: Debugging helper (not in codebase)

Validate hand position makes sense relative to spawn position.

**Parameters**:
- `world_pos`: Current world position `[3]`
- `spawn_pos`: Spawn position `[3]`

**Usage**:
```python
# Debug coordinate transformations
def validate_hand_position(self, world_pos, spawn_pos):
    relative_pos = world_pos - spawn_pos
    assert relative_pos[2] > -0.5, f"Hand too low: {world_pos[2]:.3f}"
    assert relative_pos[2] < 1.0, f"Hand too high: {world_pos[2]:.3f}"
```

#### `log_coordinate_info() -> None`
**Usage Pattern**: Debugging helper (not in codebase)

Log coordinate transformation information for debugging.

**Usage**:
```python
# Debug coordinate transformations
def log_coordinate_info(self):
    spawn_pos = self.initial_hand_pos
    current_relative = self.hand_relative_pos[0]  # First env
    current_world = spawn_pos + current_relative
    logger.info(f"Spawn: {spawn_pos}, Relative: {current_relative}, World: {current_world}")
```

### Key Configuration Properties

#### `initialHandPos -> List[float]`
**File**: `dexhand_env/cfg/task/BaseTask.yaml`

Initial hand spawn position in world coordinates.

**Format**: `[x, y, z]` in meters

**Usage**:
```yaml
env:
  initialHandPos: [0.0, 0.0, 0.25]  # 25cm above table
```

#### `hand_pose_arr_aligned -> torch.Tensor`
**File**: `dexhand_env/components/observation_encoder.py`

Hand pose with 90° Y-axis rotation removed for standard coordinate frame.

**Shape**: `[num_envs, 7]` (position + quaternion)

**Usage**:
```python
# Use aligned pose for control algorithms
aligned_orientation = observations["hand_pose_arr_aligned"][3:7]
```

This coordinate system understanding is crucial for proper task design, reward engineering, and debugging motion issues in the DexHand environment.
