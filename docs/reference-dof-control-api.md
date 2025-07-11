# DOF and Action Control API Reference

Complete reference for DOF indices, joint names, and action mapping for the DexHand021 model.

## DOF Structure Overview

**Total DOFs**: 26
- **Base DOFs (0-5)**: Hand translation and rotation
- **Finger DOFs (6-25)**: 5 fingers × 4 joints each

**Action Space**: 12 (default finger-only) or 18 (with base control)
- **Base Actions (0-5)**: Optional base control
- **Finger Actions (6-17)**: 12 coupled finger controls

## Complete DOF Mapping

| DOF Index | Joint Name      | Type   | Description                    | Controlled by Action |
|-----------|-----------------|--------|--------------------------------|---------------------|
| 0         | ARTx            | BASE   | Translation along X-axis       | 0 (if base enabled) |
| 1         | ARTy            | BASE   | Translation along Y-axis       | 1 (if base enabled) |
| 2         | ARTz            | BASE   | Translation along Z-axis       | 2 (if base enabled) |
| 3         | ARRx            | BASE   | Rotation around X-axis         | 3 (if base enabled) |
| 4         | ARRy            | BASE   | Rotation around Y-axis         | 4 (if base enabled) |
| 5         | ARRz            | BASE   | Rotation around Z-axis         | 5 (if base enabled) |
| 6         | r_f_joint1_1    | FINGER | Thumb Joint 1                  | 6 (or 0 if base disabled) |
| 7         | r_f_joint1_2    | FINGER | Thumb Joint 2                  | 7 (or 1 if base disabled) |
| 8         | r_f_joint1_3    | FINGER | Thumb Joint 3                  | 8 (or 2 if base disabled) |
| 9         | r_f_joint1_4    | FINGER | Thumb Joint 4 (tip)            | ❌ Not controlled |
| 10        | r_f_joint2_1    | FINGER | Index Joint 1 (spread)         | 9 (or 3 if base disabled) via coupling |
| 11        | r_f_joint2_2    | FINGER | Index Joint 2                  | 10 (or 4 if base disabled) |
| 12        | r_f_joint2_3    | FINGER | Index Joint 3                  | 11 (or 5 if base disabled) |
| 13        | r_f_joint2_4    | FINGER | Index Joint 4 (tip)            | ❌ Not controlled |
| 14        | r_f_joint3_1    | FINGER | Middle Joint 1 (fixed)         | ❌ Fixed joint |
| 15        | r_f_joint3_2    | FINGER | Middle Joint 2                 | 12 (or 6 if base disabled) |
| 16        | r_f_joint3_3    | FINGER | Middle Joint 3                 | 13 (or 7 if base disabled) |
| 17        | r_f_joint3_4    | FINGER | Middle Joint 4 (tip)           | ❌ Not controlled |
| 18        | r_f_joint4_1    | FINGER | Ring Joint 1 (spread)          | 9 (or 3 if base disabled) via coupling |
| 19        | r_f_joint4_2    | FINGER | Ring Joint 2                   | 14 (or 8 if base disabled) |
| 20        | r_f_joint4_3    | FINGER | Ring Joint 3                   | 15 (or 9 if base disabled) |
| 21        | r_f_joint4_4    | FINGER | Ring Joint 4 (tip)             | ❌ Not controlled |
| 22        | r_f_joint5_1    | FINGER | Pinky Joint 1 (spread)         | 9 (or 3 if base disabled) via coupling 2x |
| 23        | r_f_joint5_2    | FINGER | Pinky Joint 2                  | 16 (or 10 if base disabled) |
| 24        | r_f_joint5_3    | FINGER | Pinky Joint 3                  | 17 (or 11 if base disabled) |
| 25        | r_f_joint5_4    | FINGER | Pinky Joint 4 (tip)            | ❌ Not controlled |

## Action-to-DOF Coupling

### Finger Coupling System
The 12 finger actions control 19 finger DOFs through coupling:

| Action (base enabled/disabled) | Control Name | Primary DOF | Coupled DOFs | Coupling Scale |
|-------------------------------|--------------|-------------|--------------|----------------|
| 6 / 0  | th_rot | r_f_joint1_1 | None | 1.0 |
| 7 / 1  | th_mcp | r_f_joint1_2 | None | 1.0 |
| 8 / 2  | th_dip | r_f_joint1_3 | r_f_joint1_4 | 1.0, 1.0 |
| 9 / 3  | ff_spr | r_f_joint2_1 | r_f_joint4_1, r_f_joint5_1 | 1.0, 1.0, 2.0 |
| 10 / 4 | ff_mcp | r_f_joint2_2 | None | 1.0 |
| 11 / 5 | ff_dip | r_f_joint2_3 | r_f_joint2_4 | 1.0, 1.0 |
| 12 / 6 | mf_mcp | r_f_joint3_2 | None | 1.0 |
| 13 / 7 | mf_dip | r_f_joint3_3 | r_f_joint3_4 | 1.0, 1.0 |
| 14 / 8 | rf_mcp | r_f_joint4_2 | None | 1.0 |
| 15 / 9 | rf_dip | r_f_joint4_3 | r_f_joint4_4 | 1.0, 1.0 |
| 16 / 10 | lf_mcp | r_f_joint5_2 | None | 1.0 |
| 17 / 11 | lf_dip | r_f_joint5_3 | r_f_joint5_4 | 1.0, 1.0 |

### Control Naming Convention:
- **th_**: Thumb controls (rot=rotation/spread, mcp=metacarpophalangeal, dip=distal interphalangeal)
- **ff_**: First finger (index) controls (spr=spread, mcp, dip)
- **mf_**: Middle finger controls (mcp, dip)
- **rf_**: Ring finger controls (mcp, dip)
- **lf_**: Little finger (pinky) controls (mcp, dip)

### Key Coupling Details:
- **DIP Controls**: Each finger's `*_dip` control simultaneously moves both the `*_3` and `*_4` joints (coupled 1:1)
- **Finger Spread (Action 9/3)**: Controls spread motion of index, ring, and pinky fingers simultaneously
  - r_f_joint2_1 (index) moves with 1.0x scale
  - r_f_joint4_1 (ring) moves with 1.0x scale
  - r_f_joint5_1 (pinky) moves with 2.0x scale (twice the motion)

## Action Scaling Formula

Actions are scaled from `[-1, +1]` to DOF limit ranges:

```python
# Map action from [-1, +1] to [dof_min, dof_max]
scaled_action = (action_value + 1.0) * 0.5 * (dof_max - dof_min) + dof_min
final_target = scaled_action * coupling_scale
```

### DOF Limit Examples
- **Spread joints (2_1, 4_1)**: `[0.0, 0.3]` radians
- **Spread joint (5_1)**: `[0.0, 0.6]` radians (accommodates 2x coupling scale)
- **Bend joints**: `[0.0, 1.57]` radians (π/2)

### Action Examples for Finger Spread (Action 9/3)
- `action = -1.0`:
  - r_f_joint2_1 → 0.0 rad
  - r_f_joint4_1 → 0.0 rad
  - r_f_joint5_1 → 0.0 rad
- `action = 0.0`:
  - r_f_joint2_1 → 0.15 rad
  - r_f_joint4_1 → 0.15 rad
  - r_f_joint5_1 → 0.3 rad (2x scale)
- `action = +1.0`:
  - r_f_joint2_1 → 0.3 rad
  - r_f_joint4_1 → 0.3 rad
  - r_f_joint5_1 → 0.6 rad (2x scale)

## Control Configuration

### Base Control Toggle
```yaml
env:
  controlHandBase: true   # Enable base DOF control (actions 0-5)
  controlFingers: true    # Enable finger DOF control (actions 6-17 or 0-11)
```

### Control Modes
- **Position Control**: Actions specify absolute position targets
- **Position Delta Control**: Actions specify incremental changes from current position

## Related Observations

See [`guide-observation-system.md`](guide-observation-system.md) for complete observation documentation.

Key observations for DOF control:
- `base_dof_pos/vel/target`: Base DOF states (6D each)
- `active_finger_dof_pos/vel/target`: Active finger states (12D each)
- `hand_pose_arr_aligned`: Hand pose with orientation aligned to ARR DOFs (7D)

## Method Signatures

### ActionProcessor Core Methods

#### `process_actions(actions: torch.Tensor, active_rule_targets: torch.Tensor) -> bool`
**File**: `dexhand_env/components/action_processor.py`

Process policy actions through the complete action pipeline.

**Parameters**:
- `actions`: Action tensor from policy `[num_envs, num_actions]`
- `active_rule_targets`: Pre-computed rule targets `[num_envs, 18]` (6 base + 12 finger)

**Returns**: Success flag

**Usage**:
```python
# Typical usage in environment step
success = self.action_processor.process_actions(actions, rule_targets)
```

#### `apply_pre_action_rule(active_prev_targets: torch.Tensor, state: Dict[str, Any]) -> torch.Tensor`
**File**: `dexhand_env/components/action_processor.py`

Apply pre-action rule to compute rule-based targets before action processing.

**Parameters**:
- `active_prev_targets`: Previous active targets `[num_envs, 18]`
- `state`: State dictionary with `'obs_dict'` and `'env'` keys

**Returns**: Rule-based targets `[num_envs, 18]`

**Usage**:
```python
# Called before process_actions
rule_targets = self.action_processor.apply_pre_action_rule(prev_targets, state)
```

#### `finalize_setup() -> None`
**File**: `dexhand_env/components/action_processor.py`

Complete two-stage initialization after `control_dt` is available.

**Usage**:
```python
# Called after control_dt measurement
self.action_processor.finalize_setup()
```

### Action Scaling and Limits

#### `_precompute_active_limits() -> None`
**File**: `dexhand_env/components/action_processor.py`

Precompute DOF limits for active DOFs maintaining full 6D+12D structure.

**Internal method** - called during initialization to establish action space scaling.

### DOF Target Properties

#### `current_targets -> torch.Tensor`
**File**: `dexhand_env/components/action_processor.py`

Access current DOF targets applied to simulation.

**Returns**: Current DOF targets `[num_envs, num_dofs]`

**Usage**:
```python
# Access current targets for observations
targets = self.action_processor.current_targets
```

## Implementation Locations

- **Action Processing**: `ActionProcessor.process_actions()`
- **Coupling System**: `ActionRules.apply_coupling_rule()`
- **DOF Initialization**: `HandInitializer.get_dof_names()`
- **Configuration**: `dexhand_env/cfg/task/BaseTask.yaml`
- **Observation System**: `ObservationEncoder.compute_observations()`
