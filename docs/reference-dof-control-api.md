# DOF and Action Control API Reference

This reference explains the complex DOF-to-action mapping that enables intuitive control of a 26-DOF dexterous hand with just 12-18 actions.

## The Problem: Too Many DOFs for Direct Control

The DexHand021 has 26 degrees of freedom:
- 6 DOFs for hand position/orientation in 3D space
- 20 DOFs across 5 fingers (4 joints per finger)

Directly controlling 26 DOFs creates several problems:
1. **Action space explosion**: RL algorithms struggle with high-dimensional actions
2. **Unnatural control**: Humans don't think about individual joint angles
3. **Redundant DOFs**: Many finger joints move together naturally (like when making a fist)
4. **Stability issues**: Independent control of coupled joints leads to unnatural poses

## The Solution: Intelligent Action Coupling

The system reduces 26 DOFs to just 12-18 intuitive actions through biomechanically-inspired coupling:

### Action Space Design
- **12 Actions (default)**: Natural finger control patterns
- **18 Actions (with base)**: Add 6 DOFs for hand movement

### Key Innovation: Finger Coupling
Instead of controlling each joint independently, actions map to natural grasp patterns:
- **DIP coupling**: Fingertip joints move together (you can't bend just your fingertip)
- **Spread coupling**: Multiple fingers spread together (like opening your hand)
- **Scale coupling**: Pinky spreads twice as much as other fingers (anatomically correct)

## Complete DOF Mapping

### DOF Structure Overview

**Total DOFs**: 26
- **Base DOFs (0-5)**: Hand translation and rotation
- **Finger DOFs (6-25)**: 5 fingers × 4 joints each

**Action Space**: 12 (default finger-only) or 18 (with base control)
- **Base Actions (0-5)**: Optional base control
- **Finger Actions (6-17)**: 12 coupled finger controls

### Detailed DOF Table

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
| 9         | r_f_joint1_4    | FINGER | Thumb Joint 4 (tip)            | ❌ Coupled with joint 3 |
| 10        | r_f_joint2_1    | FINGER | Index Joint 1 (spread)         | 9 (or 3 if base disabled) |
| 11        | r_f_joint2_2    | FINGER | Index Joint 2                  | 10 (or 4 if base disabled) |
| 12        | r_f_joint2_3    | FINGER | Index Joint 3                  | 11 (or 5 if base disabled) |
| 13        | r_f_joint2_4    | FINGER | Index Joint 4 (tip)            | ❌ Coupled with joint 3 |
| 14        | r_f_joint3_1    | FINGER | Middle Joint 1 (fixed)         | ❌ Fixed joint |
| 15        | r_f_joint3_2    | FINGER | Middle Joint 2                 | 12 (or 6 if base disabled) |
| 16        | r_f_joint3_3    | FINGER | Middle Joint 3                 | 13 (or 7 if base disabled) |
| 17        | r_f_joint3_4    | FINGER | Middle Joint 4 (tip)           | ❌ Coupled with joint 3 |
| 18        | r_f_joint4_1    | FINGER | Ring Joint 1 (spread)          | 9 (or 3 if base disabled) |
| 19        | r_f_joint4_2    | FINGER | Ring Joint 2                   | 14 (or 8 if base disabled) |
| 20        | r_f_joint4_3    | FINGER | Ring Joint 3                   | 15 (or 9 if base disabled) |
| 21        | r_f_joint4_4    | FINGER | Ring Joint 4 (tip)             | ❌ Coupled with joint 3 |
| 22        | r_f_joint5_1    | FINGER | Pinky Joint 1 (spread)         | 9 (or 3 if base disabled) 2x |
| 23        | r_f_joint5_2    | FINGER | Pinky Joint 2                  | 16 (or 10 if base disabled) |
| 24        | r_f_joint5_3    | FINGER | Pinky Joint 3                  | 17 (or 11 if base disabled) |
| 25        | r_f_joint5_4    | FINGER | Pinky Joint 4 (tip)            | ❌ Coupled with joint 3 |

## Action-to-DOF Coupling System

### Why Coupling Matters

Natural hand movements involve coordinated joint motion:
- When you curl a finger, the tip joints move together
- When you spread your fingers, they move in a coordinated pattern
- The pinky finger naturally spreads more than other fingers

The coupling system encodes these biomechanical constraints.

### Finger Coupling Details

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

### Control Naming Convention
- **th_**: Thumb (rot=rotation/spread, mcp=metacarpophalangeal, dip=distal interphalangeal)
- **ff_**: First finger/index (spr=spread, mcp, dip)
- **mf_**: Middle finger (mcp, dip)
- **rf_**: Ring finger (mcp, dip)
- **lf_**: Little finger/pinky (mcp, dip)

### Key Coupling Behaviors

#### DIP Coupling (Fingertip Control)
Each finger's `*_dip` action controls both distal joints together:
```
Action th_dip → r_f_joint1_3 + r_f_joint1_4 (move together)
Action ff_dip → r_f_joint2_3 + r_f_joint2_4 (move together)
```
This mirrors human anatomy where you can't bend just your fingertip independently.

#### Spread Coupling (Hand Opening)
Action 9/3 (`ff_spr`) controls multiple fingers spreading:
```
One action → Three fingers spread:
- Index finger: 1.0x motion
- Ring finger: 1.0x motion
- Pinky finger: 2.0x motion (anatomically correct)
```

## Action Scaling Mathematics

### From Actions to Joint Angles

Actions range from -1 to +1 and map to joint limits:

```python
# Map action from [-1, +1] to [dof_min, dof_max]
scaled_action = (action_value + 1.0) * 0.5 * (dof_max - dof_min) + dof_min
final_target = scaled_action * coupling_scale
```

### Example: Finger Spread Control

For the spread action (Action 9/3):
- **Action = -1.0** (closed hand):
  - Index spread: 0.0 rad
  - Ring spread: 0.0 rad
  - Pinky spread: 0.0 rad

- **Action = 0.0** (neutral):
  - Index spread: 0.15 rad
  - Ring spread: 0.15 rad
  - Pinky spread: 0.3 rad (2x scale)

- **Action = +1.0** (spread fingers):
  - Index spread: 0.3 rad
  - Ring spread: 0.3 rad
  - Pinky spread: 0.6 rad (2x scale)

## Control Configuration

### Enabling/Disabling Control Groups

```yaml
task:
  policy_controls_hand_base: true   # Enable base DOF control (actions 0-5)
  policy_controls_fingers: true     # Enable finger DOF control (actions 6-17 or 0-11)
```

### Control Modes

**Position Control**: Actions specify absolute target positions
- Good for precise, repeatable motions
- Actions directly map to desired joint angles

**Position Delta Control**: Actions specify relative changes
- Better for smooth, reactive control
- Actions add to current positions each timestep

## Key Observations for Control

The observation system provides feedback for closed-loop control:

- **`base_dof_pos/vel/target`**: Current base state (6D each)
- **`active_finger_dof_pos/vel/target`**: Current finger state (12D each)
- **`hand_pose_arr_aligned`**: Hand pose aligned with ARR DOFs (see DESIGN_DECISIONS.md)

## Implementation Reference

### Core Methods

#### `process_actions(actions, active_rule_targets) -> bool`
Main entry point for action processing. Handles scaling, coupling, and applies to simulation.

#### `apply_pre_action_rule(active_prev_targets, state) -> Tensor`
Applies rule-based behaviors before policy actions (e.g., default positions, safety constraints).

#### `finalize_setup() -> None`
Completes two-stage initialization after control_dt measurement.

### Key Files
- **Action Processing**: `dexhand_env/components/action_processor.py`
- **Coupling Rules**: `dexhand_env/components/action_rules.py`
- **Configuration**: `dexhand_env/cfg/task/BaseTask.yaml`

## Summary

The DOF control system transforms 26 raw degrees of freedom into 12-18 intuitive actions through biomechanically-inspired coupling. This design enables:

1. **Natural control**: Actions map to human-like grasping patterns
2. **Efficient learning**: Reduced action space accelerates RL training
3. **Stable behavior**: Coupled joints prevent unnatural configurations
4. **Flexible deployment**: Easy to add task-specific action rules

The coupling system is the key innovation that makes learning dexterous manipulation tractable.

## See Also

- **[Observation System](guide-observation-system.md)** - Feedback for closed-loop control
- **[Design Decisions](DESIGN_DECISIONS.md)** - Why coordinates work this way
- **[Component Initialization](guide-component-initialization.md)** - Two-stage initialization details
