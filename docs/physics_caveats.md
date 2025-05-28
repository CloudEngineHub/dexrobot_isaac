# Physics and Model Configuration Caveats

This document outlines critical design decisions and requirements for the DexHand Isaac Gym environment that may not be immediately obvious to developers.

## 1. Fixed Hand Base Configuration

### Design Decision
The hand model uses a **fixed base link** (`asset_options.fix_base_link = True`) to anchor the hand to the world coordinate system.

### Why This Matters
- **The hand doesn't fall under gravity** when no actions are applied
- **Base movement is controlled through internal DOFs**, not by moving the entire actor
- **ARTx/y/z control translation** relative to the fixed base position
- **ARRx/y/z control rotation** relative to the fixed base orientation

### Code Location
```python
# dex_hand_env/components/hand_initializer.py:181
asset_options.fix_base_link = True
```

### Implications
- When `action=0` is applied to base DOFs, the hand maintains its current pose
- Base DOFs allow controlled movement within their joint limits
- This ensures stable physics simulation without unexpected drift

## 2. ARTz Controls Relative Z-Motion

### Important Caveat
**ARTz does NOT control absolute world Z-position.** It controls **relative motion** from the current hand position.

### What This Means
- **Initial hand position** is set by `initialHandPos` in the config (e.g., `[0.0, 0.0, 0.5]`)
- **ARTz=0.0** means "stay at initial Z position" (the spawn point)
- **ARTz=+0.1** means "move 0.1 units up from initial position"
- **ARTz=-0.1** means "move 0.1 units down from initial position"

### Joint Limits
```xml
<!-- From floating_base.xml -->
<joint name="ARTz" ... range="-1 1" ... />
```
- ARTz can move ±1.0 units relative to the base position
- If hand starts at Z=0.5, it can move between Z=-0.5 and Z=+1.5

### Control Modes
- **Position control**: ARTz directly sets the relative position
- **Position delta control**: ARTz increments are added to current position

## 3. Isaac Gym Requires "limited" Attribute

### Critical Requirement
Isaac Gym **requires** the `limited="true"` attribute in MJCF joint definitions to properly recognize joint limits, even when `range` is specified.

### Problem Without "limited"
```xml
<!-- This WILL NOT work in Isaac Gym -->
<joint name="r_f_joint1_1" range="0 2.2" />
```
Result: `hasLimits=False`, joint limits ignored

### Correct Solution
```xml
<!-- This WILL work in Isaac Gym -->
<joint name="r_f_joint1_1" range="0 2.2" limited="true" />
```
Result: `hasLimits=True`, joint limits properly enforced

### Automatic Handling
Our model generation scripts automatically add `limited="true"` to all joints with `range` attributes:

```python
# dexrobot_mujoco/utils/mjcf_utils.py:add_joint_limits()
for joint in root.findall(".//joint"):
    if joint.get("range") is not None and joint.get("limited") is None:
        joint.set("limited", "true")
```

### Verification
Check that all joints have proper limits:
```python
# Expected output: all finger joints should have hasLimits=True
hasLimits field: [True True True ... True] (shape: (26,))
Number of joints with hasLimits=True: 26  # All joints
Joints without limits: []  # Should be empty for finger joints
```

## 4. Joint Properties from MJCF vs Code

### Current Behavior
Isaac Gym loads stiffness and damping **directly from the MJCF model**:

**Stiffness** comes from actuator `kp` values:
```xml
<position name="act_ARTx" joint="ARTx" kp="10000" ... />  
<!-- Results in: stiff=10000.0 -->
```

**Damping** comes from joint `damping` or defaults:
```xml
<joint name="ARTx" damping="20" ... />  <!-- Explicit -->
<default><joint damping="1"/></default>  <!-- Default for others -->
```

### What Was Removed
Previously, the code **overrode** MJCF values with hardcoded config values:
```yaml
# REMOVED from BaseTask.yaml
baseStiffness: 400.0     # Was overriding MJCF kp="10000"
fingerStiffness: 100.0   # Was overriding MJCF kp="20"
```

### Current Values in Use
```
Base joints:     stiff=10000.0, damp=20.0  (from floating_base.xml)
Finger joints:   stiff=20.0,    damp=1.0   (from actuators + defaults.xml)
```

## 5. DOF Indexing and Control

### DOF Order
The 26 DOFs are ordered as follows:
```
Indices 0-5:   Base DOFs [ARTx, ARTy, ARTz, ARRx, ARRy, ARRz]
Indices 6-25:  Finger DOFs [r_f_joint1_1, r_f_joint1_2, ..., r_f_joint5_4]
```

### Action Space Mapping
The environment provides 18 actions controlling:
```
Actions 0-5:   Base DOFs (6 DOFs)
Actions 6-17:  Finger DOFs (12 DOFs, subset of all 20 finger joints)
```

### Uncontrolled DOFs
- **r_f_joint1_4, r_f_joint2_4, r_f_joint3_4, r_f_joint4_4, r_f_joint5_4**: Fingertip joints
- **r_f_joint3_1**: Middle finger base (fixed joint with tiny range)
- **r_f_joint5_1**: Pinky base joint

These maintain their default positions or are controlled by task-specific logic.

## 6. Physics Simulation Parameters

### Critical PhysX Settings for GPU Pipeline
```yaml
# BaseTask.yaml - Required for GPU pipeline stability
physx:
  contact_collection: 1              # CC_LAST_SUBSTEP - critical for GPU
  default_buffer_size_multiplier: 5.0  # Prevent buffer overflow
  always_use_articulations: true    # Required for hand model
```

### Physics Steps Detection
The environment automatically detects physics steps per control step:
```
Auto-detected physics_steps_per_control_step: 2
```
This accounts for additional physics steps during resets and ensures consistent control timing.

## References

- **Model generation**: `assets/dexrobot_mujoco/scripts/regenerate_all.sh`
- **Joint limits fix**: `dexrobot_mujoco/utils/mjcf_utils.py:add_joint_limits()`
- **Base configuration**: `dex_hand_env/components/hand_initializer.py:181`
- **DOF mapping**: `docs/action_to_dof_mapping.md`

## Troubleshooting

### If joints don't respect limits:
1. Check `hasLimits` field in DOF properties output
2. Verify `limited="true"` in MJCF joint definitions  
3. Regenerate models using `scripts/regenerate_all.sh`

### If hand falls or drifts:
1. Verify `fix_base_link = True` in hand_initializer.py
2. Check that base DOF targets are properly set
3. Ensure action=0 maintains current position

### If physics is unstable:
1. Use `--no-gpu-pipeline` for debugging
2. Check PhysX parameters in config
3. Verify model has proper collision geometry