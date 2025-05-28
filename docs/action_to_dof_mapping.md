# Action-to-DOF Mapping Documentation

This document provides the verified mapping between policy actions and actual DOF indices for the DexHand021 model.

## Overview

The environment uses **18 actions** in both `position` and `position_delta` control modes:
- **6 Base Actions**: Control hand base translation and rotation
- **12 Finger Actions**: Control 3 joints for each of 4 fingers (excluding thumb joint 4 and pinky)

## Action-to-DOF Mapping Table

| Action Index | Action Target Joint | Description | Actual DOF Index | DOF Name | Notes |
|--------------|---------------------|-------------|------------------|----------|-------|
| 0 | ARTx | Base Translation X | 0 | ARTx | ✓ Direct mapping |
| 1 | ARTy | Base Translation Y | 1 | ARTy | ✓ Direct mapping |
| 2 | ARTz | Base Translation Z | 2 | ARTz | ✓ Direct mapping |
| 3 | ARRx | Base Rotation X | 3 | ARRx | ✓ Direct mapping |
| 4 | ARRy | Base Rotation Y | 4 | ARRy | ✓ Direct mapping |
| 5 | ARRz | Base Rotation Z | 5 | ARRz | ✓ Direct mapping |
| 6 | r_f_joint1_1 | Thumb Joint 1 | 6 | r_f_joint1_1 | Maps to finger 1 |
| 7 | r_f_joint1_2 | Thumb Joint 2 | 7 | r_f_joint1_2 | Maps to finger 1 |
| 8 | r_f_joint1_3 | Thumb Joint 3 | 8 | r_f_joint1_3 | Maps to finger 1 |
| 9 | r_f_joint2_1 | Index Joint 1 | 10 | r_f_joint2_1 | Maps to finger 2 |
| 10 | r_f_joint2_2 | Index Joint 2 | 11 | r_f_joint2_2 | Maps to finger 2 |
| 11 | r_f_joint2_3 | Index Joint 3 | 12 | r_f_joint2_3 | Maps to finger 2 |
| 12 | r_f_joint3_1 | Middle Joint 1 | 14 | r_f_joint3_1 | Maps to finger 3 |
| 13 | r_f_joint3_2 | Middle Joint 2 | 15 | r_f_joint3_2 | Maps to finger 3 |
| 14 | r_f_joint3_3 | Middle Joint 3 | 16 | r_f_joint3_3 | Maps to finger 3 |
| 15 | r_f_joint4_1 | Ring Joint 1 | 18 | r_f_joint4_1 | Maps to finger 4 |
| 16 | r_f_joint4_2 | Ring Joint 2 | 19 | r_f_joint4_2 | Maps to finger 4 |
| 17 | r_f_joint4_3 | Ring Joint 3 | 20 | r_f_joint4_3 | Maps to finger 4 |

## Control Configuration

### Base DOFs (Actions 0-5)
- **Direct Control**: All 6 base actions map directly to the corresponding DOF indices 0-5
- **Position Control**: Actions specify absolute position targets
- **Position Delta Control**: Actions specify incremental changes from current position

### Finger DOFs (Actions 6-17)
- **Selective Control**: Only 12 out of 20 finger DOFs are controlled by the policy
- **Excluded DOFs**: 
  - r_f_joint1_4 (Thumb tip joint - DOF 9)
  - r_f_joint2_4 (Index tip joint - DOF 13)
  - r_f_joint3_4 (Middle tip joint - DOF 17)
  - r_f_joint4_4 (Ring tip joint - DOF 21)
  - r_f_joint5_* (All pinky joints - DOFs 22-25)

### Action Processing
The action processor component handles the mapping between policy actions and DOF targets:

1. **Base Actions**: Direct 1:1 mapping to DOF indices 0-5
2. **Finger Actions**: Mapped through `joint_to_control` and `active_joint_names` arrays
3. **Uncontrolled DOFs**: Set to default targets or task-specific targets

## Implementation Details

### Key Components
- **ActionProcessor**: `dex_hand_env/components/action_processor.py`
- **HandInitializer**: `dex_hand_env/components/hand_initializer.py` 
- **Configuration**: `dex_hand_env/cfg/task/BaseTask.yaml`

### Active Joint Names
The environment uses these 12 active joint names for finger control:
```python
active_joint_names = [
    "r_f_joint1_1", "r_f_joint1_2", "r_f_joint1_3",  # Thumb (3 joints)
    "r_f_joint2_1", "r_f_joint2_2", "r_f_joint2_3",  # Index (3 joints)  
    "r_f_joint3_1", "r_f_joint3_2", "r_f_joint3_3",  # Middle (3 joints)
    "r_f_joint4_1", "r_f_joint4_2", "r_f_joint4_3",  # Ring (3 joints)
]
```

### Configuration Settings
```yaml
env:
  controlMode: "position"        # or "position_delta"
  controlHandBase: true          # Enable base DOF control
  controlFingers: true           # Enable finger DOF control
```

## Verification Status

- ✅ **DOF Names**: All 26 DOFs correctly identified and named
- ✅ **Action Space**: 18 actions correctly configured
- ✅ **Base Mapping**: Direct mapping for base DOFs verified
- ⚠️ **Finger Mapping**: Requires further testing to verify actual movement
- ⚠️ **Position Control**: Movement verification pending

## Usage Notes

1. **Action Range**: Actions are typically in the range [-1, 1] and scaled by the action processor
2. **Safety Limits**: DOF position limits are enforced by the action processor
3. **Control Frequency**: Actions are applied at the control frequency (typically 50Hz)
4. **Reset Behavior**: DOFs reset to default positions during environment reset

## Related Files

- DOF mapping: `/docs/dof_mapping.md`
- Action processor: `/dex_hand_env/components/action_processor.py`
- Hand initializer: `/dex_hand_env/components/hand_initializer.py` 
- Base configuration: `/dex_hand_env/cfg/task/BaseTask.yaml`
- Test script: `/examples/dexhand_test.py`