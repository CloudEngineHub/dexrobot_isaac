# DOF Index to Joint Name Mapping

This document provides the complete mapping between DOF indices and joint names for the DexHand021 model in Isaac Gym.

## Overview

The DexHand021 model contains **26 total DOFs**:
- **6 Base DOFs**: Control translation and rotation of the hand base
- **20 Finger DOFs**: Control the 5 fingers with 4 DOFs each

## Complete DOF Mapping

| DOF Index | Joint Name      | Type   | Description                    |
|-----------|-----------------|--------|--------------------------------|
| 0         | ARTx            | BASE   | Translation along X-axis       |
| 1         | ARTy            | BASE   | Translation along Y-axis       |
| 2         | ARTz            | BASE   | Translation along Z-axis       |
| 3         | ARRx            | BASE   | Rotation around X-axis         |
| 4         | ARRy            | BASE   | Rotation around Y-axis         |
| 5         | ARRz            | BASE   | Rotation around Z-axis         |
| 6         | r_f_joint1_1    | FINGER | Finger 1, Joint 1              |
| 7         | r_f_joint1_2    | FINGER | Finger 1, Joint 2              |
| 8         | r_f_joint1_3    | FINGER | Finger 1, Joint 3              |
| 9         | r_f_joint1_4    | FINGER | Finger 1, Joint 4              |
| 10        | r_f_joint2_1    | FINGER | Finger 2 (Index), Joint 1      |
| 11        | r_f_joint2_2    | FINGER | Finger 2 (Index), Joint 2      |
| 12        | r_f_joint2_3    | FINGER | Finger 2 (Index), Joint 3      |
| 13        | r_f_joint2_4    | FINGER | Finger 2 (Index), Joint 4      |
| 14        | r_f_joint3_1    | FINGER | Finger 3 (Middle), Joint 1     |
| 15        | r_f_joint3_2    | FINGER | Finger 3 (Middle), Joint 2     |
| 16        | r_f_joint3_3    | FINGER | Finger 3 (Middle), Joint 3     |
| 17        | r_f_joint3_4    | FINGER | Finger 3 (Middle), Joint 4     |
| 18        | r_f_joint4_1    | FINGER | Finger 4 (Ring), Joint 1       |
| 19        | r_f_joint4_2    | FINGER | Finger 4 (Ring), Joint 2       |
| 20        | r_f_joint4_3    | FINGER | Finger 4 (Ring), Joint 3       |
| 21        | r_f_joint4_4    | FINGER | Finger 4 (Ring), Joint 4       |
| 22        | r_f_joint5_1    | FINGER | Finger 5 (Pinky), Joint 1      |
| 23        | r_f_joint5_2    | FINGER | Finger 5 (Pinky), Joint 2      |
| 24        | r_f_joint5_3    | FINGER | Finger 5 (Pinky), Joint 3      |
| 25        | r_f_joint5_4    | FINGER | Finger 5 (Pinky), Joint 4      |

## DOF Groups

### Base DOFs (0-5)
The base DOFs control the position and orientation of the entire hand:
- **Translation DOFs (0-2)**: ARTx, ARTy, ARTz
  - Control the position of the hand in 3D space
- **Rotation DOFs (3-5)**: ARRx, ARRy, ARRz  
  - Control the orientation of the hand (roll, pitch, yaw)

### Finger DOFs (6-25)
Each finger has 4 DOFs controlling its articulation:
- **Joint 1**: Base joint of the finger (closest to palm)
- **Joint 2**: Second joint
- **Joint 3**: Third joint  
- **Joint 4**: Tip joint (fingertip)

## Control Notes

- **Action Space**: The environment may use a subset of these 26 DOFs for control
- **Position Control**: All DOFs support position control mode
- **Position Delta Control**: All DOFs support position delta control mode
- **Fixed Base**: The hand model uses `fix_base_link = True`, meaning the base DOFs control movement relative to the world frame while keeping the hand anchored

## Technical Details

- **Asset DOF Count**: 26 (verified from Isaac Gym asset loading)
- **Model Source**: `dexhand021_right_simplified_floating.xml`
- **Floating Base Fix**: ARTx DOF was restored by adding a `floating_base_root` wrapper in the MJCF model
- **Isaac Gym Version**: Compatible with current Isaac Gym installation

## Verification

This mapping is automatically verified during environment initialization. The console output shows:

```
===== DOF NAMES VERIFICATION =====
Total DOFs found: 26
DOF Index -> Joint Name:
   0: ARTx                 (BASE)
   1: ARTy                 (BASE)
   ...
```

## Related Files

- Model definition: `assets/dexrobot_mujoco/dexrobot_mujoco/models/dexhand021_right_simplified_floating.xml`
- Floating base fix: `assets/dexrobot_mujoco/dexrobot_mujoco/models/floating_base.xml`
- DOF verification code: `dex_hand_env/components/hand_initializer.py:243-261`