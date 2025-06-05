# Observation System Issues

This document tracks issues found during real-time plotting validation of the observation system.

## Issues Identified

### 1. ARR Velocities Look Messy
**Status**: Open  
**Description**: The angular velocities (ARRx, ARRy, ARRz) appear noisy or erratic in the plots.  
**Possible Causes**:
- Angular velocity computation in Isaac Gym may be sensitive to numerical precision
- Base DOF velocity extraction may be incorrect
- Physics timestep too large causing instability in angular velocity calculation
- DOF indexing issue (wrong DOF being read)

**Investigation**: Check if linear velocities (ARTx, ARTy, ARTz) are also messy to isolate linear vs angular issue.

### 2. mf_mcp_pos Value Always 0
**Status**: ✅ **SOLVED**  
**Description**: Middle finger MCP position (`mf_mcp_pos`) consistently reads 0, indicating incorrect data retrieval.  
**Root Cause**: The `_compute_active_finger_dof_indices()` method had a bug where multiple DOFs mapping to the same control would overwrite each other, leaving some controls unmapped.

**Fix Applied**:
- ✅ Updated `_compute_active_finger_dof_indices()` to use **primary DOF** (first joint) for each control
- ✅ `mf_mcp` now correctly maps to `r_f_joint3_2` (DOF index 15)
- ✅ Fixed architectural mismatch between ActionProcessor (many-to-many) and ObservationEncoder (1:1) 
- ✅ All 12 finger controls now properly mapped to their primary joints

### 3. Hand Pose Always at Spawn Point
**Status**: ✅ **SOLVED**  
**Description**: `hand_pos` and `hand_quat` remain constant at spawn location despite hand movement.  
**Root Cause**: Reading pose from hand actor instead of hand base link.  
**Technical Details**:
- Current code: `root_state_tensor[env_idx, hand_idx, :3]` (hand actor)
- Needed: Pose of hand base link within the hand actor
- Hand actor is fixed in world, hand base link moves via DOFs

**Fix Applied**: 
- ✅ Updated to use `rigid_body_states` tensor instead of `actor_root_state_tensor`
- ✅ Fixed Isaac Gym tensor naming throughout codebase
- ✅ Implemented vectorized hand pose extraction using `rigid_body_states[:, hand_base_idx, :7]`
- ✅ Updated `_compute_default_observations()` in `observation_encoder.py`

### 4. No Contact Forces
**Status**: Expected (No objects in scene)  
**Description**: Contact forces are zero, which is correct for current test setup.  
**Future Work**: Add objects to scene to test contact force observation.

### 5. Action Scaling Discrepancy
**Status**: Open  
**Description**: `unscaled_artx` is over twice `artx_vel_times_dt`.  
**Possible Causes**:
- `control_dt` vs `physics_dt` mismatch in scaling
- Actuator performance limitations (PD controller can't achieve commanded velocities)
- Action scaling coefficients in `unscale_actions()` incorrect
- Integration error: velocity should be integrated over actual timestep, not control_dt

**Investigation**:
- Compare `control_dt` vs actual physics timestep
- Check if DOF actually reaches target positions
- Verify action scaling math in `ActionProcessor.unscale_actions()`

### 6. World to Hand Frame Transformation Wrong
**Status**: ✅ **SOLVED**  
**Description**: Coordinate frame transformation from world to hand frame appears incorrect.  
**Root Cause**: Likely using hand actor pose instead of hand base link pose as reference frame.  
**Technical Details**:
- Transformation requires hand base link pose, not hand actor pose
- Hand actor is fixed, so transformations would be identity
- Need pose of moving hand base link for proper coordinate transformation

**Fix Applied**:
- ✅ Updated `_transform_poses_to_hand_frame()` in `observation_encoder.py`
- ✅ Now uses hand base link pose from `rigid_body_states` tensor
- ✅ Implemented vectorized coordinate transformations without loops
- ✅ Fixed to work with correct tensor shapes and Isaac Gym naming

### 7. Missing Scaling Between ff_spr and joint5_1
**Status**: ✅ **SOLVED**  
**Description**: Expected scaling relationship between active finger control `ff_spr` and raw DOF `r_f_joint5_1` not observed.  
**Root Cause**: Same issue as #2 - the observation encoder was incorrectly mapping multi-DOF controls to finger observations, plus a critical bug where DOF targets were reset to current positions every step.

**Fix Applied**:
- ✅ Fixed `_compute_active_finger_dof_indices()` to use primary DOF (`r_f_joint2_1`) for `ff_spr` observations
- ✅ **CRITICAL**: Fixed ActionProcessor target persistence bug where all targets were reset to current positions every step
- ✅ DOF targets now properly persist between steps unless actively commanded
- ✅ `ff_spr` observations now correctly read from `r_f_joint2_1` while ActionProcessor still applies 2x scaling to `r_f_joint5_1`

### 8. Quaternion to Euler Conversion for Human Inspection
**Status**: Enhancement (Low Priority)  
**Description**: Quaternion rotations in observations (hand pose, fingertip poses, fingerpad poses) are difficult for humans to interpret during debugging and visualization.  
**Proposed Enhancement**: Convert quaternions to Euler angles in observation plotting for more intuitive human inspection.  
**Benefits**:
- More intuitive rotation values for debugging
- Easier to understand hand orientation changes
- Better visualization in plotting tools
- Simplified manual inspection of rotation observations

**Implementation Notes**:
- Add conversion utility for plotting/debugging purposes only
- Keep quaternions in actual observation tensors for RL algorithms
- Apply to hand pose, fingertip poses, and fingerpad poses in plotting tools

## Next Steps

1. ✅ **COMPLETED**: Fixed hand pose reading (Issues #3, #6) - critical for coordinate transformations
2. ✅ **COMPLETED**: Fixed DOF reading issues (#2, #7) - critical target persistence bug resolved
3. **High Priority**: Investigate velocity issues (#1, #5) - affects action-observation consistency
4. **Future**: Add contact force testing (#4) - requires scene modifications
5. **Enhancement**: Convert quaternions to Euler angles for plotting (#8) - improves debugging experience

## Code Locations

- **Hand pose reading**: `dex_hand_env/components/observation_encoder.py:276-295`
- **DOF access**: `dex_hand_env/components/observation_encoder.py:778-828`
- **Frame transformations**: `dex_hand_env/components/observation_encoder.py:493-543`
- **Action scaling**: `dex_hand_env/components/action_processor.py:720-747`
- **Coupling mapping**: `dex_hand_env/components/action_processor.py:75-88`