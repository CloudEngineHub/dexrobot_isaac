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
**Status**: Open  
**Description**: Middle finger MCP position (`mf_mcp_pos`) consistently reads 0, indicating incorrect data retrieval.  
**Possible Causes**:
- Control name `"mf_mcp"` not found in `control_name_to_index` mapping
- Active finger DOF indices computation error in `_compute_active_finger_dof_indices()`
- Coupling mapping issue between action space and DOF space
- DOF not being actuated, so it stays at default position

**Investigation**: 
- Verify `control_name_to_index` contains `"mf_mcp"`
- Check if other finger DOFs (e.g., `th_rot_pos`) also read 0
- Print DOF position tensor directly to verify raw values

### 3. Hand Pose Always at Spawn Point
**Status**: Open  
**Description**: `hand_pos` and `hand_quat` remain constant at spawn location despite hand movement.  
**Root Cause**: Reading pose from hand actor instead of hand base link.  
**Technical Details**:
- Current code: `root_state_tensor[env_idx, hand_idx, :3]` (hand actor)
- Needed: Pose of hand base link within the hand actor
- Hand actor is fixed in world, hand base link moves via DOFs

**Fix Required**: 
- Use rigid body states to get pose of specific hand base link
- Identify correct rigid body index for hand base link
- Update `_compute_default_observations()` in `observation_encoder.py`

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
**Status**: Open  
**Description**: Coordinate frame transformation from world to hand frame appears incorrect.  
**Root Cause**: Likely using hand actor pose instead of hand base link pose as reference frame.  
**Technical Details**:
- Transformation requires hand base link pose, not hand actor pose
- Hand actor is fixed, so transformations would be identity
- Need pose of moving hand base link for proper coordinate transformation

**Fix Required**:
- Update `_transform_poses_to_hand_frame()` in `observation_encoder.py`
- Use hand base link pose instead of hand actor pose
- Coordinate with fix for Issue #3

### 7. Missing Scaling Between ff_spr and joint5_1
**Status**: Open  
**Description**: Expected scaling relationship between active finger control `ff_spr` and raw DOF `r_f_joint5_1` not observed.  
**Expected Behavior**: `ff_spr` should have 2x scaling factor relative to `joint5_1` based on coupling map.  
**Possible Causes**:
- Coupling scaling not applied correctly in action processing
- Wrong DOF being read for `joint5_1`
- Finger spread action not being applied properly

**Investigation**:
- Check `finger_coupling_map` in `ActionProcessor`
- Verify mapping: `3: [("r_f_joint2_1", 1.0), ("r_f_joint4_1", 1.0), ("r_f_joint5_1", 2.0)]`
- Confirm `ff_spr` corresponds to action index 3

## Next Steps

1. **Immediate**: Fix hand pose reading (Issues #3, #6) - most critical for coordinate transformations
2. **High Priority**: Debug DOF reading issues (#2, #7) - affects finger control validation  
3. **Medium Priority**: Investigate velocity issues (#1, #5) - affects action-observation consistency
4. **Future**: Add contact force testing (#4) - requires scene modifications

## Code Locations

- **Hand pose reading**: `dex_hand_env/components/observation_encoder.py:276-295`
- **DOF access**: `dex_hand_env/components/observation_encoder.py:778-828`
- **Frame transformations**: `dex_hand_env/components/observation_encoder.py:493-543`
- **Action scaling**: `dex_hand_env/components/action_processor.py:720-747`
- **Coupling mapping**: `dex_hand_env/components/action_processor.py:75-88`