# fix-001-contact-viz.md

Contact visualization is implemented but not working correctly due to physics data pipeline and timing issues.

## Context

The contact visualization system consists of multiple components that must work together:
- Configuration loading from `task.contactVisualization` in BaseTask.yaml
- Contact force data pipeline from Isaac Gym → TensorManager → ViewerController
- Real-time color updates based on contact force magnitudes
- Keyboard toggle ('C' key) for enabling/disabling visualization

Initial investigation suggested configuration issues, but thorough analysis revealed the configuration loading path works correctly.

## Current State

**Working Components:**
- ✅ Configuration properly defined in BaseTask.yaml with correct inheritance to BlindGrasping
- ✅ ViewerController correctly accesses config via `parent.task_cfg.get("contactVisualization", {})`
- ✅ Contact body names (`r_f_link*_4`) exist in MJCF and resolve to valid indices
- ✅ Keyboard shortcut 'C' registered and toggle logging implemented
- ✅ Contact visualization rendering pipeline implemented

**Root Cause Identified:**

**Architecture Issue**: ViewerController accesses parent's `contact_forces` tensor via `self.parent.contact_forces`, but this tensor is only refreshed during the main simulation step through `physics_manager.step_physics(refresh_tensors=True)`.

In `ViewerController.render()`, the code calls `gym.refresh_net_contact_force_tensor(self.sim)` to refresh Isaac Gym's tensor, but doesn't update the parent's `contact_forces` tensor through `TensorManager.refresh_tensors()`. This creates a timing mismatch where ViewerController sees stale contact force data.

**Investigation Results:**

✅ **Physics setup works correctly**: ObservationEncoder can access non-zero contact forces, confirming Isaac Gym generates proper contact data
✅ **TensorManager refresh works correctly**: The main simulation loop properly calls `refresh_tensors()`
❌ **ViewerController has stale data**: It calls Isaac Gym refresh but doesn't update parent's tensor

## Alternative Architecture Solution

**Proposed Fix**: Instead of ViewerController accessing `self.parent.contact_forces` (which requires coordinated tensor refresh timing), ViewerController should access contact forces from the already-computed `obs_dict`.

**Available Contact Data in obs_dict:**
- `contact_forces`: Raw 3D force vectors per contact body [num_envs, num_bodies * 3]
- `contact_force_magnitude`: Computed force magnitudes [num_envs, num_bodies]
- `contact_binary`: Binary contact indicators [num_envs, num_bodies]

**Architectural Benefits:**
1. **Single source of truth**: Contact forces already computed correctly in observation pipeline
2. **No timing issues**: `obs_dict` computed at right time in simulation loop
3. **Clean separation**: ViewerController becomes consumer of processed data, not raw physics tensors
4. **No coupling**: ViewerController doesn't need TensorManager knowledge
5. **Reuses working code**: ObservationEncoder already processes contact forces correctly

## Desired Outcome

Contact visualization should work reliably:
- Bodies change color (red intensity) based on contact force magnitude
- Colors update in real-time during simulation
- Toggle with 'C' key shows proper enable/disable logging
- System handles edge cases gracefully with proper error messages

## Implementation Strategy

**Recommended Fix**: Modify ViewerController to access contact forces from `obs_dict` instead of `self.parent.contact_forces`

**Implementation Steps:**
1. **Add obs_dict access**: ViewerController needs access to current observation dictionary
2. **Update contact force source**: Use `obs_dict["contact_force_magnitude"]` instead of computing `torch.norm(self.parent.contact_forces, dim=2)`
3. **Verify data format**: Ensure obs_dict contact data matches visualization expectations
4. **Clean up**: Remove unused Isaac Gym tensor refresh calls in ViewerController

**Technical Details:**
- ViewerController currently computes: `force_magnitudes = torch.norm(contact_forces, dim=2)`
- obs_dict already provides: `contact_force_magnitude` with identical computation
- Shape compatibility: Both are [num_envs, num_bodies] tensors with force magnitudes

**Testing Approach:**
- Run with BlindGrasping task and make hand contact with box
- Press 'C' to toggle contact visualization and verify logging shows correct config values
- Confirm color changes occur during contact events based on obs_dict data
- Verify performance impact is minimal (should be better since no duplicate computation)

## Constraints

- Must maintain fail-fast philosophy - prefer clear crashes over silent failures
- Respect component architecture patterns and property decorators
- Cannot modify MJCF collision exclusions without understanding full physics implications
- Must preserve existing visualization performance optimizations

## Dependencies

None - this is a self-contained fix within the graphics and physics data pipeline.

## Implementation Status - ✅ **COMPLETED** (2025-07-28)

### ✅ **COMPLETED TASKS:**
1. **Modified ViewerController.render()** - Added obs_dict parameter with fail-fast validation
2. **Updated DexHandBase integration** - Now passes self.obs_dict to viewer_controller.render()
3. **Implemented fail-fast architecture** - Removed all fallback logic per CLAUDE.md guidelines
4. **Updated method signature** - update_contact_force_colors() now expects contact_force_magnitudes tensor
5. **Fixed NameError** - Changed `contact_forces.device` to `contact_force_magnitudes.device` at line 510
6. **Fixed tensor indexing** - Corrected subset selection for contact bodies with valid indices
7. **Fixed color comparison logic** - Updated torch.allclose to torch.isclose with proper tensor dimensions
8. **Fixed dimension handling** - Corrected tensor indexing for color update operations

### ✅ **FINAL TESTING RESULTS:**
- Environment initialization completes without crashes
- Contact visualization keyboard shortcut ('C' key) properly registered
- No NameError exceptions during rendering
- System correctly handles obs_dict-based contact force data
- Contact visualization displays red color intensity on finger bodies based on contact force magnitude

**Architecture Benefits Achieved:**
- Single source of truth: obs_dict contact data
- Eliminated timing issues with stale tensor data
- Fail-fast validation prevents silent failures
- Better performance: no duplicate force magnitude computation
- Robust tensor handling for variable numbers of valid contact bodies
