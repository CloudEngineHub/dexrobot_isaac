# Development Roadmap for DexRobot Isaac

## 🟡 Current Issues (In Progress)

### 1. Contact Force Verification (Issue #4)
- **Status**: ✅ INVESTIGATION COMPLETE - System working correctly
- **Problem**: Contact forces showing as zeros despite visual contact
- **Root Cause**: Contact forces are detected correctly but on hand base/palm (rigid body indices 9,11,17,19) rather than fingertips (indices 14,20,26,32,38)
- **Key Findings**:
  - ✅ TensorManager.refresh_tensors() successfully detects contact forces up to 1211N magnitude
  - ✅ Contact force tensor and indexing system working correctly
  - ✅ Debug logging confirmed contact forces are detected at non-fingertip rigid bodies
- **Next Steps**: Verify fingertip collision geometry in MJCF model or adjust test scene for fingertip contact
- **Impact**: Contact force system is functional - just need proper contact setup

### 2. Episode Length Parameter Not Working
- **Status**: ✅ FIXED
- **Problem**: The `--episode-length` parameter in test script does not properly limit episode duration
- **Root Cause**: `reset_buf` was not cleared after resetting environments, causing continuous resets
- **Fix**: Added `self.reset_buf[env_ids] = 0` in `reset_manager.reset_idx()` following Isaac Gym standard
- **Documentation**: Created `docs/guide_environment_resets.md` explaining the reset system
- **Verified**: Episodes now properly reset at configured length without continuous resets

### 3. th_rot_target Decreases Instead of Increasing
- **Status**: ✅ FIXED
- **Problem**: With `--policy-controls-fingers=true --policy-controls-base=true`, th_rot_target was decreasing when it should increase
- **Root Cause**: Base and finger actions were tested sequentially, causing interference
- **Fix**: Implemented concurrent testing mode where base and finger actions move together
- **Changes**:
  - Added base action descriptions in test script
  - Implemented concurrent testing when both base and fingers are controlled
  - Base and finger actions now use different patterns and cycle independently
- **Verified**: Thumb rotation now increases correctly during concurrent control

### 3. Multi-Environment Testing
- **Status**: ✅ FIXED
- **Problem**: Rigid body indexing failed with multiple environments
- **Root Cause**: Global indices were used to index tensors expecting local indices
- **Fix**: Added conversion from global to local indices in ObservationEncoder
- **Changes**:
  - Modified `fingertip_indices` and `fingerpad_indices` properties
  - Convert global indices to local using modulo operation
  - Works correctly for any number of environments
- **Verified**: Successfully tested with 2 environments, no indexing errors

### 4. Actions Not Applied in Multi-Environment Setup
- **Status**: ✅ FIXED
- **Problem**: When running with multiple environments (num_envs > 1), actions were not being applied correctly when tasks added objects
- **Root Cause**: Actor creation order was inconsistent - task objects were created before hands, causing incorrect actor indexing
- **Fix**: Refactored actor creation flow to ensure consistent order:
  1. Load task assets early (before any actors)
  2. Create hands FIRST (always actor index 0)
  3. Create task objects AFTER hands (actor indices 1+)
- **Changes**:
  - Modified factory.py to remove late asset loading
  - Updated dex_hand_base.py to create actors in correct order
  - Added set_tensor_references() to task interface
  - Updated DexGraspTask to use correct actor indices
  - Created comprehensive documentation in docs/guide_actor_creation.md
- **Verified**: Actor indexing is now consistent across all environments

### 5. CPU Pipeline Multi-Environment DOF Control Issue
- **Status**: 🔴 CRITICAL BUG - Isaac Gym Issue
- **Problem**: In CPU pipeline, `gym.set_dof_position_target_tensor()` only applies to environment 0, other environments ignored
- **Symptoms**:
  - Robot 0 moves correctly, robot 1+ remain stationary despite identical targets
  - GPU pipeline works correctly - all environments move as expected
- **Test Commands**:
  - CPU (broken): `python examples/dexhand_test.py --device cpu --policy-controls-base true --num-envs 2`
  - GPU (works): `python examples/dexhand_test.py --device cuda:0 --policy-controls-base true --num-envs 2`
- **Investigation Results**:
  - ✅ Actions applied correctly to all environments in test script
  - ✅ ActionProcessor computes targets correctly for all environments
  - ✅ `gym.set_dof_position_target_tensor()` receives correct tensor with all environment targets
  - ❌ Isaac Gym CPU pipeline ignores environments 1+ (contradicts documentation)
- **Evidence**:
  - Debug logs show identical actions and targets for all environments
  - DOF positions show only env 0 moving: `Env 0: [-0.051, ...]` vs `Env 1: [0.000, ...]`
  - Same test on GPU works: both environments show movement
- **Documentation Contradiction**: Isaac Gym docs state CPU pipeline has fewer restrictions than GPU pipeline
- **Workaround**: Use GPU pipeline for multi-environment simulations
- **Status**: Reported to Isaac Gym team (potential bug in CPU pipeline tensor handling)

### 6. Camera Following Only Moves Slightly Between Robots
- **Status**: 🟡 PENDING
- **Problem**: Camera switch logs "Following robot 1" but camera barely moves
- **Possible Cause**: Hand positions might be identical for all environments (same spawn location)
- **Impact**: Debugging multi-environment setups is difficult

### 7. Contact Force Visualization Enhancement
- **Status**: 🟡 PENDING
- **Problem**: fingertip_visualizer module is obsolete and should be replaced
- **Solution**: Add contact force visualization directly to ViewerController
- **Implementation**:
  - Add option to highlight bodies when contact force > epsilon
  - Use configurable contact force bodies from BaseTask.yaml
  - Implement visual feedback (e.g., color change) for active contacts
  - Remove dependency on separate fingertip_visualizer module
- **Benefits**:
  - Cleaner architecture with visualization in ViewerController
  - Real-time visual debugging of contact forces
  - Configurable for any body, not just fingertips

## 🔵 Future Architecture Improvements

### Comprehensive Rule+Policy Control Framework
A more flexible control architecture that supports pre-action rules, policy actions, and post-action safety filters.

#### Proposed Architecture:
```python
class AdvancedActionProcessor:
    def process_actions_with_rules(self, policy_actions, state):
        # 1. Pre-action rules
        rule_targets = self.pre_action_rule(state)
        # Optionally provide rule targets as observation to policy

        # 2. Policy action application
        if self.action_mode == "overwrite":
            targets = self.apply_policy_actions(policy_actions)
        elif self.action_mode == "residual":
            targets = rule_targets + self.residual_scale * policy_actions
        elif self.action_mode == "selective":
            # Policy controls subset of DOFs
            targets = self.blend_targets(rule_targets, policy_actions)

        # 3. Post-action safety filter
        safe_targets = self.post_action_filter(targets, state)

        return safe_targets
```

#### Key Features:
1. **Pre-action Rules**:
   - Generate baseline targets for all DOFs
   - Can provide rule-based targets as observation to policy
   - Examples: gravity compensation, default poses, trajectory following

2. **Action Modes**:
   - **Overwrite**: Policy completely replaces rule targets
   - **Residual**: Policy adds corrections to rule targets
   - **Selective**: Policy controls subset of DOFs, rules control others

3. **Post-action Filters**:
   - Safety limiting (velocity, acceleration, workspace)
   - Collision avoidance
   - Joint limit enforcement

#### Benefits:
- More flexible than current binary policy/rule control
- Enables hierarchical control strategies
- Safety guarantees through post-processing
- Easier to implement complex behaviors

#### Implementation Notes:
- Similar patterns found in reference tasks (action scaling, transformations)
- Would extend rather than replace current ActionProcessor
- Consider using hooks/callbacks for extensibility

## 🟢 Future Development

### Phase 2 - Architecture & Quality
- [ ] Standardize naming (dex_hand vs dexhand) throughout codebase
- [ ] Update README files with current architecture documentation
- [ ] Document the fail-fast philosophy with concrete examples from this project

### Phase 3 - RL Integration
- [ ] Test with PPO/SAC algorithms
- [ ] Implement modular reward system (reaching, grasping, manipulation components)
- [ ] Add reward visualization in rerun for debugging
- [ ] Create training scripts with DexHand-specific hyperparameters

### Phase 3 - Advanced Features
- [ ] ROS2 interface for real robot deployment
- [ ] Record/playback functionality for demonstration learning
- [ ] Add manipulation task environments (pick-and-place, in-hand rotation, tool use)
- [ ] Implement contact-rich manipulation primitives
- [ ] Support bimanual environment with two dexterous hands
  - Extend HandInitializer to support multiple hands per environment
  - Update observation/action spaces for dual hand control
  - Add inter-hand coordination capabilities
  - Create bimanual manipulation tasks (e.g., two-handed grasping, assembly)

## 📋 Implementation Guidelines

### Priority Order
1. Fix bugs (th_rot_target issue)
2. Complete architecture documentation
3. Implement core RL features
4. Add advanced capabilities
5. Optimize performance

### Development Principles
- Maintain fail-fast behavior - no defensive programming for internal logic
- Follow scientific computing mindset with vectorization
- Keep single source of truth for all shared state
- Use proper logging levels throughout
- Test changes with `python examples/dexhand_test.py` before committing
- Write clear documentation for new features

---

## ✅ Completed Items (Archive)

### Critical Issues (Resolved)
- ✅ Replace print() with logger - All components now use proper logging
- ✅ Fix single source of truth violations - Created constants.py, parent reference pattern
- ✅ Refactor action_processor.py - Reduced from 431 to ~30 lines with helper methods
- ✅ Remove excessive DEBUG logging - Cleaned up all components
- ✅ Fix fail-fast violations - All fallbacks replaced with RuntimeError
- ✅ Fix architectural violations - Proper component delegation
- ✅ Make required arguments non-optional - All APIs updated
- ✅ Device mismatch error - Fixed torch.device comparison
- ✅ Viewer camera focusing - Fixed rigid body index architecture

### Important Issues (Resolved)
- ✅ Vectorize loops - All critical paths now use tensor operations
- ✅ Extract repeated scaling logic - Consolidated into helper methods
- ✅ Set proper log levels - Consistent throughout codebase
- ✅ Add real-time viewer synchronization - Via gym.sync_frame_time()
- ✅ Move magic numbers to configuration - Velocity limits in config
- ✅ Consolidate ActionProcessor initialization - Single initialize_from_config()
- ✅ Eliminate runtime branching - Function pointers for control modes
- ✅ --log-level flag - Working correctly, respects command-line setup
- ✅ Controller logic in task_interface.py - Examples in docs are valuable
- ✅ Performance issue - Resolved with 200Hz physics configuration
- ✅ Random action logic - Implemented in ViewerController
- ✅ Reset manager compliance - Fully vectorized, no fallbacks

### Nice to Have (Resolved)
- ✅ Improve variable naming - episode_step_count, clear actor/rigid body distinction
- ✅ Add verbosity configuration - logLevel in config, component debug flags

### Architecture Improvements
- ✅ Component-based architecture with clear separation of concerns
- ✅ Property decorators for single source of truth
- ✅ Fail-fast principles throughout
- ✅ Scientific computing approach with vectorization

## Overall Assessment: 🎯 **EXCELLENT**

The codebase has been successfully refactored to follow best practices:
- Clean component architecture with clear interfaces
- Fail-fast behavior with no defensive programming
- Efficient vectorized operations throughout
- Proper logging and configuration management
- Single source of truth for all shared state
