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
- **Status**: 🟡 PENDING
- **Problem**: The `--episode-length` parameter in test script does not properly limit episode duration
- **Expected**: Episodes should reset after specified number of steps
- **Location**: Likely in episode termination logic or reset handling
- **Impact**: Test scripts run longer than intended, making debugging difficult
- **Steps to reproduce**: Run `python examples/dexhand_test.py --episode-length 10` and observe it runs beyond 10 steps

### 3. th_rot_target Decreases Instead of Increasing
- **Status**: 🟡 PENDING
- **Problem**: With `--policy-controls-fingers=true`, th_rot_target decreases from step 20 to 25 in test script
- **Expected**: Should increase linearly during this period
- **Location**: Likely in action_processor.py finger coupling logic or test script action generation
- **Impact**: Incorrect thumb rotation control
- **Steps to reproduce**: Run `python examples/dexhand_test.py --policy-controls-fingers=true` and observe th_rot_target values

### 4. Multi-Environment Testing
- **Status**: 🟡 PENDING
- **Problem**: Need to verify system works correctly with num_envs > 1
- **Tasks**:
  - Test with various num_envs (2, 4, 16, 64)
  - Verify reset_idx works correctly with different environment indices
  - Check performance scaling with multiple environments
  - Ensure observations/actions are properly batched

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
