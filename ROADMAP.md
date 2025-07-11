# Development Roadmap for DexRobot Isaac

## 🟡 Current Issues

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
- **Documentation**: Created `docs/guide-environment-resets.md` explaining the reset system
- **Verified**: Episodes now properly reset at configured length without continuous resets

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

## 🟢 Future Development

### Phase 2 - Architecture & Quality
- [x] Standardize naming (dex_hand vs dexhand) throughout codebase - refactored to use "dexhand"
- [x] Update README files with current architecture documentation - simplified to focus on usage
- [x] Document the fail-fast philosophy with concrete examples from this project - documented in CLAUDE.md

### Phase 3 - RL Integration
- [x] Test with PPO/SAC algorithms
  - ✅ Fixed config loading issues (duplicate sim sections)
  - ✅ Removed deprecated use_gpu_pipeline key
  - ✅ Fixed num_actions fail-fast property access
  - ✅ Verified environment creation, reset, and step work correctly
  - ✅ Basic PPO training loop tested successfully
- [x] Implement modular reward system (reaching, grasping, manipulation components)
  - ✅ BaseTaskRewards with weighted components (alive, height_safety, finger_velocity)
  - ✅ Clean separation between reward computation and termination logic
- [x] Add reward visualization/debugging tools
  - ✅ TensorBoard integration with comprehensive reward component logging
  - ✅ RewardComponentObserver tracks episode totals and per-step averages
  - ✅ Hierarchical organization by termination type and weight type
- [ ] **Box Grasping Task** (Priority: High)
  - Environment setup:
    - Single 5cm box at origin with slight position randomization
    - 10-second episode timeout (failure)
  - Observations:
    - All base observations except exact contact forces
    - Binary contact indicators (touch/no-touch) per finger instead of force magnitudes
  - Rewards:
    - Standard base rewards (alive, height_safety, finger_velocity)
    - Additional: object height reward component
  - Termination criteria:
    - Success: object height > 20cm AND ≥2 fingers in contact for ≥2 seconds
    - Failure: any fingertip/pad or hand base z ≤ 0
    - Timeout: 10 seconds
  - **Implementation Plan**:
    - **Phase 1: Base Framework Enhancements** ✅ COMPLETED
      - ✅ Add binary contact observation to ObservationEncoder
        - New observation type: "contact_binary" with configurable threshold
        - Reduces sim2real gap by abstracting force magnitudes
        - BaseTask now uses binary contacts instead of exact forces
      - ✅ Add contact duration tracking as observer state
        - Track per-body contact durations indexed by contactForceBodies
        - Vectorized duration updates with transition detection
        - Query interface: get_contact_duration_by_body(body_name)
        - Integrated reset handling for proper state management
    - **Phase 2: BoxGraspingTask Implementation**
      - Create BoxGraspingTask class implementing DexTask interface
      - Box asset loading and per-environment actor creation
      - Box state tracking and observations (pose, velocities)
      - Object height reward computation
      - Success/failure criteria using ContactDurationTracker
      - Box reset with position randomization
    - **Phase 3: Configuration and Testing**
      - Create BoxGrasping.yaml configuration
      - Set episodeLength: 2000 (10 seconds at 200Hz)
      - Configure box physics (mass: 0.1kg, friction: 1.0)
      - Tune reward weights for stable grasping behavior
      - Test success/failure detection and contact tracking
- [ ] Create training scripts with DexHand-specific hyperparameters
- [ ] Benchmark learning performance

### Phase 3 - Advanced Features
- [ ] **Domain Randomization Framework** (Phased Approach)
  - Phase 1: Enhance ResetManager with configurable randomization ranges
    - Keep simple `randomize: true/false` flag for ease of use
    - Add optional `randomization_params` for custom ranges
    - Support hand position, orientation, and DOF randomization
  - Phase 2: Full domain randomization system
    - Create RandomizationManager component
    - Support categories: observations, actions, physics, actor_properties
    - Add distribution types (uniform, gaussian, loguniform)
    - Implement scheduling (constant, linear)
    - Enable per-category and per-parameter control
- [ ] Create hierarchical configuration system
  - Support nested YAML files for modular task definitions
  - Allow per-task overrides of base configurations
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

## 🔴 Won't Fix

### CPU Pipeline Multi-Environment DOF Control Issue
- **Problem**: In CPU pipeline, `gym.set_dof_position_target_tensor()` only applies to environment 0
- **Evidence**: GPU works correctly, CPU ignores environments 1+ (contradicts Isaac Gym documentation)
- **Workaround**: Use GPU pipeline for multi-environment simulations
- **Decision**: Won't fix - Isaac Gym limitation, use GPU pipeline instead

## ✅ Completed Items

### Recent Fixes
- ✅ **Contact Force Visualization**: Replaced fingertip_visualizer with integrated ViewerController visualization
- ✅ **Camera Following Issue**: Fixed camera coordinate system handling for multi-environment setups
- ✅ **Contact Force Verification**: System working correctly - forces detected on hand base/palm
- ✅ **Episode Length Parameter**: Fixed reset buffer clearing to prevent continuous resets
- ✅ **th_rot_target Issue**: Fixed concurrent base/finger control interference
- ✅ **Multi-Environment Testing**: Fixed global-to-local rigid body index conversion
- ✅ **Multi-Environment Actions**: Fixed actor creation order for consistent indexing

### Architecture Refactoring
- ✅ Component-based architecture with clear separation of concerns
- ✅ Fail-fast principles with no defensive programming for internal logic
- ✅ Scientific computing approach with vectorization throughout
- ✅ Single source of truth via property decorators and constants
- ✅ Proper logging and configuration management
- ✅ **Functional Rule+Policy Control Framework**: Implemented comprehensive pipeline with pre-action rules, action rules, post-action filters, and coupling rules
- ✅ **TerminationManager**: Clean separation of termination decisions vs reset execution
  - Replaced SuccessFailureTracker with cleaner three-state termination model
  - ResetManager now purely handles physical reset execution
  - Pure functional approach with clear data flow
  - Registry pattern for post-action filters
  - Two-stage observation initialization to resolve circular dependencies
  - Supports overwrite, residual, and selective action modes
  - Properly handles rule-based control for uncontrolled DOFs
