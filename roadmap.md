# Development Roadmap for DexRobot Isaac

## 🟡 Current Issues

(Currently no pending issues)

## 🟢 Future Development

### Phase 2 - Architecture & Quality
- [x] Standardize naming (dex_hand vs dexhand) throughout codebase - refactored to use "dexhand"
- [x] Update README files with current architecture documentation - simplified to focus on usage
- [x] Document the fail-fast philosophy with concrete examples from this project - documented in CLAUDE.md

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
  - Pure functional approach with clear data flow
  - Registry pattern for post-action filters
  - Two-stage observation initialization to resolve circular dependencies
  - Supports overwrite, residual, and selective action modes
  - Properly handles rule-based control for uncontrolled DOFs
