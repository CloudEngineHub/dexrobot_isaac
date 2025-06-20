# Development Roadmap for DexRobot Isaac

## 🟡 Current Issues

(Currently no pending issues)

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
