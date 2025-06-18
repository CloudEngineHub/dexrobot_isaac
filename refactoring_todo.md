# Refactoring Todo List for DexRobot Isaac

## 🔴 Critical Issues (High Priority)

### 1. Replace print() with logger ✅
- [x] `hand_initializer.py`: ~~11~~ 5 print statements converted to logger.debug/error
- [x] `observation_encoder.py`: ~~9~~ 9 print statements converted to logger
- [x] `tensor_manager.py`: ~~25+~~ 49 print statements converted to logger
- [x] `reset_manager.py`: ~~30+~~ 37 print statements cleaned up
- [x] `dex_hand_base.py`: ~~20+~~ 42 print statements converted to logger
- [ ] `test_coordinate_transforms.py`: Test output (acceptable for tests)

### 2. Fix single source of truth violations ✅
- [x] Created `constants.py` module for shared constants
- [x] Update components to import from constants.py (action_processor, observation_encoder, dex_hand_base)
- [x] DOF properties stored in 3 places - now accessed from TensorManager via parent reference
- [x] DOF names duplicated in 3 components - now single source in HandInitializer
- [x] Device stored in every component - now all components use parent reference pattern

### 3. Refactor action_processor.py process_actions() method ✅
- [x] Method was 431 lines, now ~30 lines
- [x] Extracted helper methods:
  - `_validate_inputs()` - Input validation logic
  - `_scale_base_actions()` - Base action scaling
  - `_process_base_dofs()` - Base DOF processing
  - `_process_finger_dofs()` - Finger DOF processing
  - `_apply_finger_coupling()` - Finger coupling logic
  - `_apply_default_finger_targets()` - Default target handling
  - `_apply_pd_control()` - PD control application
- [x] Reduced deep nesting and improved readability

### 4. Remove excessive DEBUG logging ✅
- [x] `reset_manager.py`: ~~30+~~ 37 prints cleaned (converted to logger or removed)
- [x] Remove logs exposing implementation details
- [x] Stop logging inside loops

### 5. Fix fail-fast violations
- [x] `action_processor.py` L654-658: ~~Remove magic number fallbacks~~ Now raises RuntimeError
- [x] `dex_hand_base.py` L507-509: ~~Remove fallback~~ Now raises RuntimeError
- [x] `observation_encoder.py` L500-501, 517-518: ~~Remove return torch.zeros()~~ Now raises RuntimeError
- [x] `reset_manager.py`: physics_manager=None now raises RuntimeError

### 6. Fix architectural violation: dex_hand_base bypasses reset_manager ✅
- [x] `dex_hand_base.reset_idx()` duplicates reset logic instead of using `reset_manager.reset_idx()`
- [x] Reset randomization settings are configured but never used
- [x] Refactor to properly delegate to reset_manager component
- [x] Remove duplicated reset logic from dex_hand_base

### 7. Make required arguments non-optional ✅
- [x] `reset_manager.reset_idx()`: physics_manager, dof_state, root_state_tensor no longer have default=None
- [x] `reset_manager.reset_all()`: Made all required parameters non-optional
- [x] `observation_encoder.__init__()`: hand_initializer and hand_asset are now required
- [x] `action_processor.__init__()`: dof_props and hand_asset are now required
- [x] `action_processor.setup()`: dof_props is now required
- [x] Fixed indentation error introduced during refactoring

## 🟡 Important Issues (Medium Priority)

### 8. Vectorize loops ✅
- [x] `action_processor.py` L578-615: `_apply_coupling_to_targets` - nested loops over 12 actions × 1-3 joints × all envs
  - Precomputed tensor mappings: `coupling_indices`, `coupling_scales`, `action_to_coupling_range`
  - Use vectorized gather/scatter operations for ~10-100x speedup
  - Removed loop-based fallback - always use vectorized implementation
- [x] `reset_manager.py` L275-328: Environment reset loop - vectorized with advanced indexing
- [x] `dex_hand_base.py` L724-731: Hand position extraction loop - single tensor operation
- [x] `observation_encoder.py` L286-320: Nested loops for DOF mapping - dict lookups instead of loops

### 9. Extract repeated scaling logic into helper methods ✅
- [x] Scaling logic consolidated with action_space_scale/bias coefficients
- [x] DOF name-to-index mapping cached in `_get_dof_name_to_idx_mapping()`
- [x] Joint specification parsing in `_parse_joint_spec()`
- [x] Target expansion logic in `_expand_to_batch()`

### 10. Set proper log levels
- [ ] Use DEBUG for detailed diagnostics
- [ ] Use INFO for high-level progress
- [ ] Use WARNING for potential issues
- [ ] Use ERROR for failures

### 11. Add real-time viewer synchronization
- [ ] Track elapsed time vs simulated time when viewer is active
- [ ] Add sleep if simulation runs faster than real-time
- [ ] Log warning if simulation runs slower than real-time
- [ ] Make real-time sync optional via config

### 11. Move magic numbers to configuration ✅
- [x] `action_processor.py`: Velocity limits now come from config
  - Velocity limits set via `_set_velocity_limits()` method
  - Must be called before `setup()` to ensure proper initialization
  - Clear error messages if limits not set
- [x] Removed DOF limit fallbacks - now raises RuntimeError per fail-fast policy

### 12. Consolidate ActionProcessor initialization ✅
- [x] Multiple setup methods consolidated into `initialize_from_config()`
  - Single atomic initialization method replaces error-prone multi-step setup
  - All setter methods made private (_set_control_mode, _set_velocity_limits, etc.)
  - Ensures correct initialization order automatically
- [x] Added `_initialized` flag to prevent double initialization

### 13. Eliminate runtime branching on control mode ✅
- [x] `_compute_position_targets` - function pointer assigned in setup
- [x] `_get_control_mode_limits` - function pointer assigned in setup
- [x] `_compute_joint_target` - function pointer assigned in setup
- [x] `_process_finger_dofs` - no runtime branching needed (policy control is fixed)
- [x] `unscale_actions` uses `_unscale_actions_fn` pointer
- [x] Added mode-specific methods for all operations

## 🟢 Nice to Have (Low Priority)

### 12. Improve variable naming for clarity
- [ ] Rename `progress_buf` to `episode_step_count` throughout codebase
- [ ] Clarify `hand_indices` naming - distinguish between actor and rigid body indices
- [ ] Add documentation explaining Isaac Gym terminology (actor = articulated/rigid body)

### 13. Add verbosity configuration
- [ ] Add config option to control log levels
- [ ] Make debug output optional
- [ ] Allow filtering by component

## 🔵 Future Architecture Improvements

### 14. Comprehensive Rule+Policy Control Framework
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

## Summary by File

- **`action_processor.py`**: ✅ Major refactoring complete (~200 lines removed), remaining: vectorization, magic numbers, initialization consolidation, runtime branching
- **`reset_manager.py`**: ✅ DEBUG logging cleaned, ✅ required params fixed, remaining: vectorization opportunities
- **`tensor_manager.py`**: ✅ Print statements converted to logger
- **`observation_encoder.py`**: ✅ Print statements converted, ✅ required params fixed, remaining: nested loops need vectorization
- **`dex_hand_base.py`**: ✅ Print statements converted, ✅ architectural violation fixed (now delegates to reset_manager), remaining: single source violations
- **`hand_initializer.py`**: ✅ Print statements converted, remaining: single source of truth issues
- **`constants.py`**: ✅ Created for single source of truth
- **`physics_manager.py`**: ✅ Single source of truth for control_dt implemented with property decorators
