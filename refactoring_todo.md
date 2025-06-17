# Refactoring Todo List for DexRobot Isaac

## 🔴 Critical Issues (High Priority)

### 1. Replace print() with logger ✅
- [x] `hand_initializer.py`: ~~11~~ 5 print statements converted to logger.debug/error
- [x] `observation_encoder.py`: ~~9~~ 9 print statements converted to logger
- [x] `tensor_manager.py`: ~~25+~~ 49 print statements converted to logger
- [x] `reset_manager.py`: ~~30+~~ 37 print statements cleaned up
- [x] `dex_hand_base.py`: ~~20+~~ 42 print statements converted to logger
- [ ] `test_coordinate_transforms.py`: Test output (acceptable for tests)

### 2. Fix single source of truth violations
- [x] Created `constants.py` module for shared constants
- [x] Update components to import from constants.py (action_processor, observation_encoder, dex_hand_base)
- [ ] DOF properties stored in 3 places (HandInitializer, TensorManager, ActionProcessor)
- [ ] DOF names duplicated in 3 components
- [ ] Device stored in every component

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

### 6. Fix architectural violation: dex_hand_base bypasses reset_manager
- [ ] `dex_hand_base.reset_idx()` duplicates reset logic instead of using `reset_manager.reset_idx()`
- [ ] Reset randomization settings are configured but never used
- [ ] Refactor to properly delegate to reset_manager component
- [ ] Remove duplicated reset logic from dex_hand_base

### 7. Make required arguments non-optional ✅
- [x] `reset_manager.reset_idx()`: physics_manager, dof_state, root_state_tensor no longer have default=None
- [x] `reset_manager.reset_all()`: Made all required parameters non-optional
- [x] `observation_encoder.__init__()`: hand_initializer and hand_asset are now required
- [x] `action_processor.__init__()`: dof_props and hand_asset are now required
- [x] `action_processor.setup()`: dof_props is now required
- [x] Fixed indentation error introduced during refactoring

## 🟡 Important Issues (Medium Priority)

### 8. Vectorize loops
- [ ] `action_processor.py` L578-615: `_apply_coupling_to_targets` - nested loops over 12 actions × 1-3 joints × all envs
  - Precompute tensor mappings: `coupling_indices`, `coupling_scales`, `action_to_coupling_range`
  - Use vectorized gather/scatter operations for ~10-100x speedup
- [ ] `reset_manager.py` L275-328: Environment reset loop
- [ ] `dex_hand_base.py` L724-731: Hand position extraction loop
- [ ] `observation_encoder.py` L286-320: Nested loops for DOF mapping

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

### 11. Move magic numbers to configuration
- [ ] `action_processor.py`: Velocity limits should come from config
  - `policy_finger_velocity_limit = 2.0` (rad/s)
  - `policy_base_lin_velocity_limit = 1.0` (m/s)
  - `policy_base_ang_velocity_limit = 1.5` (rad/s)
- [ ] Remove DOF limit fallbacks (-1.0, 1.0) - violates fail-fast policy

### 12. Consolidate ActionProcessor initialization
- [ ] Multiple setup methods need consolidation (with caveat from guide_component_initialization.md)
  - Current: `__init__` → `set_control_mode` → `setup` → `set_control_options` → `set_default_targets` → `set_velocity_limits`
  - Proposed: Single `initialize(config)` method that validates and sets up everything
  - Note: Some initialization order constraints exist due to component dependencies
- [ ] Add `_initialized` flag to prevent double initialization

### 13. Eliminate runtime branching on control mode
- [ ] `_compute_position_targets` - assign function pointer in setup
- [ ] `_get_control_mode_limits` - assign function pointer in setup
- [ ] `_compute_joint_target` - assign function pointer in setup
- [ ] `_process_finger_dofs` - refactor inline branching
- [ ] Already done: `unscale_actions` uses `_unscale_actions_fn` pointer

## 🟢 Nice to Have (Low Priority)

### 14. Add verbosity configuration
- [ ] Add config option to control log levels
- [ ] Make debug output optional
- [ ] Allow filtering by component

## Summary by File

- **`action_processor.py`**: ✅ Major refactoring complete (~200 lines removed), remaining: vectorization, magic numbers, initialization consolidation, runtime branching
- **`reset_manager.py`**: ✅ DEBUG logging cleaned, ✅ required params fixed, remaining: vectorization opportunities
- **`tensor_manager.py`**: ✅ Print statements converted to logger
- **`observation_encoder.py`**: ✅ Print statements converted, ✅ required params fixed, remaining: nested loops need vectorization
- **`dex_hand_base.py`**: ✅ Print statements converted, remaining: architectural violation (bypasses reset_manager), single source violations
- **`hand_initializer.py`**: ✅ Print statements converted, remaining: single source of truth issues
- **`constants.py`**: ✅ Created for single source of truth
- **`physics_manager.py`**: ✅ Single source of truth for control_dt implemented with property decorators
