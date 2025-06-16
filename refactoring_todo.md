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

### 3. Refactor action_processor.py process_actions() method
- [ ] Method is 319 lines (should be <50)
- [ ] Extract helper methods for repeated patterns
- [ ] Reduce deep nesting and complexity

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

### 7. Make required arguments non-optional
- [ ] `reset_manager.reset_idx()`: physics_manager, dof_state, root_state_tensor should not have default=None
- [ ] Review all component methods for optional parameters that are actually required
- [ ] Remove =None defaults and fail at function definition time, not runtime
- [ ] Update method signatures to enforce required dependencies

## 🟡 Important Issues (Medium Priority)

### 8. Vectorize loops
- [ ] `action_processor.py` L408-460: Finger coupling loop
- [ ] `reset_manager.py` L275-328: Environment reset loop
- [ ] `dex_hand_base.py` L724-731: Hand position extraction loop
- [ ] `observation_encoder.py` L286-320: Nested loops for DOF mapping

### 9. Extract repeated scaling logic into helper methods
- [ ] DOF scaling logic (repeated 3x)
- [ ] DOF name-to-index mapping (repeated 3x)
- [ ] Joint specification parsing (repeated 3x)
- [ ] Target expansion logic (repeated 2x)

### 10. Set proper log levels
- [ ] Use DEBUG for detailed diagnostics
- [ ] Use INFO for high-level progress
- [ ] Use WARNING for potential issues
- [ ] Use ERROR for failures

## 🟢 Nice to Have (Low Priority)

### 11. Add verbosity configuration
- [ ] Add config option to control log levels
- [ ] Make debug output optional
- [ ] Allow filtering by component

## Summary by File

- **`action_processor.py`**: Long methods (319 lines), repeated code, vectorization issues
- **`reset_manager.py`**: ✅ DEBUG logging cleaned, optional params that should be required, vectorization opportunities
- **`tensor_manager.py`**: ✅ Print statements converted to logger
- **`observation_encoder.py`**: ✅ Print statements converted, nested loops need vectorization
- **`dex_hand_base.py`**: Print statements, architectural violation (bypasses reset_manager), single source violations
- **`hand_initializer.py`**: ✅ Print statements converted, single source of truth issues
- **`constants.py`**: ✅ Created for single source of truth
