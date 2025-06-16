# Refactoring Todo List for DexRobot Isaac

## 🔴 Critical Issues (High Priority)

### 1. Replace print() with logger ✅
- [x] `hand_initializer.py`: ~~11~~ 5 print statements converted to logger.debug/error
- [x] `observation_encoder.py`: ~~9~~ 9 print statements converted to logger
- [x] `tensor_manager.py`: ~~25+~~ 49 print statements converted to logger
- [ ] `reset_manager.py`: 30+ DEBUG print statements
- [ ] `dex_hand_base.py`: 20+ print statements
- [ ] `test_coordinate_transforms.py`: Test output (acceptable for tests)

### 2. Fix single source of truth violations
- [x] Created `constants.py` module for shared constants
- [ ] Update components to import from constants.py
- [ ] DOF properties stored in 3 places (HandInitializer, TensorManager, ActionProcessor)
- [ ] DOF names duplicated in 3 components
- [ ] Device stored in every component

### 3. Refactor action_processor.py process_actions() method
- [ ] Method is 319 lines (should be <50)
- [ ] Extract helper methods for repeated patterns
- [ ] Reduce deep nesting and complexity

### 4. Remove excessive DEBUG logging
- [ ] `reset_manager.py`: 30+ prints in loops (will spam with 100s of envs)
- [ ] Remove logs exposing implementation details
- [ ] Stop logging inside loops

### 5. Fix fail-fast violations
- [x] `action_processor.py` L654-658: ~~Remove magic number fallbacks~~ Now raises RuntimeError
- [x] `dex_hand_base.py` L507-509: ~~Remove fallback~~ Now raises RuntimeError
- [x] `observation_encoder.py` L500-501, 517-518: ~~Remove return torch.zeros()~~ Now raises RuntimeError

## 🟡 Important Issues (Medium Priority)

### 6. Vectorize loops
- [ ] `action_processor.py` L408-460: Finger coupling loop
- [ ] `reset_manager.py` L275-328: Environment reset loop
- [ ] `dex_hand_base.py` L724-731: Hand position extraction loop
- [ ] `observation_encoder.py` L286-320: Nested loops for DOF mapping

### 7. Create central constants.py module
- [ ] Define NUM_BASE_DOFS, NUM_ACTIVE_FINGER_DOFS once
- [ ] Centralize joint name lists
- [ ] Centralize body name lists

### 8. Extract repeated scaling logic into helper methods
- [ ] DOF scaling logic (repeated 3x)
- [ ] DOF name-to-index mapping (repeated 3x)
- [ ] Joint specification parsing (repeated 3x)
- [ ] Target expansion logic (repeated 2x)

### 9. Set proper log levels
- [ ] Use DEBUG for detailed diagnostics
- [ ] Use INFO for high-level progress
- [ ] Use WARNING for potential issues
- [ ] Use ERROR for failures

## 🟢 Nice to Have (Low Priority)

### 10. Add verbosity configuration
- [ ] Add config option to control log levels
- [ ] Make debug output optional
- [ ] Allow filtering by component

## Summary by File

- **`action_processor.py`**: Long methods, fail-fast violations, repeated code, vectorization issues
- **`reset_manager.py`**: Excessive DEBUG logging, vectorization opportunities
- **`tensor_manager.py`**: Too many low-level print statements
- **`observation_encoder.py`**: Print statements, fail-fast violations, nested loops
- **`dex_hand_base.py`**: Print statements, fail-fast violations, single source violations
- **`hand_initializer.py`**: Print statements, single source of truth issues
