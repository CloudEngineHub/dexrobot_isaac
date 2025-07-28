# Fix-004: DexHand Test Script Scope and Functionality Issues

## Problems Identified

### 1. Task Scope Confusion
**Current Issue**: Test script attempts to support multiple tasks (BaseTask, BoxGrasping) but this creates compatibility problems
**Root Cause**: Different tasks may implement their own action parsing logic incompatible with BaseTask assumptions
**Expected Behavior**: dexhand_test.py should focus exclusively on BaseTask testing

### 2. Position Control Mode Broken
**Error Encountered**:
```
AttributeError: 'ActionProcessor' object has no attribute '_scale_actions_to_limits'
```
**Command**: `python examples/dexhand_test.py device=cpu task.controlMode=position`
**Root Cause**: Test script's position mode action rule uses internal ActionProcessor method that doesn't exist
**Impact**: Position control mode completely non-functional

### 3. Unclear Script Purpose and Scope
**Current State**: Script tries to be a general-purpose testing tool for any task
**Issues**:
- Overly complex logic trying to handle multiple task types
- Action parsing assumes BaseTask behavior patterns
- Unclear what the script is actually meant to test
**Expected Scope**: Focus on core DexHand environment functionality testing

### 4. Verbose Logging Issues
**Problem**: Excessive debug output makes it hard to see important information
**Examples**:
- Isaac Gym visual geometry warnings
- Detailed action logging every few steps
- Unnecessary environment state information
**Impact**: Poor user experience, hard to debug actual issues

### 5. Script Location and Naming Concerns
**Current**: `examples/dexhand_test.py`
**Question**: Is this the right path/name for a basic environment testing script?
**Consideration**: Script is more about testing basic environment functionality without RL training infrastructure

## Recommended Solutions

### 1. **Restrict Task Scope to BaseTask Only**
- Remove task switching logic entirely
- Hardcode BaseTask usage with clear documentation
- Focus on testing core hand control and observation systems
- Remove BoxGrasping compatibility attempts

### 2. **Fix Position Control Mode**
- Investigate actual ActionProcessor API for action scaling
- Replace `_scale_actions_to_limits()` with correct public method
- Test both position and position_delta modes thoroughly
- Ensure action rules use only public ActionProcessor interface

### 3. **Define Clear Script Purpose**
**Primary Goals**:
- Test position and position_delta control modes
- Verify different policyControls configurations (base/fingers)
- Validate observation system functionality
- Enable real-time plotting with Rerun for debugging
- Provide contact force testing with simple box obstacles

**Explicitly NOT Goals**:
- Support arbitrary task types
- Replace RL training workflows
- Provide production-ready testing infrastructure

### 4. **Clean Up Logging**
- Reduce Isaac Gym visual geometry error spam
- Make action logging less verbose (every 50 steps instead of 25)
- Add log level controls for different information types
- Focus on actionable information for users

### 5. **Consider Script Renaming/Relocation**
**Options**:
- Keep current path but clarify purpose in documentation
- Rename to `basic_env_test.py` or `hand_control_test.py`
- Move to `tools/` directory if it's more of a development tool
- Add clear docstring explaining scope and limitations

## Implementation Priority

### High Priority
1. **Fix position control mode** - Core functionality broken
2. **Restrict to BaseTask only** - Remove confusing multi-task logic
3. **Clean up verbose logging** - Improve user experience

### Medium Priority
4. **Define and document clear scope** - Prevent future scope creep
5. **Consider renaming/relocation** - Clarify script purpose

### Low Priority
6. **Optimize test patterns** - Improve test coverage and efficiency

## Success Criteria

- [ ] Position control mode works without errors
- [ ] Script only supports BaseTask (no task switching)
- [ ] Logging is clean and focused on useful information
- [ ] Script purpose is clearly documented
- [ ] Both control modes (position/position_delta) function correctly
- [ ] Action rule implementation uses only public ActionProcessor APIs

## Implementation Notes

- This builds on the foundation established in fix-002-consistency.md
- Focus on simplification rather than feature addition
- Maintain the good work done on Hydra configuration consistency
- Keep the valuable Rerun plotting and contact testing functionality
