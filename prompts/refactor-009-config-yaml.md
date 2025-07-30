# Configuration Architecture Cleanup

## Problem Statement

`config.yaml` is serving two different purposes and becoming unwieldy:

1. **Primary purpose**: Training pipeline configuration for `train.py`
2. **Secondary purpose**: Test script configuration for `examples/dexhand_test.py`

The test script settings (lines 16-23: `steps`, `sleep`, `debug`, `log_level`, `enablePlotting`, `plotEnvIdx`) are mixed into the main configuration, creating separation of concerns issues.

Additionally, `debug.yaml` has naming inconsistency, using `training:` section instead of `train:` which conflicts with the main config structure.

## Configuration Analysis

**Three distinct "test" concepts identified:**

1. **Test Script** (`examples/dexhand_test.py`): Environment functional testing
   - Uses `test_render.yaml` currently
   - Settings: `steps`, `sleep`, `debug`, `log_level`, `enablePlotting`, `plotEnvIdx`
   - Purpose: Validate environment implementation

2. **Policy Testing** (`base/test.yaml`): Evaluation of trained RL policies
   - Settings: `env.numEnvs`, `train.test`, `train.maxIterations`
   - Purpose: Evaluate trained policies

3. **Policy Testing with Viewer** (`test_render.yaml`): Policy evaluation with visualization
   - Inherits from `base/test.yaml` + enables `env.viewer: true`
   - Purpose: Visual policy evaluation

**Naming Issue**: `test_render.yaml` uses deprecated "render" terminology. Per refactor-004-render.md, "viewer" is the correct semantic term.

## Solution

### 1. Create Dedicated Test Script Configuration
Create `test_script.yaml` in `dexhand_env/cfg/` with:
- Inherits from main `config.yaml` via Hydra defaults
- Contains only test script specific settings: `steps`, `sleep`, `debug`, `log_level`, `enablePlotting`, `plotEnvIdx`
- Removes clutter from main training configuration

### 2. Clean Main Configuration
Remove test script settings from `config.yaml` (lines 16-23) to focus on training pipeline.

### 3. Clean Base Test Configuration
Remove duplicate test script settings from `base/test.yaml` (lines 5-13), keep only policy evaluation settings.

### 4. Update Test Script
Modify `examples/dexhand_test.py` to use dedicated `test_script.yaml` configuration.

### 5. Fix Debug Configuration
Correct naming inconsistency in `debug.yaml` from `training:` to `train:`.

### 6. Rename for Semantic Clarity
Rename `test_render.yaml` → `test_viewer.yaml` to follow new naming conventions from refactor-004-render.md.

### 7. Update Documentation References
Update all documentation files that reference `test_render.yaml` to use `test_viewer.yaml`:
- Search codebase for `test_render` references in documentation
- Update CLI examples, usage instructions, and configuration guides
- Ensure consistency with new naming conventions

## Expected Outcome

- Clear separation of concerns between three test types
- `config.yaml` focused purely on training pipeline
- `test_script.yaml` focused on environment functional testing
- `test_viewer.yaml` focused on policy evaluation with visualization
- `base/test.yaml` focused on common policy evaluation settings
- Consistent naming conventions throughout
- All documentation updated to reflect new file names
- No functionality changes
