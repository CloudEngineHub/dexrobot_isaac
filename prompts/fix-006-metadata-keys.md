# fix-006-metadata-keys.md

✅ **COMPLETED** - Git metadata saving fails due to hardcoded config key assumptions.

## Context

The training script's git metadata saving functionality tries to reconstruct CLI arguments by hardcoding expected config keys like `env.render`. This violates fail-fast principles and breaks when configuration keys change (as happened in refactor-004-render.md where `render` became `viewer`).

Current error:
```
WARNING | Could not save git metadata: Key 'render' is not in struct
    full_key: env.render
    object_type=dict
```

## Current State

The `get_config_overrides()` function in train.py attempts to reconstruct CLI arguments for reproducibility by:
1. Hardcoding assumed "important" config keys
2. Building a synthetic "Hydra equivalent" command
3. Failing when keys don't exist (violating fail-fast)

This approach has fundamental flaws:
- **Information Loss**: Only captures subset of assumed important values
- **Hardcoded Assumptions**: Breaks when config structure changes
- **Incomplete Reconstruction**: Cannot fully reproduce complex configuration hierarchies
- **Defensive Programming**: Uses existence checks that mask configuration issues

## Desired Outcome

Replace flawed reconstruction approach with comprehensive config saving:

1. **Remove hardcoded key assumptions** - eliminate `get_config_overrides()` function entirely
2. **Save complete resolved config** - serialize the full OmegaConf configuration as YAML
3. **Preserve existing working functionality** - keep original CLI args and git info unchanged
4. **Full reproducibility** - anyone can recreate exact training conditions

## Constraints

- **Fail-fast compliance**: No defensive programming or hardcoded key checks
- **Single source of truth**: Config values come from resolved configuration only
- **Architectural alignment**: Follows recent configuration refactoring principles
- **Backward compatibility**: Don't break existing experiment tracking

## Implementation Notes

**Files to modify:**
- `train.py`: Remove `get_config_overrides()`, save complete config instead

**Testing approach:**
- Verify no warnings during metadata saving
- Confirm complete config is saved in readable format
- Test with BaseTask and BlindGrasping configurations

**Config serialization considerations:**
- Use `OmegaConf.to_yaml()` for human-readable format
- Handle any non-serializable objects gracefully
- Save to dedicated file for easy inspection
