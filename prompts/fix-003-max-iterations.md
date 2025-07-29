# fix-003-max-iterations.md

Fix maxIterations config override and train.py cleanup

## Context

The maxIterations configuration system has several issues that violate the fail-fast philosophy:
1. **Hardcoded defaults**: `get_config_overrides()` in train.py has brittle hardcoded checks like `if cfg.train.maxIterations != 10000`
2. **Missing shorthand alias**: No simple `maxIterations` alias (requires full `train.maxIterations`)
3. **Test mode doesn't respect maxIterations**: `python train.py train.test=true train.maxIterations=500` has no effect in test mode
4. **Defensive programming**: train.py contains hardcoded fallbacks that should be eliminated
5. **Configuration structure inconsistency**: train_headless.yaml uses wrong section name

**Note**: CLI overrides like `python train.py train.maxIterations=5000` DO work correctly for training mode. The interpolation in BaseTaskPPO.yaml works as expected.

## Current State

**Problematic code in train.py:146**:
```python
def get_config_overrides(cfg: DictConfig) -> List[str]:
    # ... other checks with hardcoded defaults ...
    if cfg.train.maxIterations != 10000:  # Default from config.yaml
        overrides.append(f"train.maxIterations={cfg.train.maxIterations}")
    # ... more hardcoded checks ...
```

**Inconsistent alias naming in cli_utils.py:48**:
```python
ALIASES = {
    "numEnvs": "env.numEnvs",
    "maxIter": "train.maxIterations",  # Should be replaced with "maxIterations"
    # Missing: "maxIterations": "train.maxIterations"
}
```

**Decision**: Expert consensus recommends standardizing on `maxIterations` for clarity and consistency with config files. The `maxIter` alias should be removed in favor of the explicit form.

**Test mode issue**:
- `python train.py train.test=true train.maxIterations=500` - maxIterations ignored in test mode
- Test mode runs indefinitely or until manual termination
- Related to feat-002-indefinite-testing.md

**Config structure inconsistency in train_headless.yaml:12-14**:
```yaml
training:  # Should be "train:"
  maxIterations: 10000
```

## Desired Outcome

1. **Remove hardcoded defaults**: Follow fail-fast philosophy - always include configuration values in reproducible commands
2. **Standardize on explicit alias**: Replace `maxIter` with `maxIterations` for clarity and consistency with config files
3. **Fix config structure**: Correct train_headless.yaml section name
4. **Clean code quality**: Remove defensive programming patterns from get_config_overrides()

**Note**: Test mode iteration control is handled separately in feat-002-indefinite-testing.md

## Constraints

- **Fail-fast philosophy**: No defensive programming with hardcoded fallbacks
- **Single source of truth**: Configuration values come from config files only
- **Reproducibility**: get_config_overrides() must generate accurate command reconstruction
- **Breaking change acceptable**: `maxIter` removal justified by clarity benefits in research context

## Implementation Notes

1. **Remove hardcoded checks**: Change from "only include if different from default" to "always include key values"
2. **Replace alias**: Change `"maxIter": "train.maxIterations"` to `"maxIterations": "train.maxIterations"` in cli_utils.py ALIASES
3. **Fix config**: Change `training:` to `train:` in train_headless.yaml
4. **Clean up function**: Apply same principle to other hardcoded checks in get_config_overrides()
5. **Breaking change**: `maxIter` will no longer work - users must use `maxIterations`

**Rationale**: Expert consensus (o3-mini + Gemini Pro) strongly favors explicit naming for research/ML contexts where clarity and reproducibility outweigh CLI brevity concerns.

## Dependencies

None - isolated configuration management fix.
