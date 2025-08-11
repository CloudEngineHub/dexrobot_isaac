# fix-005-box-bounce-physics.md

Fix box bouncing at initialization in BlindGrasping task

## Issue Description

After completing refactor-005-default-values, the BlindGrasping task exhibits a consistent physics behavior change where the box bounces slightly during initialization. This did not occur before the refactor.

## Symptoms

- **Consistent behavior**: Box bounces every time BlindGrasping environment initializes
- **Timing**: Occurs during environment startup/initialization phase
- **Task-specific**: Affects BlindGrasping task (BaseTask has no box to compare)
- **Physics-related**: Appears to be actual physics simulation bounce, not visual glitch

## Investigation Context

The refactor-005-default-values changes removed hardcoded defaults from `.get()` patterns throughout the codebase. While most changes should be functionally equivalent (replacing hardcoded defaults with explicit config values), one of the changes may have introduced a subtle difference affecting box initialization physics.

## Potential Cause Areas

Based on the refactor changes, the most likely causes are:

### 1. Physics Simulation Parameters
- **VecTask substeps change**: Original default was 2, config.yaml has 4
- **Physics dt**: Now uses explicit config value instead of fallback
- **Client threads**: Now uses explicit config value

### 2. Box Positioning Logic
- Box spawn height calculation
- Reference frame changes
- Table/ground plane positioning

### 3. Initialization Timing
- Order of physics parameter application
- Tensor initialization sequence
- Reset logic timing

## Current Status

- **Root cause**: Not yet identified
- **Workaround**: None needed (cosmetic issue)
- **Priority**: Medium (affects realism but not functionality)

## Investigation Steps

1. Compare physics parameters before/after refactor
2. Check box initial position calculation
3. Verify ground plane/table reference positioning
4. Test with different substeps values to isolate physics parameter effects
5. Check initialization order and timing

## Success Criteria

- Box initializes without bouncing
- Physics behavior matches pre-refactor baseline
- No regression in other physics aspects

## Configuration Context

**Box Configuration (BlindGrasping.yaml):**
```yaml
env:
  box:
    size: 0.05  # 5cm cube
    mass: 0.1   # 100g
    initial_position:
      z: 0.025  # Should place box on table surface (half-height)
```

**Ground plane**: z = 0 (unchanged)
**Expected result**: Box should rest stable on table without bouncing

## Notes

- Issue is reproducible but cosmetic (doesn't break functionality)
- Physics behavior appears more realistic (bouncing may be correct physics)
- Original behavior may have been artificially stable due to physics parameter differences
