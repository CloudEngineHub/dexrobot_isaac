# feat-002-indefinite-testing.md

Enable indefinite testing mode

## Context

Testing mode (`python train.py train.test=true`) currently doesn't respect the `maxIterations` parameter, making it difficult to control test duration. Setting `train.maxIterations=500` has no effect in test mode - the test runs indefinitely until manual termination.

## Current Behavior

- **Training mode**: `maxIterations` controls training epochs correctly
- **Test mode**: `maxIterations` is ignored, test runs indefinitely
- **Workaround**: Manual termination (Ctrl+C) required

## Desired Outcome

1. **Finite test control**: `python train.py train.test=true maxIterations=500` should run exactly 500 test episodes/steps
2. **Indefinite mode**: Support truly indefinite testing with special value (e.g., `maxIterations=-1` or `maxIterations=infinite`)
3. **Consistency**: Same parameter should control duration in both train and test modes

## Implementation Notes

- Test mode uses different runner logic than training mode
- Need to examine how `runner.run()` handles test mode parameters
- May require separate test duration parameter if training iterations concept doesn't apply to testing

## Related Issues

- Partially addresses issue from fix-003-max-iterations.md (test mode maxIterations behavior)
- Different from examples/dexhand_test.py which uses `steps` parameter
