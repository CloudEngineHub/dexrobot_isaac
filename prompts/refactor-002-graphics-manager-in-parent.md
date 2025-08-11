# refactor-002-graphics-manager-in-parent.md

Align GraphicsManager ecosystem with established component architecture patterns.

## Context

The DexRobot Isaac project follows a strict component architecture pattern where components access sibling components through parent references and property decorators, maintaining single source of truth principles. This pattern ensures:

- Clean separation of concerns
- Fail-fast behavior when dependencies are missing
- Consistent initialization order through two-stage pattern
- Reduced coupling between components

From CLAUDE.md architectural guidelines:
- Components should only take `parent` in constructor
- Use `@property` decorators to access sibling components via parent
- Never store direct references to sibling components

## Current State

**✅ GraphicsManager correctly follows the pattern:**
```python
class GraphicsManager:
    def __init__(self, parent):  # ✅ Only parent reference
        self.parent = parent

    @property
    def device(self):  # ✅ Property decorator for parent access
        return self.parent.device
```

**❌ VideoManager violates the pattern:**
```python
class VideoManager:
    def __init__(self, parent, graphics_manager):  # ❌ Direct sibling reference
        self.parent = parent
        self.graphics_manager = graphics_manager  # ❌ Stored direct reference
```

**❌ ViewerController violates the pattern:**
```python
class ViewerController:
    def __init__(self, parent, gym, sim, env_handles, headless, graphics_manager):  # ❌ Multiple direct references
        # ... stores direct references instead of using parent
```

## Desired Outcome

All graphics-related components follow the established architectural pattern:

1. **VideoManager** - Access graphics_manager via property decorator
2. **ViewerController** - Access all dependencies via parent/property decorators
3. **DexHandBase** - Update instantiation calls to only pass parent references

This creates consistent, maintainable architecture aligned with other components like ActionProcessor, RewardCalculator, etc.

## Constraints

- **Maintain exact functionality** - No behavioral changes, only architectural alignment
- **Respect two-stage initialization** - Components may need finalize_setup() if they depend on control_dt
- **Follow fail-fast philosophy** - Let dependencies crash if parent/sibling is None
- **Single source of truth** - Parent holds canonical references, components access via properties

## Implementation Notes

**VideoManager refactoring:**
- Remove `graphics_manager` parameter from constructor
- Add `@property def graphics_manager(self): return self.parent.graphics_manager`
- Update instantiation in DexHandBase

**ViewerController refactoring:**
- Remove direct dependency parameters (gym, sim, env_handles, graphics_manager)
- Add property decorators for accessing these via parent
- May need property decorators for gym, sim, env_handles if not already available on parent

**Testing approach:**
- Use existing test command: `python train.py config=test_stream render=true task=BlindGrasping device=cuda:0 checkpoint=runs/BlindGrasping_train_20250724_120120/nn/BlindGrasping.pth numEnvs=1`
- Verify video recording and viewer functionality unchanged
- Test both headless and viewer modes

## Dependencies

None - this is a pure architectural refactoring that doesn't affect external interfaces.
