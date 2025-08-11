# refactor-006-action-processing.md

Split action processing timing to align with RL rollout patterns for better clarity and coherence.

## Context

The current action processing logic is bundled together in `pre_physics_step()`, which doesn't align well with standard RL rollout patterns. The refactoring will split action processing into two phases:
- Pre-action computation in post_physics (step N-1) to prepare DOF targets for next step's observations
- Post-action processing in pre_physics (step N) to apply policy actions

This improves clarity and makes the timing more coherent with RL frameworks where observations for step N are computed in step N-1.

## Current State

**Current Flow (in `pre_physics_step()`):**
1. Compute observations excluding active_rule_targets
2. Apply pre-action rule → compute active_rule_targets
3. Add active_rule_targets to observations
4. Process policy actions (action rule + post filters + coupling)

**Current `post_physics_step()`:**
- Only processes rewards, termination, resets
- Returns already-computed observations from pre_physics_step
- Comments indicate "Observations were already computed in pre_physics_step"

## Desired Outcome

**New Flow:**

**Post-physics (step N-1):**
1. Compute observations for step N (excluding active_rule_targets)
2. Apply pre-action rule using these observations → get active_rule_targets
3. Add active_rule_targets to observations
4. Return complete observations for step N

**Pre-physics (step N):**
- Apply policy actions only (action rule + post filters + coupling)
- Skip observation computation and pre-action rule

## Constraints

- Must preserve two-stage initialization pattern
- Must respect component boundaries and single source of truth
- Must not break existing tasks or functionality
- Should maintain or improve performance

## Implementation Notes

**Key Changes Required:**
1. **StepProcessor**: Move observation computation and pre-action rule to post_physics
2. **DexHandBase**: Modify pre_physics_step to only handle post-action processing
3. **ActionProcessor**: Ensure pre-action and post-action can be called separately

**Component Modifications:**
- `StepProcessor.process_physics_step()`: Add observation computation + pre-action
- `DexHandBase.pre_physics_step()`: Remove stages 1-3, keep only stage 4
- Ensure ActionProcessor can handle split pre/post action processing

**Testing Approach:**
- Verify identical behavior before/after refactoring
- Test with existing BlindGrasping task
- Validate timing and performance

## Dependencies

None - this is a self-contained timing refactoring.
