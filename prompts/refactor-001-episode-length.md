# refactor-001-episode-length.md

Resolve architectural inconsistency in episodeLength configuration placement.

## Context

The DexHand codebase has an architectural inconsistency where `episodeLength` is placed in different config sections:
- BaseTask.yaml: `task.episodeLength: 300`
- BoxGrasping.yaml: `env.episodeLength: 500`
- DexHandBase code: expects `env_cfg["episodeLength"]`

This creates potential runtime failures when BaseTask is used directly, since the code looks for the parameter in the env section but BaseTask defines it in the task section.

## Current State

- **BaseTask.yaml**: Places `episodeLength` in `task` section (line 24)
- **BoxGrasping.yaml**: Places `episodeLength` in `env` section (line 15)
- **DexHandBase**: Reads from `self.env_cfg["episodeLength"]` (line 148)
- **CLI Integration**: `dexhand_test.py` overrides `cfg["env"]["episodeLength"]`

## Desired Outcome

Consistent placement of `episodeLength` parameter across all config files and code, eliminating the architectural inconsistency.

## Constraints

- Must maintain backward compatibility with existing BoxGrasping.yaml
- Must align with existing code expectations in DexHandBase
- Must preserve CLI override functionality
- Should follow architectural principles for config section organization

## Implementation Notes

**Expert Consensus Results:**
- Multiple AI models unanimously agreed `episodeLength` belongs in `env` section
- Parameter is classified as runtime execution constraint, similar to `numEnvs`, `device`
- Simple fix: move one line from task to env section in BaseTask.yaml

**Key Insight from User Challenge:**
`episodeLength` has task-semantic properties (affects task difficulty/feasibility) but is architecturally an environment runtime constraint (timeout mechanism). Current code expects env section placement.

**Technical Approach:**
1. Move `episodeLength: 300` from `task` to `env` section in BaseTask.yaml
2. Test BaseTask environment creation to verify fix
3. No code changes required (DexHandBase already expects env section)

## Dependencies

None - this is a standalone configuration fix.
