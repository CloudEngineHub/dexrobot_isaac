# [PREFIX]-[NUMBER]-[SHORT-DESCRIPTION].md

Brief one-line description of the task.

## Context
<!-- Optional: Expand if needed during Ultrathink phase -->
<!-- Provide background information, motivation, and scope -->
<!-- For RL tasks: Include physics/policy implications -->
<!-- For refactor tasks: Explain architectural concerns -->

## Current State
<!-- Optional: Describe what exists now -->

## Desired Outcome
<!-- Optional: Describe the target state -->

## Constraints
<!-- Optional: List architectural/design constraints -->
<!-- Reference component boundaries, two-stage init, fail-fast principles -->

## Implementation Notes
<!-- Optional: Technical considerations -->
<!-- Component modifications, testing approach, etc. -->

## Dependencies
<!-- Optional: Other tasks or components this depends on -->

---

## Instructions

### File Naming Convention
Use format: `[prefix]-[number]-[short-description].md`

**Prefixes:**
- `meta_` - Workflow, tooling, project organization
- `refactor_` - Code quality, architectural improvements
- `feat_` - New functionality, API enhancements
- `fix_` - Bug fixes, issue resolution
- `rl_` - Research tasks (policy tuning, physics, rewards)

**Examples:**
- `refactor-001-episode-length.md`
- `rl-002-reward-tuning.md`
- `feat-003-new-observation.md`

### Content Guidelines

**Minimal Format (for simple tasks):**
```markdown
# refactor-001-example-task.md
Brief description of what needs to be done.
```

**Expanded Format (after Ultrathink phase):**
- Fill in relevant sections as understanding deepens
- Not all sections required for every task
- Use during Phase 1 (Ultrathink) to develop detailed understanding

### Workflow Integration

1. **Create**: Start with minimal format (brief description)
2. **Ultrathink**: Expand sections as needed during Phase 1
3. **Plan**: Reference expanded content during Phase 2 planning
4. **Complete**: Mark as done in ROADMAP.md after Phase 3
