# DexRobot Isaac Development Roadmap

Task tracking and project status for the DexRobot Isaac development. See @CLAUDE.md for the complete 3-phase development workflow process and architectural guidelines.

## Active Sprint

Workflow implementation completed. Ready for next development cycle.

## Backlog

### Meta Tasks (`meta_*`)
Project organization, tooling, and workflow improvements.

- [x] `meta-000-workflow-setup.md` - ✅ **COMPLETED** - AI development workflow design and implementation
- [x] `meta-001-programming-guideline.md` - ✅ **COMPLETED** - Consolidate programming guidelines to user memory

### Refactor Tasks (`refactor_*`)
Code quality improvements and architectural enhancements.

- [ ] `refactor-001-episode-length.md` - Determine optimal config location for `episode_length` (env vs task)

### RL Research Tasks (`rl_*`)
Machine learning research, policy tuning, physics, and reward engineering.

- [ ] `rl-000-penetration.md` - Prevent policy from exploiting penetration in physics simulation

### Feature Tasks (`feat_*`)
New functionality and API enhancements.

*No active feature requests*

### Bug Fixes (`fix_*`)
Issue resolution and bug fixes.

*No active bug reports*

## Completed Tasks

### Recently Completed
- ✅ **meta-000-workflow-setup.md** (2025-01-25) - AI development workflow implementation
  - Created structured 3-phase workflow process
  - Established task organization system
  - Implemented progress tracking with @ROADMAP.md and @CLAUDE.md
- ✅ **meta-001-programming-guideline.md** (2025-01-25) - Programming guidelines consolidation
  - Extracted universal fail-fast philosophy for research code
  - Added scientific computing mindset guidelines
  - Implemented issue resolution protocol
  - Added study-first development principles to global CLAUDE.md

## Next Priorities

1. **refactor-001-episode-length.md** - Resolve config structure question
2. **rl-000-penetration.md** - Address RL policy exploitation issue

---

*Task details and implementation guidance available in @prompts/ directory. Development process documented in @CLAUDE.md.*
