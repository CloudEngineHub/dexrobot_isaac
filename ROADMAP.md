# DexRobot Isaac Development Roadmap

Task tracking and project status for the DexRobot Isaac development. See @CLAUDE.md for the complete 3-phase development workflow process and architectural guidelines.

## Active Sprint

Sprint focus: **Foundation & Critical Issues**
Prioritizing architectural consistency and RL stability.

## Backlog

### High Priority Tasks

#### Critical RL Issues (`rl_*`)
Machine learning research, policy tuning, physics, and reward engineering.
- [ ] `rl-000-penetration.md` - **URGENT** - Prevent policy from exploiting penetration in physics simulation

#### Core Architecture Fixes (`refactor_*`)
Code quality improvements and architectural enhancements.
- [ ] `refactor-006-action-processing.md` - **MAJOR** - Refactor action processing timing and coordination
- [ ] `refactor-002-graphics-manager-in-parent.md` - Align GraphicsManager with component architecture patterns
- [ ] `refactor-001-episode-length.md` - Determine optimal config location for `episode_length` (env vs task)

### Medium Priority Tasks

#### Bug Fixes (`fix_*`)
Issue resolution and bug fixes.
- [ ] `fix-000-tb-metrics.md` - Fix tensorboard curve display in long experiment runs
- [ ] `fix-001-contact-viz.md` - Fix contact visualization config and rendering

#### Code Quality (`refactor_*`)
- [ ] `refactor-005-default-values.md` - Move hardcoded defaults to config files
- [ ] `refactor-004-render.md` - Clarify render option semantics (viewer vs background rendering)
- [ ] `refactor-003-imports.md` - Clean up mid-file imports for opencv and flask

### Low Priority Tasks

#### Feature Enhancements (`feat_*`)
New functionality and API enhancements.
- [ ] `feat-000-streaming-port.md` - Improve port management and binding options
- [ ] `feat-001-video-fps-control.md` - Implement FPS-aware video saving
- [ ] `feat-002-indefinite-testing.md` - Enable indefinite testing mode

#### Completed Meta Tasks (`meta_*`)
Project organization, tooling, and workflow improvements.
- [x] `meta-000-workflow-setup.md` - ✅ **COMPLETED** - AI development workflow design and implementation
- [x] `meta-001-programming-guideline.md` - ✅ **COMPLETED** - Consolidate programming guidelines to user memory

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

## Strategic Development Plan

### Phase 1: Critical Foundation (Immediate)
1. **rl-000-penetration.md** - Fix physics exploitation (blocks RL training)
2. **refactor-006-action-processing.md** - Core action timing architecture

### Phase 2: Architecture Consistency (Short-term)
3. **refactor-002-graphics-manager-in-parent.md** - Component pattern alignment
4. **fix-000-tb-metrics.md** - Essential debugging capability
5. **refactor-001-episode-length.md** - Config structure cleanup

### Phase 3: Polish & Enhancement (Medium-term)
6. Remaining refactor tasks (defaults, imports, render semantics)
7. Feature enhancements (streaming, video, testing modes)

### Task Complexity Assessment
- **High complexity**: refactor-006 (action processing timing)
- **Medium complexity**: rl-000 (physics penetration), refactor-002 (graphics architecture)
- **Low complexity**: Most feat-* tasks, config cleanups

---

*Task details and implementation guidance available in @prompts/ directory. Development process documented in @CLAUDE.md.*
