# DexRobot Isaac Development Roadmap

Task tracking and project status for the DexRobot Isaac development. See @CLAUDE.md for the complete 3-phase development workflow process and architectural guidelines.

## Active Sprint

Sprint focus: **Architecture Consistency & Core Fixes**
With critical RL penetration issue resolved, focusing on architectural improvements and core system stability.

## Backlog

### High Priority Tasks

#### Core Architecture Fixes (`refactor_*`)
Code quality improvements and architectural enhancements.
- [ ] `refactor-001-episode-length.md` - Determine optimal config location for `episode_length` (env vs task)

### Medium Priority Tasks

#### Bug Fixes (`fix_*`)
Issue resolution and bug fixes.
- [ ] `fix-000-tb-metrics.md` - Fix tensorboard curve display in long experiment runs
- [ ] `fix-001-contact-viz.md` - Fix contact visualization config and rendering
- [ ] `fix-002-consistency.md` - Fix consistency issues
- [ ] `fix-003-max-iterations.md` - Fix maxIterations config override and train.py cleanup

#### Code Quality (`refactor_*`)
- [ ] `refactor-005-default-values.md` - Move hardcoded defaults to config files
- [ ] `refactor-004-render.md` - Clarify render option semantics (viewer vs background rendering)
- [ ] `refactor-003-imports.md` - Clean up mid-file imports for opencv and flask
- [ ] `refactor-007-blind-grasping.md` - Rename BoxGrasping to BlindGrasping task
- [ ] `refactor-007-step-architecture.md` - Investigate step processing architecture consistency
- [ ] `refactor-008-config-key-casing.md` - Unify config key naming to lower_case under task section

### Low Priority Tasks

#### Feature Enhancements (`feat_*`)
New functionality and API enhancements.
- [ ] `feat-000-streaming-port.md` - Improve port management and binding options
- [ ] `feat-001-video-fps-control.md` - Implement FPS-aware video saving
- [ ] `feat-002-indefinite-testing.md` - Enable indefinite testing mode
- [ ] `feat-004-action-rule-example.md` - Action rule example implementation

#### Documentation Tasks (`doc_*`)
Documentation improvements and illustrations.
- [ ] `doc-000-cp.md` - Documentation task
- [ ] `doc-001-video.md` - Video documentation
- [ ] `doc-002-control-dt-illustration.md` - Control dt illustration documentation
- [ ] `doc-003-action-processing-illustration.md` - Action processing illustration documentation

#### Completed Meta Tasks (`meta_*`)
Project organization, tooling, and workflow improvements.
- [x] `meta-000-workflow-setup.md` - ✅ **COMPLETED** - AI development workflow design and implementation
- [x] `meta-001-programming-guideline.md` - ✅ **COMPLETED** - Consolidate programming guidelines to user memory

## Completed Tasks

### Recently Completed
- ✅ **refactor-002-graphics-manager-in-parent.md** (2025-07-25) - **MAJOR** - Graphics component architecture alignment
  - Refactored VideoManager and ViewerController to follow established component architecture pattern
  - Removed direct sibling dependencies from constructors (graphics_manager, gym, sim, env_handles)
  - Added property decorators for accessing dependencies via parent references
  - Updated DexHandBase instantiation calls to pass only parent references
  - Achieved architectural consistency with ActionProcessor, RewardCalculator, and other components
  - Maintained exact functionality while improving fail-fast behavior and maintainability
- ✅ **refactor-006-action-processing.md** (2025-07-25) - **MAJOR** - Action processing timing refactoring
  - Split action processing into pre-action (post_physics) and post-action (pre_physics) phases
  - Moved observation computation from DexHandBase.pre_physics_step to StepProcessor.process_physics_step
  - Aligned with RL rollout patterns where observations for step N are computed in step N-1
  - Improved clarity by separating pre-action computation from post-action processing
  - Maintains identical behavior while improving timing coherence and architectural consistency
- ✅ **rl-000-penetration.md** (2025-07-25) - **CRITICAL** - Penetration prevention system implementation
  - Implemented continuous geometric penetration detection using fingertip-to-box-center distance
  - Added penetration penalty proportional to penetration depth for smooth gradients
  - Modified s2_fingerpad_proximity reward to prevent exploitation via distance clamping
  - Added configurable penetrationPrevention parameters to BoxGrasping.yaml
  - Prevents RL policies from exploiting physics simulation artifacts for realistic training
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

### Phase 1: Core Architecture (Current Priority)
1. **refactor-001-episode-length.md** - Config structure cleanup (highest priority)

### Phase 2: System Stability (Short-term)
3. **fix-000-tb-metrics.md** - Essential debugging capability
4. **fix-001-contact-viz.md** - Contact visualization fixes
5. **fix-002-consistency.md** - Fix consistency issues
6. **fix-003-max-iterations.md** - Config override fixes and train.py cleanup
7. **refactor-005-default-values.md** - Move hardcoded defaults to config
8. **refactor-008-config-key-casing.md** - Unify config key naming conventions

### Phase 3: Polish & Enhancement (Medium-term)
9. **refactor-004-render.md** - Render option semantics clarification
10. **refactor-003-imports.md** - Clean up mid-file imports
11. **refactor-007-blind-grasping.md** - Rename BoxGrasping to BlindGrasping task
12. **refactor-007-step-architecture.md** - Investigate step processing architecture consistency
13. **feat-***: Feature enhancements (streaming, video, testing modes)
14. **doc-***: Documentation improvements and illustrations

### Task Complexity Assessment
- **High complexity**: refactor-007-step-architecture (architecture investigation)
- **Medium complexity**: fix-000 (tensorboard metrics), refactor-001 (episode-length config structure)
- **Low complexity**: Most feat-* tasks, config cleanups, import organization, doc-* tasks

---

*Task details and implementation guidance available in @prompts/ directory. Development process documented in @CLAUDE.md.*
