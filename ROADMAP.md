# DexRobot Isaac Development Roadmap

Task tracking and project status for the DexRobot Isaac development. See @CLAUDE.md for the complete 3-phase development workflow process and architectural guidelines.

## Active Sprint

Sprint focus: **System Stability & Bug Fixes**
With major architectural improvements complete (episode length config, graphics component alignment, action processing timing), focusing on core system stability and essential debugging capabilities.

## Backlog

### High Priority Tasks

#### Core Architecture Fixes (`refactor_*`)
Code quality improvements and architectural enhancements.

### Medium Priority Tasks

#### Bug Fixes (`fix_*`)
Issue resolution and bug fixes.
- [ ] `fix-003-max-iterations.md` - Fix maxIterations config override and train.py cleanup

#### Code Quality (`refactor_*`)
- [ ] `refactor-005-default-values.md` - Move hardcoded defaults to config files
- [ ] `refactor-004-render.md` - Clarify render option semantics (viewer vs background rendering)
- [ ] `refactor-003-imports.md` - Clean up mid-file imports for opencv and flask
- [ ] `refactor-007-blind-grasping.md` - Rename BoxGrasping to BlindGrasping task
- [ ] `refactor-007-step-architecture.md` - Investigate step processing architecture consistency
- [ ] `refactor-008-config-key-casing.md` - Unify config key naming to lower_case under task section (file currently has typo: rafactor-008-config-key-casing.md)

### Low Priority Tasks

#### Research Tasks (`rl_*`)
Policy tuning, physics improvements, and reward engineering.
- [ ] `rl-001-blind-grasping-task.md` - Break down blind grasping training difficulties into specific fixes

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
- ✅ **fix-002-consistency.md** (2025-07-28) - **CRITICAL** - Test script and training consistency fixes
  - Fixed test script environment count issue by switching to existing test_render.yaml configuration (4 environments vs 1024)
  - Updated control mode validation to accept both position and position_delta modes, resolving BoxGrasping compatibility
  - Leveraged existing test configuration infrastructure instead of creating new files
  - Verified both BaseTask and BoxGrasping work properly with Hydra inheritance and CLI overrides
  - Achieved full consistency between test and train scripts using identical Hydra configuration system
  - Minimal code changes with maximum reuse of existing well-designed test configurations
- ✅ **fix-001-contact-viz.md** (2025-07-28) - **ESSENTIAL** - Contact visualization NameError fix
  - Fixed NameError in ViewerController.update_contact_force_colors() where contact_forces.device was referenced after parameter rename
  - Corrected tensor indexing to handle subset of contact bodies with valid indices
  - Fixed color comparison logic using torch.isclose instead of torch.allclose for proper tensor dimensions
  - Updated dimension handling for color update operations
  - Contact visualization now works correctly with 'C' key toggle, displaying red intensity based on contact force magnitude
  - Minimal code changes with immediate impact on debugging and visualization capabilities
- ✅ **fix-001-reward-logging-logic.md** (2025-07-28) - **CRITICAL** - RewardComponentObserver windowed statistics fix
  - Fixed RewardComponentObserver logging cumulative averages instead of windowed statistics
  - Added reset logic in _log_to_tensorboard() to clear cumulative_sums after each logging interval
  - Converts step-level logging from "average since training start" to meaningful windowed statistics
  - Enables better training insights by showing recent performance trends instead of slowly-changing overall averages
  - Minimal code change with significant impact on TensorBoard reward component trending visibility
- ✅ **fix-000-tb-metrics.md** (2025-07-28) - **ESSENTIAL** - TensorBoard data retention fix
  - Changed rewardLogInterval from 10 to 100 episodes in config.yaml
  - Added clarifying comment explaining TensorBoard 1000-point sampling limit prevention
  - Reduces logging frequency by 10x, extending visible reward breakdown history from ~780.5M to ~7.8B+ steps
  - RewardComponentObserver operations confirmed to be well-vectorized for scalability
  - Single configuration change with immediate impact on long training runs
- ✅ **refactor-001-episode-length.md** (2025-07-28) - **MAJOR** - Episode length configuration architectural fix
  - Resolved inconsistency where episodeLength was placed in different config sections (task vs env)
  - Moved episodeLength from task to env section in BaseTask.yaml to align with DexHandBase expectations
  - Maintained CLI override functionality (--episode-length) and BoxGrasping.yaml compatibility
  - Fixed potential runtime failures when BaseTask was used directly
  - Single line configuration change with comprehensive testing validation
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
With refactor-001-episode-length.md completed, focus shifts to next high priority architectural tasks.

### Phase 2: System Stability (Short-term)
1. **fix-002-consistency.md** - Fix consistency issues
2. **fix-003-max-iterations.md** - Config override fixes and train.py cleanup
3. **refactor-005-default-values.md** - Move hardcoded defaults to config
4. **refactor-008-config-key-casing.md** - Unify config key naming conventions

### Phase 3: Polish & Enhancement (Medium-term)
5. **refactor-004-render.md** - Render option semantics clarification
6. **refactor-003-imports.md** - Clean up mid-file imports
7. **refactor-007-blind-grasping.md** - Rename BoxGrasping to BlindGrasping task
8. **refactor-007-step-architecture.md** - Investigate step processing architecture consistency
9. **feat-***: Feature enhancements (streaming, video, testing modes)
10. **doc-***: Documentation improvements and illustrations

### Task Complexity Assessment
- **High complexity**: refactor-007-step-architecture (architecture investigation)
- **Medium complexity**: fix-003 (maxIterations config override), fix-002 (consistency issues)
- **Low complexity**: Most feat-* tasks, config cleanups, import organization, doc-* tasks

---

*Task details and implementation guidance available in @prompts/ directory. Development process documented in @CLAUDE.md.*
