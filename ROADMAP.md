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

#### Code Quality (`refactor_*`)
- [ ] `refactor-007-blind-grasping.md` - Rename BoxGrasping to BlindGrasping task
- [ ] `refactor-007-step-architecture.md` - Investigate step processing architecture consistency
- [ ] `refactor-008-config-key-casing.md` - Unify config key naming to lower_case under task section (file currently has typo: rafactor-008-config-key-casing.md)
- [ ] `refactor-009-fix-prompt-contact-viz.md` - Fix prompt file name from fix-001-contact-viz.md to correct duplicate numbering

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
- [ ] `feat-100-bimanual.md` - Support bimanual environment with dexhand_left and dexhand_right

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
- ✅ **refactor-003-imports.md** (2025-07-30) - **MEDIUM** - Clean up mid-file imports for opencv and flask
  - **Root Cause**: Mid-file imports with conditional checking violated fail-fast philosophy and created unnecessary defensive programming patterns
  - **Dependencies Made Required**: Moved opencv-python>=4.5.0 and flask>=2.0.0 from extras_require to INSTALL_REQUIRES in setup.py for consistent availability
  - **Import Cleanup**: Added cv2 and flask imports to top of train.py, removed 35+ lines of conditional import checking and dependency validation logic
  - **Configuration Fix**: Added base/video to config.yaml defaults to ensure video configuration keys are always available
  - **Architecture Compliance**: Eliminated defensive programming patterns, following research code fail-fast philosophy from CLAUDE.md
  - **Testing Verified**: Both basic functionality and video features (recording/streaming) work correctly without import errors
  - **Impact**: Cleaner code structure with ~31 net line reduction, eliminated complex conditional branching, made video dependencies consistently available for training workflows
  - Two files modified with comprehensive testing across video recording and HTTP streaming functionality
- ✅ **refactor-004-render.md** (2025-07-30) - **MEDIUM** - Clarify render option semantics (viewer vs background rendering)
  - **Root Cause**: Legacy render configuration was ambiguous and violated single source of truth principles with assumption-based logic
  - **Configuration Cleanup**: Removed deprecated `recordVideo: false` from config.yaml root level, eliminated all legacy video option references
  - **Consistent Structure**: Added explicit viewer/videoRecord/videoStream defaults to BaseTask.yaml, migrated all test config files to new structure
  - **CLI Enhancement**: Added `viewer=true` CLI alias mapping to `env.viewer=true` for improved usability in train.py
  - **Legacy Removal**: Cleaned up recordVideo handling in examples/dexhand_test.py, updated train.py log messages, fixed documentation examples
  - **Architecture Compliance**: Eliminated assumption-based rendering logic, all decisions now come from explicit configuration (no code defaults)
  - **Impact**: Clear render semantics with independent control of interactive visualization, video recording, and video streaming
  - **Documentation Updates** (2025-07-30): Completed legacy option removal - updated all documentation files (TRAINING.md, guide-http-video-streaming.md, guide-configuration-system.md, README.md, GETTING_STARTED.md, TROUBLESHOOTING.md, guide-physics-tuning.md) to use new configuration names (viewer/videoRecord/videoStream instead of render/recordVideo/streamVideo)
  - Comprehensive cleanup across 23 files with improved configuration clarity and removed technical debt
- ✅ **fix-005-box-bounce-physics.md** (2025-07-29) - **ESSENTIAL** - Fix box bouncing at initialization in BoxGrasping task
  - **Root Cause**: Refactor-005-default-values changed VecTask substeps from hardcoded default 2 to explicit config value 4, making physics simulation more accurate and exposing box positioning precision issues
  - **Physics Analysis**: Higher substeps (4 vs 2) = more accurate collision detection, revealing that box center at z=0.025m placed bottom exactly at z=0 with no clearance for collision sensitivity
  - **Principled Solution**: Adjusted box initial z position from 0.025m to 0.027m (box half-size + 2mm clearance) to work with accurate physics rather than masking the issue
  - **Configuration Fix**: Updated BoxGrasping.yaml with clear comment explaining the clearance requirement for accurate physics simulation
  - **Comprehensive Testing**: Validated fix works with both substeps=2 and substeps=4, confirmed no regression in BaseTask functionality
  - **Architecture Compliance**: Maintained improved physics accuracy (substeps=4) while addressing root positioning cause, aligning with fail-fast philosophy
  - **Impact**: Eliminated box bouncing behavior while preserving accurate physics simulation, no defensive programming added
  - Single configuration change with thorough validation across multiple physics parameter settings
- ✅ **refactor-005-default-values.md** (2025-07-29) - **CRITICAL** - Remove hardcoded defaults from .get() patterns
  - **Root Cause**: Hardcoded numerical defaults in `.get()` patterns violated fail-fast philosophy and masked configuration issues
  - **Critical Fix**: Fixed training pipeline config access pattern (cfg.train["seed"] → cfg["train"]["seed"]) for dict vs OmegaConf compatibility
  - **Comprehensive Cleanup**: Removed all hardcoded defaults from ActionProcessor, VideoManager, ViewerController, ObservationEncoder, TerminationManager, VecTask, and DexHandBase
  - **Config Transparency**: Added explicit clipObservations/clipActions to config.yaml, termination rewards to BaseTask.yaml
  - **Architecture Compliance**: Eliminated defensive programming while preserving legitimate external interface defaults
  - **Testing Verified**: Both test script and training pipeline (BaseTask/BoxGrasping) work correctly
  - **Impact**: Restored training pipeline, improved fail-fast behavior, made all parameters discoverable in config files
  - **Side Effect**: Box bouncing physics behavior change identified and resolved (fixed in fix-005-box-bounce-physics.md)
- ✅ **fix-003-max-iterations.md** (2025-07-29) - **ESSENTIAL** - maxIterations config override and train.py cleanup
  - **Root Cause**: Hardcoded default checks in get_config_overrides() violated fail-fast philosophy and created brittle configuration system
  - **Configuration System Fix**: Removed all hardcoded default comparisons from get_config_overrides(), now always includes key parameters for complete reproducibility
  - **Alias Standardization**: Replaced confusing `maxIter` alias with explicit `maxIterations` for research code clarity (breaking change)
  - **Config Structure Fix**: Corrected train_headless.yaml section from `training:` to `train:` for consistency
  - **Fail-Fast Compliance**: Eliminated defensive programming patterns, configuration values now come from single source of truth
  - **Impact**: Improved configuration system maintainability, eliminated brittle hardcoded checks, enhanced command reproducibility
  - Three files modified with minimal changes following architectural principles
- ✅ **fix-004-dexhand-test-scope.md** (2025-07-29) - **CRITICAL** - DexHand test script functionality restoration
  - **Root Cause**: Missing `unscale_actions` method removed during ActionProcessor refactoring (commit 2bbbaaff) broke plotting functionality
  - **ActionProcessor Enhancement**: Added missing `unscale_actions` method (lines 723-757) with proper action unscaling for both position and position_delta control modes
  - **Documentation Fix**: Updated usage examples to show correct `task.controlMode=position` syntax instead of incorrect `env.controlMode=position`
  - **Configuration Clarity**: Added important note explaining proper BaseTask configuration override approach
  - **Testing Verified**: Both control modes work without AttributeError, test script completes successfully, plotting functionality fully restored
  - **Impact**: Fixed critical test script crash when plotting enabled, restored debugging capabilities essential for development workflow
  - Minimal code changes with maximum impact on essential debugging and testing infrastructure
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
