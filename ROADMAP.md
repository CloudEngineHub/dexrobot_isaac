# DexRobot Isaac Development Roadmap

Task tracking and project status for the DexRobot Isaac development. See @CLAUDE.md for the complete 3-phase development workflow process and architectural guidelines.

## Active Sprint

Sprint focus: **Research & Performance Optimization**
Documentation system overhaul completed successfully. All documentation now follows motivation-first protocol with accurate parameter names and cross-references. Focus shifts to research tasks and performance optimization.

## Backlog

### High Priority Tasks

#### Research Tasks (`rl_*`)
Policy tuning, physics improvements, and reward engineering.
- [ ] `rl-001-blind-grasping-task.md` - Break down blind grasping training difficulties into specific fixes

#### Performance Tasks (`perf_*`)
System performance optimization and analysis.
- [ ] `perf-000-physics-speed.md` - Determine optimal physics accuracy for training performance

### Medium Priority Tasks

#### Feature Enhancements (`feat_*`)
New functionality and API enhancements.
- [ ] `feat-100-bimanual.md` - Support bimanual environment with dexhand_left and dexhand_right
- [ ] `feat-110-domain-randomization.md` - Structured domain randomization scheme

### Low Priority Tasks

#### Feature Enhancements (`feat_*`)
Advanced features and ecosystem integration.
- [ ] `feat-200-task-support.md` - Support more manipulation tasks (IsaacGymEnvs, RoboHive examples)
- [ ] `feat-300-simulator-backend.md` - Support multiple simulators as backends with unified interface (IsaacSim/IsaacLab, Genesis, MJX/MuJoCo Playground)

#### Completed Meta Tasks (`meta_*`)
- [x] `meta-000-workflow-setup.md` - ✅ **COMPLETED** - AI development workflow design and implementation
- [x] `meta-001-programming-guideline.md` - ✅ **COMPLETED** - Consolidate programming guidelines to user memory
- [x] `meta-002-backward-compatibility.md` - ✅ **COMPLETED** - Remove backward compatibility requirement from CLAUDE.md guidelines
- [x] `meta-003-precommit.md` - ✅ **COMPLETED** - Add precommit hook handling to CLAUDE.md git workflow
- [x] `meta-004-docs.md` - ✅ **COMPLETED** (2025-08-01) - Documentation development protocol added to CLAUDE.md guidelines

## Completed Tasks

### Recently Completed
- ✅ **doc-005-system-overhaul.md** (2025-08-11) - **DOC** - Documentation system compliance overhaul with CLAUDE.md protocol
- ✅ **doc-003-action-processing-illustration.md** (2025-08-11) - **DOC** - Action processing illustration and documentation enhancements
- ✅ **doc-004-training.md** (2025-08-11) - **DOC** - TRAINING.md integration and outdated options fix
- ✅ **doc-002-control-dt-illustration.md** (2025-08-07) - **DOC** - Control dt vs physics_dt illustration showing parallel simulation constraint
- ✅ **guide-indefinite-testing.md** (2025-08-01) - **DOC** - Complete rewrite of indefinite testing guide with motivation-first structure
- ✅ **doc-000-cp.md** (2025-08-01) - **MEDIUM** - Documentation for `cp -P` symbolic link copying in experiment management
- ✅ **feat-004-action-rule-example.md** (2025-08-01) - **MEDIUM** - Action rule conceptual examples integrated into action pipeline documentation
- ✅ **config-simplification** (2025-08-01) - **REFACTOR** - Test configuration files simplification and physics standardization
- ✅ **fix-010-max-deltas.md** (2025-07-31) - **RESOLVED** - max_deltas scaling investigation and configuration fix
- ✅ **fix-009-config-consistency.md** (2025-07-31) - **ESSENTIAL** - Configuration files cleanup for obsolete legacy options
- ✅ **feat-000-streaming-port.md** (2025-07-30) - **MEDIUM** - HTTP streaming port management enhancement
- ✅ **feat-001-video-fps-control.md** (2025-07-31) - **MEDIUM** - Automatic video FPS calculation for accurate playback timing
- ✅ **fix-008-termination-reason-logging.md** (2025-07-31) - **ESSENTIAL** - Fix termination reason logging to show current status
- ✅ **fix-006-metadata-keys.md** (2025-07-30) - **ESSENTIAL** - Fix git metadata saving error with config keys
- ✅ **fix-007-episode-length-of-grasping.md** (2025-07-30) - **ESSENTIAL** - Fix BlindGrasping task early termination behavior
- ✅ **feat-002-indefinite-testing.md** (2025-07-30) - **MEDIUM** - Enable indefinite testing mode
- ✅ **refactor-009-config-yaml.md** (2025-07-30) - **MEDIUM** - Configuration architecture cleanup
- ✅ **refactor-008-config-key-casing.md** (2025-07-30) - **MEDIUM** - Unify config key naming to snake_case under task section
- ✅ **refactor-007-step-architecture.md** (2025-07-30) - **MEDIUM** - Investigate step processing architecture consistency
- ✅ **refactor-007-blind-grasping.md** (2025-07-30) - **MEDIUM** - Rename BoxGrasping to BlindGrasping task
- ✅ **refactor-003-imports.md** (2025-07-30) - **MEDIUM** - Clean up mid-file imports for opencv and flask
- ✅ **refactor-004-render.md** (2025-07-30) - **MEDIUM** - Clarify render option semantics (viewer vs background rendering)
- ✅ **fix-005-box-bounce-physics.md** (2025-07-29) - **ESSENTIAL** - Fix box bouncing at initialization in BlindGrasping task
- ✅ **refactor-005-default-values.md** (2025-07-29) - **CRITICAL** - Remove hardcoded defaults from .get() patterns
- ✅ **fix-003-max-iterations.md** (2025-07-29) - **ESSENTIAL** - maxIterations config override and train.py cleanup
- ✅ **fix-004-dexhand-test-scope.md** (2025-07-29) - **CRITICAL** - DexHand test script functionality restoration
- ✅ **fix-002-consistency.md** (2025-07-28) - **CRITICAL** - Test script and training consistency fixes
- ✅ **fix-001-contact-viz.md** (2025-07-28) - **ESSENTIAL** - Contact visualization NameError fix
- ✅ **fix-001-reward-logging-logic.md** (2025-07-28) - **CRITICAL** - RewardComponentObserver windowed statistics fix
- ✅ **fix-000-tb-metrics.md** (2025-07-28) - **ESSENTIAL** - TensorBoard data retention fix
- ✅ **refactor-001-episode-length.md** (2025-07-28) - **MAJOR** - Episode length configuration architectural fix
- ✅ **refactor-002-graphics-manager-in-parent.md** (2025-07-25) - **MAJOR** - Graphics component architecture alignment
- ✅ **refactor-006-action-processing.md** (2025-07-25) - **MAJOR** - Action processing timing refactoring
- ✅ **rl-000-penetration.md** (2025-07-25) - **CRITICAL** - Penetration prevention system implementation
- ✅ **meta-000-workflow-setup.md** (2025-01-25) - AI development workflow implementation
- ✅ **meta-001-programming-guideline.md** (2025-01-25) - Programming guidelines consolidation

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
