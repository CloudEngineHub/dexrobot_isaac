# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands
- Install: `pip install -e .`
- Run simple test: `python examples/dexhand_test.py`
- Run with options: `python examples/dexhand_test.py --episode-length 200 --debug --movement-speed 0.5`
- Test different control modes: `python examples/dexhand_test.py --control-mode position_delta --policy-controls-fingers true --policy-controls-base false`

## Development Philosophy

### Fail Fast - No Defensive Programming
This is research code where exposing bugs immediately is critical. NEVER hide errors with fallbacks.

❌ FORBIDDEN:
```python
if x is None: x = default_value  # NO!
try: ... except: use_fallback   # NO!
value = x if x else fallback     # NO!
if hasattr(obj, 'attr'): ...     # NO! Let AttributeError expose bugs
```

✅ REQUIRED:
```python
if x is None:
    raise RuntimeError("x is None - this indicates initialization bug")
# Let code crash immediately to expose problems at their source
```

### Think Like a Scientist
Write elegant mathematical code, not defensive business logic.

**Defensive Programming Clarification:**
- ✅ DO check for external failures (hardware, file I/O, network)
- ❌ DON'T check if your own dependencies are None
- ❌ DON'T add fallbacks for your own logic errors
- If a dependency is required at init, it should NEVER be None later

❌ WRONG - Business programmer mindset:
```python
if self.policy_controls_hand_base:
    if self.actions.shape[1] > 0:
        base_actions = self.actions[:, :self.NUM_BASE_DOFS]
        # ... 20 more lines of branching
```

✅ CORRECT - Scientific computing mindset:
```python
# Use masks and vectorization
scaled_actions[:, self.active_target_mask] = self._scale_actions_to_limits(actions)
```

**Key Principles:**
1. **Masking > Branching**: Boolean masks replace conditional logic
2. **Precompute Everything**: Runtime should be pure math
3. **Vectorize Operations**: Think in tensors, not loops
4. **Function Pointers**: Assign functions during init, not runtime branching

### Fail-Fast for Required Dependencies
When a component requires something to function, NEVER check if it exists before using it. This was the root cause of the viewer camera not focusing on startup.

❌ WRONG - Defensive check that silently fails:
```python
# Update camera position if following robot
if self.viewer_controller:
    if self.hand_rigid_body_indices:  # Silent failure if None!
        hand_positions = self.rigid_body_states[env_indices, self.hand_rigid_body_indices, :3]
        self.viewer_controller.update_camera_position(hand_positions)
```

✅ CORRECT - Fail fast if required dependency is missing:
```python
# Camera MUST follow robot - fail if indices not initialized
if self.hand_rigid_body_index is None:
    raise RuntimeError("hand_rigid_body_index is None - initialization failed")
hand_positions = self.rigid_body_states[:, self.hand_rigid_body_index, :3]
self.viewer_controller.update_camera_position(hand_positions)
```

Key principle: If something is required for correct operation, it should NEVER be None after initialization. Don't check - just use it and let it fail fast if there's a bug.

### Issue Resolution Protocol - CRITICAL
The AI must NEVER claim an issue is resolved without explicit user confirmation, especially in long-running troubleshooting tasks.

❌ WRONG - Premature resolution claims:
```
"The issue has been fixed by updating the configuration."
"This should resolve the problem."
"The bug is now resolved."
"Now HTTP video streaming is working."
"The fix resolves the original issue."
```

✅ CORRECT - Seek explicit confirmation:
```
"I've implemented a potential fix. Please test and confirm if this resolves the issue."
"The changes are complete. Can you verify the problem is fixed?"
"Please run the test and let me know if the issue persists."
"I've made changes that should help - please test if the streaming works now."
```

**Key principle:** Only the user can confirm that an issue is truly resolved. The AI provides fixes and requests verification, but never assumes success without user confirmation.

### Debugging Protocol - CRITICAL
When investigating issues:
1. **Test thoroughly** - Run the actual failing scenario, don't just assume fixes work
2. **Check end-to-end** - Verify the complete workflow, not just individual components
3. **Wait for user feedback** - Always ask the user to confirm if the issue is resolved
4. **Document actual behavior** - Report what actually happens, not what should happen
5. **Never claim success** - If testing shows intermittent results, crashes, or hangs, report the actual behavior
6. **Distinguish between partial progress and full resolution** - Component initialization ≠ working system

## Critical Design Caveats

### Fixed Base with Relative Motion
- Hand uses `fix_base_link = True` - won't fall under gravity
- **ALL motion is relative to spawn position**:
  - `ARTz = 0.0` → stay at spawn height
  - `ARTz = +0.1` → move 0.1m UP from spawn
  - NOT absolute world coordinates!

### Floating Hand Coordinate Quirk
- Model has built-in 90° Y-axis rotation
- When ARRx=ARRy=ARRz=0, quaternion is [0, 0.707, 0, 0.707] not [0, 0, 0, 1]
- Use `hand_pose_arr_aligned` observation for ARR-aligned orientation

### Isaac Gym Requirements
- Joint limits MUST have `limited="true"` in MJCF
- GPU pipeline needs specific call order: create actors → acquire tensors → `prepare_sim()`
- Properties come from MJCF files - no code overrides

## Code Style
- Imports: Standard library → third-party → local
- Use loguru logger, not print statements
- Include shape assertions for new tensors
- Prefer vectorized operations over loops
- Use type hints where possible

### Git Commit Messages
- **ALWAYS run `git diff` and read the entire diff before writing commit message**
- Commit message must accurately reflect ALL changes, not just the most recent fix
- Include all file changes, config updates, and code modifications in the message

## Implementation Guidelines

### Study Before Modifying
Before changing any component:
1. **Understand existing abstractions**: Use grep/search to find similar patterns
2. **Follow established patterns**: Don't reinvent what already exists
3. **Check naming conventions**: Maintain consistency with existing code

### Respect Component Responsibilities
Each component has a specific purpose. Don't violate separation of concerns:
- **Tasks**: Compute raw task-specific values (unscaled rewards, success criteria)
- **RewardCalculator**: Handles ALL reward weighting and aggregation
- **ObservationEncoder**: Manages observation space construction
- **ActionProcessor**: Handles action scaling and control mode logic

Example of CORRECT separation:
```python
# In task's compute_task_reward_terms():
rewards["object_height"] = height_above_table  # Raw value, NO multiplication by weight

# In RewardCalculator's compute_total_reward():
weight = self.reward_weights.get("object_height", 0.0)
weighted_reward = reward * weight  # Weighting happens HERE only
```

### Architectural Principles - Task Abstraction
- **NEVER put task-specific logic in base classes or general components**
- Base classes should not have knowledge of specific task implementations
- Always think abstractly - if you find yourself checking for specific task attributes (like 'box_actor_indices'), you're violating the abstraction
- When implementing fixes, choose the most general solution that works for all tasks
- Example: During reset, apply root states for ALL actors rather than checking for specific task objects

### Write Minimal Code
- Don't create unnecessary instance variables
- Use existing abstractions rather than reimplementing
- If you're writing more than 5 lines for something simple, you're probably doing it wrong

### Validate Understanding First
Before implementing:
1. Articulate what each component's responsibility is
2. Ensure changes align with the component's purpose
3. If unsure, study how existing features use the abstraction

## AI Development Workflow

This project uses a structured 3-phase workflow for managing development tasks, with all todo items organized in the `@prompts/` directory and tracked in `@ROADMAP.md`.

### Task Organization

**Task Categories (by prefix):**
- `meta_*`: Workflow, tooling, and project organization
- `refactor_*`: Code quality improvements and architectural enhancements
- `feat_*`: New functionality and API enhancements
- `fix_*`: Bug fixes and issue resolution
- `rl_*`: Research tasks - policy tuning, physics, reward engineering

**File Structure:**
```
@prompts/
├── meta-001-programming-guideline.md
├── refactor-001-episode-length.md
├── rl-000-penetration.md
└── ...
```

### 3-Phase Development Process

**Phase 1: Ultrathink & Context Gathering**
- Read todo item from `@prompts/`
- Use sequential thinking to understand problem context and scope
- Consider architectural constraints (two-stage init, component boundaries, fail-fast philosophy)
- For RL tasks: analyze physics implications and policy behavior
- Expand brief todo into detailed PRD/architecture document

**Phase 2: Implementation Planning**
- Create detailed step-by-step implementation plan
- Identify which components will be modified
- Check for architectural principle violations
- Plan testing approach using existing test commands
- Request explicit user approval before proceeding

**Phase 3: Implementation & Validation**
- Execute plan methodically using TodoWrite for progress tracking
- Maintain component responsibility separation
- Follow single source of truth principles
- Test thoroughly with project's test commands
- Request user review and explicit confirmation of issue resolution
- Update `@ROADMAP.md` with completion status

### Workflow Rules

1. **One item per session**: Focus on single todo item for quality
2. **No premature resolution claims**: Always request user confirmation
3. **Respect architectural constraints**: Two-stage init, component boundaries, fail-fast
4. **Test thoroughly**: Use existing test commands before claiming completion
5. **Update tracking**: Mark todos complete and update @ROADMAP.md

### Quality Gates

- All code must pass existing test commands
- Component responsibilities must be respected
- Two-stage initialization must be maintained
- No defensive programming allowed (fail fast principle)
- User must explicitly confirm issue resolution
- Git commits must reflect ALL changes made

## Component Architecture

### Two-Stage Initialization Pattern

The DexHand environment uses a **two-stage initialization pattern** that is a **core architectural principle** and must be respected by all components.

#### Why Two-Stage is Necessary

The pattern exists because `control_dt` can only be determined at runtime by measuring actual physics behavior:

```python
# control_dt = physics_dt × physics_steps_per_control
# where physics_steps_per_control is measured, not configured
```

**Why measurement is required:**
- Environment resets require variable physics steps to stabilize
- Isaac Gym may add internal physics steps during state changes
- GPU pipeline timing variations
- Multi-environment synchronization effects

This is **not a design flaw** - it's the correct engineering solution for interfacing with unpredictable simulation behavior.

#### The Two-Stage Lifecycle

**Stage 1: Construction + Basic Initialization**
```python
# Create components with known dependencies
action_processor = ActionProcessor(parent=self)
action_processor.initialize_from_config(config)
# Component is functional but cannot access control_dt yet
```

**Stage 2: Finalization After Measurement**
```python
# Measure control_dt by running dummy control cycle
self.physics_manager.start_control_cycle_measurement()
# ... run measurement cycle ...
self.physics_manager.finish_control_cycle_measurement()

# Now finalize all components
action_processor.finalize_setup()  # Can now access control_dt
task.finalize_setup()
```

#### Component Development Rules

**✅ CORRECT: Use property decorators for control_dt access**
```python
@property
def control_dt(self):
    """Access control_dt from physics manager (single source of truth)."""
    return self.parent.physics_manager.control_dt
```

**❌ WRONG: Don't check if control_dt exists**
```python
# This violates "fail fast" principle
if self.physics_manager.control_dt is None:
    raise RuntimeError("control_dt not available")
```

**✅ CORRECT: Trust initialization order**
```python
# After finalize_setup(), control_dt is guaranteed to exist
def compute_velocity_scaling(self):
    dt = self.control_dt  # This WILL work post-finalization
    return action_delta / dt
```

#### Implementation Guidelines

1. **Split initialization logic**: Basic setup in `__init__`/`initialize_from_config()`, control_dt-dependent logic in `finalize_setup()`
2. **Use property decorators**: Always access `control_dt` via `self.parent.physics_manager.control_dt`
3. **No defensive programming**: Don't check if `control_dt` exists - trust the initialization order
4. **Document dependencies**: Clearly mark which methods require finalization

### Component Development Guidelines

When creating new components for the DexHand environment, follow these standards:

#### Component Structure Pattern
```python
class MyComponent:
    """
    Component description and responsibilities.

    This component provides functionality to:
    - Specific responsibility 1
    - Specific responsibility 2
    """

    def __init__(self, parent):
        """Initialize with parent reference only."""
        self.parent = parent
        self.gym = parent.gym
        self.sim = parent.sim

        # Component state variables
        self._initialized = False

    @property
    def device(self):
        """Access device from parent (single source of truth)."""
        return self.parent.device

    @property
    def control_dt(self):
        """Access control_dt from physics manager (single source of truth)."""
        return self.parent.physics_manager.control_dt

    def initialize_from_config(self, config):
        """Phase 1: Initialize with configuration (no control_dt access)."""
        # Set up everything that doesn't need control_dt
        pass

    def finalize_setup(self):
        """Phase 2: Complete initialization (control_dt now available)."""
        # Set up everything that needs control_dt
        self._initialized = True
```

#### Property Decorator Standards

**✅ ALWAYS use property decorators for parent access:**
```python
@property
def tensor_manager(self):
    """Access tensor_manager from parent (single source of truth)."""
    return self.parent.tensor_manager

@property
def num_envs(self):
    """Access num_envs from parent (single source of truth)."""
    return self.parent.num_envs
```

**❌ NEVER store direct references to sibling components:**
```python
def __init__(self, parent):
    # WRONG - creates coupling
    self.tensor_manager = parent.tensor_manager
    self.physics_manager = parent.physics_manager
```

#### Responsibility Separation

**Each component has a single, clear responsibility:**

- **PhysicsManager**: Physics simulation, stepping, tensor refresh
- **TensorManager**: Simulation tensor acquisition and management
- **ActionProcessor**: Action scaling, control mode logic, DOF mapping
- **ObservationEncoder**: Observation space construction and encoding
- **RewardCalculator**: Reward computation and weighting
- **TerminationManager**: Success/failure criteria evaluation
- **ResetManager**: Environment reset and randomization logic
- **ViewerController**: Camera control and visualization

**❌ NEVER mix responsibilities across components**
**✅ ALWAYS delegate to the appropriate component**

### Single Source of Truth
- `control_dt` lives only in PhysicsManager
- `device` lives only in parent (DexHandBase)
- `num_envs` lives only in parent (DexHandBase)
- Components access via property decorators
- No manual synchronization needed
- Property decorators provide clean access without coupling

## Critical Lessons - Optimization and Refactoring

### ALWAYS Understand Before Optimizing
When asked to optimize or refactor code, FIRST understand what it does:
- Read the complete implementation and understand its output format
- Identify all API contracts and external interfaces
- Map out the data flow and dependencies
- Only then plan optimizations that preserve exact behavior

❌ WRONG:
```python
# Original logs to: reward_breakdown/all/raw/episode/component
# Changed to: reward_components/component/step
# This breaks downstream tools expecting the original format!
```

✅ CORRECT:
```python
# Preserve exact same output keys and structure
# Only change the internal implementation
```

### NEVER Hardcode Dynamic Data
If the system can discover information dynamically, use that mechanism:
- Query configuration systems for available components
- Use existing registries and managers
- Avoid duplicating knowledge that exists elsewhere

❌ WRONG - Hardcoded component list:
```python
common_components = ["alive", "height_safety", "finger_velocity", ...]
task_components = ["object_height", "grasp_approach", ...]
```

✅ CORRECT - Query from source of truth:
```python
# Get components from reward calculator's configuration
enabled_components = [name for name, weight in reward_weights.items() if weight != 0]
```

### Respect Exact Requirements
When asked to "optimize without changing logic":
- Preserve ALL aspects of the output (keys, structure, meaning)
- Only change the implementation approach
- Maintain backward compatibility
- Test that outputs remain identical

### Query Existing Systems for Truth
- Don't maintain separate lists of components/features
- Use the system's own discovery mechanisms
- Single source of truth principle applies everywhere

## Project Status

See [`@ROADMAP.md`](@ROADMAP.md) for detailed project status, completed features, and future development plans.

## Essential Documentation
- **Critical Caveats**: [`docs/design_decisions.md`](docs/design_decisions.md) - READ FIRST!
- **DOF/Action Reference**: [`docs/api_dof_control.md`](docs/api_dof_control.md)
- **Observation System**: [`docs/guide-observation-system.md`](docs/guide-observation-system.md)
- **Component Init**: [`docs/guide-component-initialization.md`](docs/guide-component-initialization.md)
