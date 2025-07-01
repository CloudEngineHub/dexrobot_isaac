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

### Write Minimal Code
- Don't create unnecessary instance variables
- Use existing abstractions rather than reimplementing
- If you're writing more than 5 lines for something simple, you're probably doing it wrong

### Validate Understanding First
Before implementing:
1. Articulate what each component's responsibility is
2. Ensure changes align with the component's purpose
3. If unsure, study how existing features use the abstraction

## Component Architecture

### Two-Phase Initialization
Many components require two-phase init due to control_dt dependency:
```python
# Phase 1: Basic setup
action_processor = ActionProcessor(...)
action_processor.initialize_from_config(config)

# Phase 2: After physics manager ready
action_processor.finalize_setup()  # Now control_dt available
```

### Single Source of Truth
- `control_dt` lives only in PhysicsManager
- Components access via property decorators
- No manual synchronization needed

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

See [`ROADMAP.md`](ROADMAP.md) for detailed project status, completed features, and future development plans.

## Essential Documentation
- **Critical Caveats**: [`docs/design_decisions.md`](docs/design_decisions.md) - READ FIRST!
- **DOF/Action Reference**: [`docs/api_dof_control.md`](docs/api_dof_control.md)
- **Observation System**: [`docs/guide_observation_system.md`](docs/guide_observation_system.md)
- **Component Init**: [`docs/guide_component_initialization.md`](docs/guide_component_initialization.md)
