# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands
- Install: `pip install -e .`
- Run simple test: `python examples/dexhand_test.py`
- Run with options: `python examples/dexhand_test.py --episode-length 200 --debug --movement-speed 0.5`
- Test GPU pipeline: `python examples/dexhand_test.py --use-gpu-pipeline --steps 10`

## Note on Code Structure
- The main implementation is in the `dex_hand_env` directory
- Legacy code is in the `legacy/DexHandEnv_obsolete` directory (for reference only)
- Simple examples are in the `examples` directory

## Code Style Guidelines
- Imports: Standard library → third-party → local application imports
- Formatting: 4-space indentation, ~80 char line length
- Types: Use PEP 484 type hints where possible
- Naming: snake_case for functions/variables, PascalCase for classes, UPPER_SNAKE_CASE for constants
- Documentation: Include docstrings for functions and classes
- Configuration: Use Hydra with YAML files in cfg/ directory
- Error handling: Use try/except with specific exceptions and informative messages
- Tensors: Always make assertions on shape when defining new tensors to catch shape issues early
- Vectorization: Prefer vectorized tensor operations over for-loops and if-else for efficiency and code aesthetics

## Development Philosophy
- IMPORTANT: This is research codebase, not production-environment code, so always prefer failing fast over suppressing the error and using a fallback.
- NEVER add fallback logic or default values that hide errors. Always expose errors immediately with clear error messages.
- When tensor operations fail, ALWAYS raise an exception rather than continuing with empty tensors.
- Let the environment crash with informative error messages instead of hiding issues.
- NEVER mark issues as fixed or completed (with checkmarks or otherwise) unless explicitly confirmed by the user.
- NEVER claim that something is working correctly if there are still errors occurring.
- DO NOT proceed to the next issue unless the current issue is fully resolved and the user explicitly confirms.
- Always be precise about the current state - segmentation faults are NEVER acceptable, even if some steps seem to run.

## CRITICAL: No Fallback Values Policy
**This is a research codebase where failing fast is essential for debugging.**

### FORBIDDEN Patterns (NEVER use these):
- `if x is None: x = default_value` → Use `raise RuntimeError("x is None")`
- `try: ... except: use_fallback` → Let it crash with informative error
- `value = x if x else fallback` → Raise error if x is invalid
- `value = 0.785 # fallback` → Raise error instead of magic numbers
- `# Fallback to default` → This comment itself indicates wrong approach
- `return default_value` → Use `raise RuntimeError("Missing required value")`

### REQUIRED Pattern:
When something is missing or invalid, ALWAYS:
```python
raise RuntimeError(f"Descriptive error message explaining what is missing/wrong")
```

### Examples:
❌ WRONG:
```python
if self.dof_props is None:
    self.dof_props = torch.zeros((26, 6))  # default
```

✅ CORRECT:
```python
if self.dof_props is None:
    raise RuntimeError("DOF properties not initialized. Cannot proceed without DOF limits.")
```

❌ WRONG:
```python
dof_idx = self.dof_names.index(joint_name) if joint_name in self.dof_names else 0
```

✅ CORRECT:
```python
if joint_name not in self.dof_names:
    raise RuntimeError(f"Joint '{joint_name}' not found in DOF names: {self.dof_names}")
dof_idx = self.dof_names.index(joint_name)
```

### CRITICAL: Don't Handle States That Should Never Occur

Beyond avoiding fallback values, **never write code to handle programming errors**:

❌ WRONG - Defensive programming that hides bugs:
```python
if contact_forces is not None:
    obs_dict["contact_forces"] = contact_forces.reshape(...)
else:
    obs_dict["contact_forces"] = torch.zeros(...)  # "handles" broken state
```

✅ CORRECT - Let it crash to expose the bug:
```python
# contact_forces should NEVER be None during observation computation
# If it is None, that indicates a tensor initialization bug that must be fixed
obs_dict["contact_forces"] = contact_forces.reshape(...)
```

❌ WRONG - Avoiding work by allowing broken behavior:
```python
if self.action_processor is not None:
    targets = self.action_processor.current_targets
else:
    targets = torch.zeros(...)  # "handles" missing component
```

✅ CORRECT - Force architectural problems to be fixed:
```python
# action_processor should NEVER be None when computing observations
# If it is None, that indicates an initialization bug that must be fixed
targets = self.action_processor.current_targets
```

**Key Insight**: If code executes correctly, only one path in if-else will be entered. Writing branches to handle "impossible" states is mental laziness that hides bugs instead of forcing them to be fixed.

### Remember:
- Research code needs to expose problems, not hide them
- A clear error message is more helpful than silently wrong behavior
- Debugging is easier when failures happen at the source of the problem
- Magic numbers and fallback values make debugging nearly impossible
- **Don't write defensive code for states that should never occur - let it crash and fix the root cause**

## Tensor Operations and Vectorization

### Prefer Vectorized Operations Over Loops

For tensor operations, always prefer vectorized operations over explicit loops and conditionals:

❌ WRONG - Explicit loops:
```python
for env_idx in range(self.num_envs):
    for finger_idx in range(5):
        start_idx = finger_idx * 7
        poses[env_idx, start_idx:start_idx+3] = rigid_body_states[env_idx, tip_idx, :3]
```

✅ CORRECT - Vectorized operations:
```python
# Process all environments and fingers at once
poses = rigid_body_states[:, fingertip_indices, :7].reshape(self.num_envs, -1)
```

❌ WRONG - Element-wise conditionals:
```python
for i in range(targets.shape[0]):
    if targets[i] > limits[i]:
        targets[i] = limits[i]
```

✅ CORRECT - Vectorized clamping:
```python
targets = torch.clamp(targets, min_limits, max_limits)
```

**Benefits:**
- **Performance**: GPU/CPU optimization for batch operations
- **Readability**: Clear intent without loop boilerplate
- **Maintainability**: Less error-prone than index manipulation
- **Scalability**: Automatically handles different batch sizes

**When loops are acceptable:**
- Small, fixed-size iterations (e.g., 5 fingers) where vectorization is complex
- One-time setup/initialization code
- When readability significantly improves over vectorized equivalent

## Isaac Gym Version
- No need to handle different isaac gym versions. Refer to the docs of the current isaac gym version in @reference/isaacgym when necessary.

## Documentation Reference
- Official API docs are in @reference/isaacgym/docs/

## Simulation Architecture Notes

### ⚠️ CRITICAL: Read Design Caveats First
**Before modifying physics or model configuration, read:**
- `docs/IMPORTANT_DESIGN_DECISIONS.md` - Quick reference of critical caveats
- `docs/physics_caveats.md` - Detailed explanations and troubleshooting

### Hand Base Configuration
- The hand model uses a fixed base link (`asset_options.fix_base_link = True`) to anchor the hand to the world
- This is an important design choice because:
  1. The physical movement of the hand is controlled by the actuated DOFs *within* the hand actor (ARTx, ARTy, ARTz, ARRx, ARRy, ARRz), not by moving the entire actor in the world
  2. With the base fixed, we don't need to worry about gravity causing the hand to fall when no actions are applied
  3. The base DOFs still exist and can be controlled, allowing for controlled movement of the hand base
  4. This approach ensures consistent and stable hand positioning during simulation

### DOF Control  
- The ARTx/y/z DOFs control **RELATIVE** translation from initial position
- The ARRx/y/z DOFs control **RELATIVE** rotation from initial orientation
- **IMPORTANT**: ARTz=0.0 means "stay at initial Z" (spawn point), NOT "go to world Z=0"
- When action=0 is applied, these DOFs should maintain their initial position

### Action Scaling and Mapping
The environment implements proper action scaling from the normalized action space [-1, +1] to actual DOF limit ranges:

#### Finger Coupling Logic
- **12 actions** are mapped to **19 finger DOFs** using a coupling system
- Coupling map in `ActionProcessor` handles the relationship between action indices and joint names
- Each action can control multiple joints with different scaling factors

#### Action Scaling Formula
```python
# Map action from [-1, +1] to [dof_min, dof_max]
scaled_action = (action_value + 1.0) * 0.5 * (dof_max - dof_min) + dof_min
final_target = scaled_action * coupling_scale
```

#### DOF Limit Verification
- **Spread joints (2_1, 4_1)**: [0.0, 0.3] radians
- **Spread joint (5_1)**: [0.0, 0.6] radians (with 2x coupling scale)
- **Bend joints**: [0.0, 1.57] radians (π/2)

#### Action Mapping Examples
- `action = -1.0` → DOF minimum limit (e.g., 0.0 rad for spread joints)
- `action = +1.0` → DOF maximum limit (e.g., 0.3 rad for joints 2_1/4_1)
- `action = 0.0` → DOF middle range (e.g., 0.15 rad for joints 2_1/4_1)

#### Implementation Location
- Primary logic: `dex_hand_env/components/action_processor.py:380-400`
- DOF limits extracted from MJCF models in `TensorManager`
- Handles both `position` and `position_delta` control modes correctly

### Isaac Gym Model Requirements
- **CRITICAL**: Isaac Gym requires `limited="true"` in MJCF for joint limits to work
- Our model generation scripts automatically add this attribute
- Joint properties (stiffness/damping) come directly from MJCF, no code overrides

### Physics Stepping Architecture
- **Multiple Physics Steps for Reliable Resets**: The environment requires multiple physics steps for reliable object reset and stabilization. This is crucial for:
  1. Allowing reset objects to settle physically before control resumes
  2. Ensuring consistent initial states across environments
  3. Preventing unstable behavior when objects are first spawned or repositioned

- **Physics vs. Control Timesteps**:
  1. `physics_dt`: The fundamental timestep for the physics simulation (typically 0.01s)
  2. `control_dt`: The timestep at which control actions are applied, equal to `physics_dt * physics_steps_per_control_step`
  3. The environment may need multiple physics steps per control step for stability

- **Automatic Step Counter**:
  1. The environment automatically detects how many physics steps are needed per control step
  2. This is implemented through the `_step_physics()` wrapper that tracks physics step counts
  3. When more physics steps than expected occur (e.g., during resets), the system adjusts `physics_steps_per_control_step` accordingly
  4. This adaptive approach ensures stable physics regardless of reset complexity

- **Physics Stepping Best Practices**:
  1. Always use the `_step_physics()` wrapper instead of calling `self.gym.simulate()` directly
  2. This ensures proper tracking of physics steps and automatic adjustment of control frequency
  3. Both the main stepping function and reset operations should use this wrapper
  4. The wrapper handles both simulation and result fetching in one call

## Observation System Design

### Observation Dictionary vs. Observation Tensor

The observation system follows a clear separation between **complete data availability** and **selective data inclusion**:

**Observation Dictionary (`obs_dict`):**
- Contains ALL computed observations regardless of configuration
- Used for debugging, inspection, analysis, and plotting
- Always available for external tools and utilities
- Includes disabled/inactive observation components with zero values

**Observation Tensor (`obs_buf`):**
- Contains ONLY observations specified in `observation_keys` configuration
- Used for RL training and policy inference
- Concatenated in the order specified by `observation_keys`
- Optimized for memory and computational efficiency

**Design Rationale:**
- **Research Flexibility**: Researchers can always inspect any observation component even if it's disabled in the RL pipeline
- **Debugging Support**: All observation data remains accessible for analysis and troubleshooting
- **Performance**: RL training only processes the minimal required observation set
- **Configuration Control**: Users explicitly specify which observations the policy should receive

**Example:**
```python
# Configuration disables hand_pose for RL training
observation_keys = ["base_dof_pos", "finger_dof_pos"]

# But obs_dict still contains hand_pose for inspection
obs_dict = env.get_observations_dict()
hand_pose = obs_dict["hand_pose"]  # Always available

# While obs_tensor only contains the enabled components
obs_tensor = env.obs_buf  # Only base_dof_pos + finger_dof_pos
```

This design ensures that **disabling an observation component means excluding it from RL training**, not making it unavailable for inspection.

## DexRobot Project Roadmap

### Current Focus
1. **Refactor DexHandBase to Component Architecture** (✅ DONE)
   - Extract initialization logic to separate component
   - Create observation encoder component
   - Create action decoder component
   - Create reset handler component
   - Move physics step management to dedicated component
   - Split large methods into smaller, focused functions
   - Fix issues in refactored version to get it running properly
   - Ensure observation space dimensions are correctly initialized
   - Fix tensor initialization order issues
   - Correct root state tensor indexing for hand position
   - Add safety checks for tensor dimensions
   - Ensure physics stepping works correctly with components

### Phase 1: Critical Fixes & Core Functionality
2. **Fix GPU Pipeline Issues** (✅ DONE)
   - Fixed CUDA memory access errors by correcting tensor acquisition order
   - Changed DOF properties handling from asset to actor-based queries for GPU compatibility
   - Added critical PhysX parameters for GPU pipeline stability
   - GPU pipeline now works correctly with `--use-gpu-pipeline` flag
   - Both `--no-gpu-pipeline` and `--use-gpu-pipeline` modes are now functional

3. **Verify DOF Control & Naming** (✅ DONE)
   - Test the correct correspondence between DoFs and names
   - Verify all 26 DOFs are properly accessible and controllable
   - Fix "Error verifying target positions" issue with get_dof_target
   - Create mapping documentation between DOF indices and names
   - Fixed ARTz initialization to 0.0 (relative motion from spawn point)

4. **Test Action Modes** (✅ DONE)
   - Verify position control mode works correctly
   - Verify position_delta control mode works correctly  
   - Test with different control frequencies
   - Ensure action=0 maintains position properly
   - Added CLI flags for testing different control configurations
   - Implemented rule-based control for non-policy controlled parts

5. **Verify Observation System**
   - Check that observation tensors have correct shape and content
   - Verify all required state information is included
   - Test observation dictionary keys and values
   - Add validation assertions for observation components

6. **Fix Reset & Initialization Logic**
   - Ensure correct physics steps per control step calculation
   - Verify initial pose is set correctly on environment creation
   - Test reset_idx functionality with different environment indices
   - Ensure tensor dimensions are consistent after reset

### Phase 2: Refactoring & Architecture
7. **Naming Consistency**
   - Decide on consistent naming (dex_hand vs dexhand)
   - Standardize class and file names
   - Update import statements and references
   - Document naming conventions in CLAUDE.md

8. **Clean up Logging & Debugging**
   - Remove excessive debug print statements
   - Implement proper logging levels (debug, info, warning, error)
   - Add configurable verbosity
   - Make error messages more informative

9. **Improve Testing Framework**
   - Add batched simulation tests (num_envs > 1)
   - Create validation tests for observations and actions
   - Add performance benchmarks
   - Create regression tests for fixed bugs

### Phase 3: Advanced Features & Validation
10. **Visualization & Analysis Tools**
    - Add Rerun integration for visualization
    - Implement plotting of observations, actions, and rewards
    - Create tools to visualize DOF positions and targets
    - Add state recording and playback capability

11. **Reward System Implementation**
    - Verify reward terms calculation
    - Add reward component visualization
    - Create reward debugging tools
    - Test reward scaling and normalization

12. **RL Algorithm Integration**
    - Test with PPO implementation
    - Verify learning works correctly
    - Benchmark performance with different configs
    - Create sample training scripts

13. **ROS2 Integration**
    - Implement ROS2 interface based on reference implementation
    - Add joint state publishers and subscribers
    - Create services for environment control
    - Ensure compatibility with ROS2 visualization tools

### Implementation Notes
- GPU pipeline issues have been resolved; both `--use-gpu-pipeline` and `--no-gpu-pipeline` modes work
- When refactoring, maintain backward compatibility where possible
- Add unit tests for each component during refactoring
- Keep the main branch stable and use feature branches for development

### GPU Pipeline Notes
- **Critical for GPU pipeline**: Call `gym.prepare_sim()` AFTER creating all actors but BEFORE acquiring tensors
- **DOF Properties**: Use `gym.get_actor_dof_properties()` instead of `gym.get_asset_dof_properties()` for GPU compatibility
- **PhysX Parameters**: GPU pipeline requires specific parameters like `contact_collection = CC_LAST_SUBSTEP` and `always_use_articulations = True`
