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

## Development Philosophy
- IMPORTANT: This is research codebase, not production-environment code, so always prefer failing fast over suppressing the error and using a fallback.
- NEVER add fallback logic or default values that hide errors. Always expose errors immediately with clear error messages.
- When tensor operations fail, ALWAYS raise an exception rather than continuing with empty tensors.
- Let the environment crash with informative error messages instead of hiding issues.
- NEVER mark issues as fixed or completed (with checkmarks or otherwise) unless explicitly confirmed by the user.
- NEVER claim that something is working correctly if there are still errors occurring.
- DO NOT proceed to the next issue unless the current issue is fully resolved and the user explicitly confirms.
- Always be precise about the current state - segmentation faults are NEVER acceptable, even if some steps seem to run.

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

3. **Verify DOF Control & Naming**
   - Test the correct correspondence between DoFs and names
   - Verify all 25 DOFs are properly accessible and controllable
   - Fix "Error verifying target positions" issue with get_dof_target
   - Create mapping documentation between DOF indices and names

4. **Test Action Modes**
   - Verify position control mode works correctly
   - Verify position_delta control mode works correctly
   - Test with different control frequencies
   - Ensure action=0 maintains position properly

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
