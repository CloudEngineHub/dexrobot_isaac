# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands
- Install: `pip install -e .`
- Run simple test: `python examples/dexhand_test.py`
- Run with options: `python examples/dexhand_test.py --episode-length 200 --debug --movement-speed 0.5 --no-gpu-pipeline`

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
- This is research codebase, not production-environment code, so always prefer failing fast over suppressing the error and using a fallback.

## Interpreter Notes
- Do not need to run the python scripts because your interpreter has some issue gathering the output. Tell me to run the test when you're ready and I'll return you the outputs.

## Isaac Gym Version
- No need to handle different isaac gym versions. Refer to the docs of the current isaac gym version in @reference/isaacgym when necessary.

## Documentation Reference
- Official API docs are in @reference/isaacgym/docs/

## Simulation Architecture Notes
### Hand Base Configuration
- The hand model uses a fixed base link (`asset_options.fix_base_link = True`) to anchor the hand to the world
- This is an important design choice because:
  1. The physical movement of the hand is controlled by the actuated DOFs *within* the hand actor (ARTx, ARTy, ARTz, ARRx, ARRy, ARRz), not by moving the entire actor in the world
  2. With the base fixed, we don't need to worry about gravity causing the hand to fall when no actions are applied
  3. The base DOFs still exist and can be controlled, allowing for controlled movement of the hand base
  4. This approach ensures consistent and stable hand positioning during simulation

### DOF Control
- The ARTx/y/z DOFs control translation along the x/y/z axes
- The ARRx/y/z DOFs control rotation around the x/y/z axes
- When action=0 is applied, these DOFs should maintain their current position

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

### Phase 1: Critical Fixes & Core Functionality
1. **Fix GPU Pipeline Issues**
   - Investigate and fix CUDA memory access errors
   - Test with both `--no-gpu-pipeline` and `--use-gpu-pipeline` modes
   - Identify specific components that cause GPU pipeline failures

2. **Verify DOF Control & Naming**
   - Test the correct correspondence between DoFs and names
   - Verify all 25 DOFs are properly accessible and controllable
   - Fix "Error verifying target positions" issue with get_dof_target
   - Create mapping documentation between DOF indices and names

3. **Test Action Modes**
   - Verify position control mode works correctly
   - Verify position_delta control mode works correctly
   - Test with different control frequencies
   - Ensure action=0 maintains position properly

4. **Verify Observation System**
   - Check that observation tensors have correct shape and content
   - Verify all required state information is included
   - Test observation dictionary keys and values
   - Add validation assertions for observation components

5. **Fix Reset & Initialization Logic**
   - Ensure correct physics steps per control step calculation
   - Verify initial pose is set correctly on environment creation
   - Test reset_idx functionality with different environment indices
   - Ensure tensor dimensions are consistent after reset

### Phase 2: Refactoring & Architecture
6. **Naming Consistency**
   - Decide on consistent naming (dex_hand vs dexhand)
   - Standardize class and file names
   - Update import statements and references
   - Document naming conventions in CLAUDE.md

7. **Refactor DexHandBase to Component Architecture**
   - Extract initialization logic to separate component
   - Create observation encoder component
   - Create action decoder component
   - Create reset handler component
   - Move physics step management to dedicated component
   - Split large methods into smaller, focused functions

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
- For CUDA/GPU pipeline issues, start debugging by isolating minimal reproducible examples
- When refactoring, maintain backward compatibility where possible
- Add unit tests for each component during refactoring
- Use the `--no-gpu-pipeline` option while developing until GPU pipeline issues are resolved
- Keep the main branch stable and use feature branches for development