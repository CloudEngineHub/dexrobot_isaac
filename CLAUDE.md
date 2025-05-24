# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands
- Install: `pip install -e .`
- Run simple test: `python examples/simple_dexhand_test.py`
- Run with options: `python examples/simple_dexhand_test.py --episode_length 200 --debug --movement_speed 0.5`

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
- This is research codebase, not production-environent code, so always prefer failing fast over suppressing the error and using a fallback.

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