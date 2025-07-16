# DexHand Architecture Overview

This document provides a high-level overview of the DexHand system architecture, design decisions, and component interactions.

**Related Documentation:**
- [Task Creation Guide](guide-task-creation.md) - For implementing custom tasks
- [Component Initialization](guide-component-initialization.md) - For detailed component setup
- [TRAINING.md](../TRAINING.md) - For usage workflows and configuration
- [Terminology Glossary](GLOSSARY.md) - For concept definitions and technical terms

## Core Architectural Principles

### 1. Component-Based Design
The system is built around modular, single-responsibility components that communicate through well-defined interfaces:

- **PhysicsManager**: Physics simulation control and timing
- **TensorManager**: Simulation tensor acquisition and caching
- **ActionProcessor**: Action scaling, control modes, and DOF mapping
- **ObservationEncoder**: Observation space construction and encoding
- **RewardCalculator**: Reward computation and weighting
- **TerminationManager**: Success/failure criteria evaluation
- **ResetManager**: Environment reset and randomization
- **ViewerController**: Camera control and visualization

### 2. Fail-Fast Philosophy
The system prioritizes exposing bugs immediately rather than hiding them:

- No defensive programming for internal logic errors
- Immediate crashes on configuration errors
- Extensive validation during initialization
- Clear error messages at the source of problems

### 3. Single Source of Truth
Shared state is managed through property decorators to prevent synchronization issues:

- `control_dt` lives only in PhysicsManager
- `device` and `num_envs` live only in DexHandBase
- Components access via property decorators, not direct references

### 4. Scientific Computing Mindset
Code is optimized for mathematical computation and research workflows:

- Vectorized tensor operations over loops
- Masking and filtering instead of conditional branching
- Precomputed indices and transforms
- Pure functional reward computations

### 5. 4-Section Configuration Hierarchy
Configuration is organized into four logical sections for clear separation of concerns:

- **`sim`**: Physics simulation parameters (dt, substeps, PhysX settings)
- **`env`**: Environment setup (numEnvs, task objects, rendering)
- **`task`**: RL task definition (episodes, observations, rewards, termination)
- **`train`**: Training algorithm configuration (rl_games parameters, logging)

This hierarchy enables modular physics configurations, task composition, and clear override patterns. See the [Configuration System Guide](guide-configuration-system.md) for detailed usage.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         DexHandBase                              │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │ PhysicsManager  │ │ TensorManager   │ │ ActionProcessor │   │
│  │                 │ │                 │ │                 │   │
│  │ - control_dt    │ │ - state tensors │ │ - action modes  │   │
│  │ - sim stepping  │ │ - refresh cycle │ │ - DOF mapping   │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
│                                                                 │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │ObservationEncode│ │ RewardCalculator│ │TerminationMgr   │   │
│  │                 │ │                 │ │                 │   │
│  │ - obs encoding  │ │ - weighted sums │ │ - success/fail  │   │
│  │ - space mgmt    │ │ - component agg │ │ - timeout logic │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
│                                                                 │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │  ResetManager   │ │ ViewerController│ │    TaskImpl     │   │
│  │                 │ │                 │ │                 │   │
│  │ - reset logic   │ │ - camera mgmt   │ │ - task rewards  │   │
│  │ - randomization │ │ - visualization │ │ - observations  │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Isaac Gym                                 │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │   Simulation    │ │   Environments  │ │     Assets      │   │
│  │                 │ │                 │ │                 │   │
│  │ - physics step  │ │ - parallel envs │ │ - robot models  │   │
│  │ - tensor API    │ │ - actor mgmt    │ │ - scene objects │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Initialization Sequence

1. **Configuration Loading**: Hydra loads hierarchical YAML configs
2. **Component Creation**: All components instantiated with config references
3. **Physics Setup**: Isaac Gym simulation and actors created
4. **Tensor Acquisition**: TensorManager gets handles to simulation tensors
5. **Control Timing Measurement**: PhysicsManager measures actual control_dt
6. **Finalization**: All components complete setup with measured parameters

### Simulation Step

```
Action Input
    │
    ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ ActionProcessor │────▶│ PhysicsManager  │────▶│  Isaac Gym Sim  │
│ - Scale actions │     │ - Apply actions │     │ - Physics step  │
│ - Control modes │     │ - Step physics  │     │ - Update state  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                         │
┌─────────────────┐     ┌─────────────────┐             │
│ Observations    │◀────│ TensorManager   │◀────────────┘
│ Rewards         │     │ - Refresh cache │
│ Terminations    │     │ - State tensors │
└─────────────────┘     └─────────────────┘
```

## Two-Stage Initialization

### Problem Statement
Isaac Gym's `control_dt` cannot be determined until after physics simulation starts, but components need this value for proper initialization. This creates a circular dependency.

### Solution
Split initialization into two phases:

**Stage 1: Basic Setup**
- Components store configuration parameters
- Set up data structures that don't depend on simulation
- Establish parent-child relationships

**Stage 2: Finalization**
- Measure `control_dt` through dummy physics cycle
- Complete setup requiring simulation parameters
- Validate all components are properly initialized

### Implementation Pattern
```python
class Component:
    def __init__(self, parent):
        self.parent = parent
        self._initialized = False

    def initialize_from_config(self, config):
        # Stage 1: Basic setup
        self.config_param = config["param"]

    def finalize_setup(self):
        # Stage 2: Complete setup
        self.dt = self.parent.physics_manager.control_dt
        self._initialized = True

    @property
    def control_dt(self):
        return self.parent.physics_manager.control_dt
```

## Configuration Management

### Hydra Integration
The system uses Hydra for hierarchical configuration:

```yaml
# Base configuration structure
defaults:
  - task: BaseTask
  - train: BaseTaskPPO
  - _self_

env:
  numEnvs: 1024
  device: "cuda:0"
  render: null  # Auto-detect

task:
  name: "BaseTask"
  episodeLength: 1000

training:
  test: false
  checkpoint: null
  maxIterations: 10000
```

### Override Patterns
```bash
# Full path overrides (recommended)
python train.py env.numEnvs=2048 task.episodeLength=500

# Configuration composition
python train.py --config-name=debug task=BoxGrasping

# Runtime behavior
python train.py training.test=true training.checkpoint=latest
```

## DexHand-Specific Performance Characteristics

### Component Architecture Impact
- Two-stage initialization adds startup overhead but prevents runtime failures
- Property decorators provide single source of truth without performance cost
- Component separation enables targeted optimization of bottlenecks

### Isaac Gym Integration Patterns
- Tensor refresh cycle synchronized with component update sequence
- GPU tensor operations minimize CPU-GPU transfers
- Batched reset operations across multiple environments
- Control timestep measurement affects overall throughput

## Extension Points

### Adding New Tasks
1. Inherit from `BaseTask`
2. Implement required methods (`compute_task_observations`, etc.)
3. Add task registration to factory
4. Create task-specific configuration

### Adding New Components
1. Follow two-stage initialization pattern
2. Use property decorators for shared state access
3. Implement clean interfaces with parent component
4. Add comprehensive logging and validation

### Custom Reward Functions
1. Implement in task's `compute_task_reward_terms()`
2. Return dictionary of component rewards
3. Configure weights in task YAML
4. RewardCalculator handles aggregation automatically

## DexHand Design Decisions

### Why Two-Stage Initialization
Isaac Gym's `control_dt` cannot be determined until physics starts, creating circular dependencies. Two-stage initialization resolves this by separating configuration from runtime-dependent setup.

### Why Fail-Fast Architecture
Research code benefits from immediate error exposure rather than silent failures. Component crashes indicate configuration or logic errors that need fixing, not working around.

### Why Component Separation
Modular architecture enables independent testing, targeted optimization, and clear responsibility boundaries essential for research iteration cycles.

This architecture enables efficient, maintainable reinforcement learning research while providing clear extension points for new capabilities.

## See Also

- **[Task Creation Guide](guide-task-creation.md)** - Practical implementation guide for new tasks
- **[TRAINING.md](../TRAINING.md)** - Training workflows and configuration
- **[Terminology Glossary](GLOSSARY.md)** - Definitions of architectural concepts
- **[Design Decisions](design_decisions.md)** - Critical caveats and implementation details
- **[Component Initialization](guide-component-initialization.md)** - Two-stage initialization deep dive
