# Documentation Index

This index organizes all DexHand documentation by topic and usage scenario.

## Quick Start

For new users getting started with the DexHand environment:

- **[Getting Started](GETTING_STARTED.md)** - Installation and first training run
- **[Training Guide](TRAINING.md)** - Complete training and testing workflows
- **[System Architecture](ARCHITECTURE.md)** - Understanding the overall system design
- **[Design Decisions](DESIGN_DECISIONS.md)** - Critical design caveats and physics quirks
- **[Git Submodule Setup](git-submodule-setup.md)** - Submodule initialization and troubleshooting

## Development Guides

For developers working with the DexHand environment:

- **[Configuration System](guide-configuration-system.md)** - 4-section hierarchy, physics configs, and customization
- **[Task Creation Guide](guide-task-creation.md)** - Creating custom manipulation tasks
- **[Component Initialization](guide-component-initialization.md)** - Understanding the two-stage initialization pattern
- **[Action Pipeline](guide-action-pipeline.md)** - Action processing and rule-based control
- **[Observation System](guide-observation-system.md)** - Observation encoding and configuration
- **[Debugging Guide](guide-debugging.md)** - Component troubleshooting and debugging workflows
- **[Environment Lifecycle](guide-environment-lifecycle.md)** - Environment creation and stepping
- **[Environment Resets](guide-environment-resets.md)** - Reset logic and randomization
- **[Termination Logic](guide-termination-logic.md)** - Success/failure criteria and episode termination
- **[Viewer Controller](guide-viewer-controller.md)** - Camera control and visualization

## Practical Workflows

Step-by-step guides for common workflows:

- **[Indefinite Testing](guide-indefinite-testing.md)** - Live policy monitoring with hot-reload during training
- **[HTTP Video Streaming](guide-http-video-streaming.md)** - Remote visualization via browser
- **[Physics Tuning](guide-physics-tuning.md)** - Optimizing simulation accuracy vs speed

## API Reference

Technical reference documentation for developers:

- **[DOF and Action Control API](reference-dof-control-api.md)** - Complete DOF mapping and action spaces
- **[Physics Implementation](reference-physics-implementation.md)** - Physics stepping and tensor management
- **[Coordinate Systems](reference-coordinate-systems.md)** - Coordinate frames and transformations
- **[Terminology Glossary](GLOSSARY.md)** - Definitions of key concepts and technical terms

## By Component

Documentation organized by system component:

### Core Components
- **ActionProcessor**: [Action Pipeline](guide-action-pipeline.md), [DOF Control API](reference-dof-control-api.md)
- **PhysicsManager**: [Physics Implementation](reference-physics-implementation.md), [Component Initialization](guide-component-initialization.md)
- **ObservationEncoder**: [Observation System](guide-observation-system.md)
- **TensorManager**: [Physics Implementation](reference-physics-implementation.md)

### Task System
- **Task Interface**: [Task Creation Guide](guide-task-creation.md)
- **Reward System**: [Task Creation Guide](guide-task-creation.md)
- **Reset Management**: [Component Initialization](guide-component-initialization.md)

### Configuration
- **Configuration System**: [Configuration Guide](guide-configuration-system.md)
- **Hydra Configuration**: [Training Guide](TRAINING.md)
- **CLI Aliases**: [Training Guide](TRAINING.md)
- **Physics Configurations**: [Configuration Guide](guide-configuration-system.md)

## By Use Case

Documentation organized by common use cases:

### Training and Testing
- [Getting Started](GETTING_STARTED.md) - First training run
- [Training Guide](TRAINING.md) - Complete training workflows
- [Configuration examples](TRAINING.md) - Common training scenarios

### Custom Task Development
- [Task Creation Guide](guide-task-creation.md) - Step-by-step task creation
- [Observation System](guide-observation-system.md) - Adding task-specific observations
- [Action Pipeline](guide-action-pipeline.md) - Custom action processing
- [Termination Logic](guide-termination-logic.md) - Defining success and failure conditions

### System Understanding
- [System Architecture](ARCHITECTURE.md) - Overall system design
- [Design Decisions](DESIGN_DECISIONS.md) - Critical design caveats and physics quirks
- [Component Initialization](guide-component-initialization.md) - How components work together
- [Physics Implementation](reference-physics-implementation.md) - Physics simulation details
- [Environment Lifecycle](guide-environment-lifecycle.md) - Simulation flow and control

### Debugging and Troubleshooting
- [Troubleshooting Guide](TROUBLESHOOTING.md) - Common issues and solutions
- [Git Submodule Setup](git-submodule-setup.md) - Submodule issues and recovery
- [Component Initialization](guide-component-initialization.md) - Common initialization errors
- [Physics Implementation](reference-physics-implementation.md) - Physics debugging
- [Coordinate Systems](reference-coordinate-systems.md) - Coordinate debugging

## File Organization

```
docs/
├── INDEX.md                           # This file
├── GETTING_STARTED.md                 # Quick setup guide
├── TRAINING.md                        # Training and testing workflows
├── ARCHITECTURE.md                    # System overview
├── DESIGN_DECISIONS.md                # Critical design caveats
├── TROUBLESHOOTING.md                 # Common issues and solutions
├── GLOSSARY.md                        # Terminology reference
├── git-submodule-setup.md             # Submodule setup and troubleshooting
├── guide-configuration-system.md      # Configuration system guide
├── guide-task-creation.md             # Task development guide
├── guide-component-initialization.md  # Component setup guide
├── guide-action-pipeline.md           # Action processing guide
├── guide-observation-system.md        # Observation system guide
├── guide-indefinite-testing.md        # Hot-reload testing guide
├── guide-http-video-streaming.md      # Video streaming setup
├── guide-physics-tuning.md            # Physics parameter tuning
├── reference-dof-control-api.md       # DOF mapping reference
├── reference-physics-implementation.md # Physics reference
└── reference-coordinate-systems.md    # Coordinate systems reference
```

## Contributing to Documentation

When adding new documentation:
1. Update this index file
2. Add cross-references to related documents
3. Follow the established naming patterns
4. Include practical examples and usage patterns
