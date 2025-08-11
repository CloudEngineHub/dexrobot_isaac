# DexHand Documentation

Welcome to the DexHand documentation. This directory contains comprehensive guides and references for the DexHand dexterous manipulation environment.

## Getting Started

**New to DexHand?** Start here:

1. **[GETTING_STARTED.md](GETTING_STARTED.md)** - 10-minute setup to your first trained policy ⭐
2. **[../README.md](../README.md)** - Project overview and installation
3. **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture and design principles
4. **[guide-environment-lifecycle.md](guide-environment-lifecycle.md)** - How all components work together
5. **[guide-task-creation.md](guide-task-creation.md)** - Create your first custom task

## Documentation Categories

### 📋 Training & Configuration

**Complete guide for training RL policies and managing experiments:**

- **[../TRAINING.md](../TRAINING.md)** - Comprehensive training workflows, configuration options, and experiment management

### 📚 Core Guides

**Essential reading for understanding the DexHand system:**

- **[guide-environment-lifecycle.md](guide-environment-lifecycle.md)** - Overview of simulation loop and component interactions
- **[guide-component-initialization.md](guide-component-initialization.md)** - Two-stage initialization and actor creation patterns
- **[guide-task-creation.md](guide-task-creation.md)** - Step-by-step task implementation guide

### 🔧 Component Guides

**Detailed guides for specific system components:**

- **[guide-action-pipeline.md](guide-action-pipeline.md)** - Action processing and control modes
- **[guide-observation-system.md](guide-observation-system.md)** - Observation encoding and management
- **[guide-environment-resets.md](guide-environment-resets.md)** - Environment reset logic and state management
- **[guide-termination-logic.md](guide-termination-logic.md)** - Episode termination criteria
- **[guide-physics-tuning.md](guide-physics-tuning.md)** - Physics parameter tuning
- **[guide-viewer-controller.md](guide-viewer-controller.md)** - Camera and visualization control

### 🐛 Debugging & Troubleshooting

**Problem-solving resources:**

- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Comprehensive troubleshooting guide for all common issues
- **[guide-debugging.md](guide-debugging.md)** - Component debugging and real-world case studies
- **[reference-coordinate-systems.md](reference-coordinate-systems.md)** - Fixed base motion and coordinate quirks

### 📖 Reference Documentation

**Technical specifications and deep dives:**

- **[reference-dof-control-api.md](reference-dof-control-api.md)** - DOF control API and action mapping
- **[reference-physics-implementation.md](reference-physics-implementation.md)** - Low-level physics implementation details
- **[reference-coordinate-systems.md](reference-coordinate-systems.md)** - Coordinate system behavior and transforms

### 🏗️ Architecture & Design

**High-level design and decisions:**

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture overview
- **[design_decisions.md](design_decisions.md)** - Critical design decisions and caveats
- **[GLOSSARY.md](GLOSSARY.md)** - Terminology definitions

## Recommended Reading Paths

### For New Users
1. [GETTING_STARTED.md](GETTING_STARTED.md) → [../README.md](../README.md) → [ARCHITECTURE.md](ARCHITECTURE.md)
2. [../TRAINING.md](../TRAINING.md) for comprehensive training workflows
3. [guide-environment-lifecycle.md](guide-environment-lifecycle.md) for understanding component interactions
4. [guide-task-creation.md](guide-task-creation.md) for implementing custom tasks
5. [guide-debugging.md](guide-debugging.md) when things go wrong

### For Developers
1. [ARCHITECTURE.md](ARCHITECTURE.md) → [guide-component-initialization.md](guide-component-initialization.md)
2. [design_decisions.md](design_decisions.md) for understanding constraints
3. [reference-coordinate-systems.md](reference-coordinate-systems.md) for coordinate quirks

### For Researchers
1. [guide-environment-lifecycle.md](guide-environment-lifecycle.md) → [guide-task-creation.md](guide-task-creation.md)
2. [guide-action-pipeline.md](guide-action-pipeline.md) + [guide-observation-system.md](guide-observation-system.md)
3. [reference-dof-control-api.md](reference-dof-control-api.md) for control details

## Documentation Conventions

### File Naming
- **`guide-*.md`**: How-to guides and tutorials
- **`reference-*.md`**: Technical specifications and API documentation
- **`UPPERCASE.md`**: High-level architecture and glossaries

### Cross-References
All documents include "See Also" sections linking to related materials. Follow these links to explore related concepts.

### Code Examples
Code examples are tested and follow the project's fail-fast philosophy. Copy-paste examples should work as-is.

## Contributing to Documentation

When adding or updating documentation:

1. **Follow naming conventions**: Use `guide-` or `reference-` prefixes with hyphens
2. **Add cross-references**: Include "See Also" sections linking to related docs
3. **Focus on repository-specific content**: Avoid duplicating external documentation
4. **Update this README**: Add new documents to the appropriate category

## External Resources

For topics not covered here, see:

- **[Isaac Gym Documentation](https://developer.nvidia.com/isaac-gym)** - Physics simulation setup
- **[rl_games Documentation](https://github.com/Denys88/rl_games)** - Training algorithms

---

**Questions?** Check [guide-debugging.md](guide-debugging.md) for troubleshooting or [GLOSSARY.md](GLOSSARY.md) for terminology definitions.
