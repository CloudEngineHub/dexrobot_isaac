# DexHand Terminology Glossary

This document defines project-specific terminology and concepts used throughout the DexHand codebase and documentation.

## The Problem: Domain-Specific Language Barriers

Robotics RL research sits at the intersection of multiple specialized domains:
- **Robotics terminology**: DOFs, rigid bodies, joint spaces, quaternions
- **RL concepts**: Episodes, policies, observation spaces, reward shaping
- **GPU computing jargon**: Vectorization, tensor operations, kernel synchronization
- **Software architecture patterns**: Components, decorators, dependency injection

Researchers waste time decoding terminology instead of understanding concepts, and miscommunication about technical terms leads to implementation errors.

## The Solution: Unified Terminology with Clear Definitions

This glossary provides authoritative definitions that:
1. **Bridge domains**: Connect robotics, RL, and software concepts
2. **Clarify project-specific usage**: Define how standard terms apply here
3. **Provide context**: Explain why certain terms matter
4. **Enable precise communication**: Reduce ambiguity in technical discussions

> **How to Use:** This glossary supports all other documentation. See [System Architecture](ARCHITECTURE.md) for design overview, [Task Creation Guide](guide-task-creation.md) for implementation details, and [TRAINING.md](../TRAINING.md) for usage workflows.

## Core Concepts

### **Component**
A modular software unit with a single responsibility that follows the two-stage initialization pattern. Examples: `PhysicsManager`, `ActionProcessor`, `RewardCalculator`. Components communicate through well-defined interfaces and property decorators.

### **Two-Stage Initialization**
An architectural pattern where component setup is split into two phases:
1. **Stage 1**: Basic configuration and setup that doesn't depend on simulation
2. **Stage 2**: Finalization after `control_dt` measurement and simulation startup

This pattern is necessary because Isaac Gym's timing parameters can only be determined at runtime.

### **Fail-Fast**
A design philosophy where the system crashes immediately on errors rather than hiding them with fallbacks or defensive programming. This exposes bugs at their source for faster debugging in research environments.

### **Single Source of Truth**
An architectural principle ensuring shared state (like `control_dt`, `device`, `num_envs`) lives in exactly one location and is accessed through property decorators to prevent synchronization issues.

## Environment and Simulation

### **Task**
A specific manipulation scenario (e.g., `BaseTask`, `BlindGrasping`) that defines:
- Scene setup (objects, spawn positions)
- Reward functions and termination conditions
- Task-specific observations and reset behavior

### **Environment (env)**
A parallel instance of the simulation. The system supports multiple environments running simultaneously (`num_envs`) for efficient batch training.

### **Episode**
A complete sequence from environment reset to termination, with a maximum length defined by `episodeLength` in the configuration.

### **Control Timestep (control_dt)**
The time interval between policy actions, measured dynamically as `physics_dt × physics_steps_per_control`. This value cannot be configured statically due to Isaac Gym's variable stepping behavior.

### **DOF (Degrees of Freedom)**
Individual controllable joints of the robot. The DexHand has specific DOF mappings for base movement and finger articulation.

## Reward System

### **Reward Component**
An individual reward term (e.g., `alive`, `height_safety`, `object_height`) that contributes to the total reward. Each component has an associated weight configured in the task YAML.

### **Task Reward Terms**
Reward components specific to a particular task, computed in the task's `compute_task_reward_terms()` method.

### **Base Reward Terms**
Common reward components (like `alive`, `finger_velocity`) that apply across multiple tasks, computed in the base task implementation.

## Action and Control

### **Action Mode**
The control scheme for robot actions:
- `position`: Absolute joint positions
- `position_delta`: Relative position changes
- `velocity`: Joint velocities
- `effort`: Joint torques/forces

### **Policy Controls**
Configuration determining which parts of the robot the RL policy directly controls:
- `policy_controls_fingers`: Whether the policy controls finger joints
- `policy_controls_base`: Whether the policy controls base position/orientation

### **Action Processor**
The component responsible for scaling, transforming, and applying actions to the simulation based on the configured action mode and control scheme.

## Observations

### **Observation Space**
The structured collection of sensor data provided to the RL policy, including robot state, object positions, contact information, etc.

### **Observation Encoder**
The component that constructs the observation space and encodes raw simulation data into the format expected by the RL algorithm.

### **Proprioceptive Observations**
Internal robot state information (joint positions, velocities, forces) that doesn't depend on external sensors.

### **Exteroceptive Observations**
External world information (object positions, contact forces) that represents the robot's perception of its environment.

## Technical Terms

### **Tensor Manager**
The component responsible for acquiring and caching Isaac Gym simulation state tensors. Handles the refresh cycle and provides efficient access to simulation data.

### **Physics Manager**
The component that controls physics simulation stepping, timing measurement, and interaction with Isaac Gym's simulation loop.

### **Rigid Body States**
Isaac Gym tensors containing position, orientation, linear velocity, and angular velocity for all rigid bodies in the simulation.
- **Shape**: `[num_envs, num_bodies, 13]` (position: 3, orientation: 4, linear_vel: 3, angular_vel: 3)
- **Access**: `self.rigid_body_states` via `TensorManager`

### **Contact Forces**
Forces generated by collisions between objects, stored as tensors and used for reward computation and contact detection.
- **Shape**: `[num_envs, num_bodies, 3]` (force vector per body)
- **Access**: `self.contact_forces` via `TensorManager`

### **DOF State**
Joint position and velocity information for all degrees of freedom in the simulation.
- **Shape**: `[num_envs, num_dofs, 2]` (position, velocity per DOF)
- **Access**: `self.dof_state` via `TensorManager`

### **DOF Targets**
Target positions for degrees of freedom used in PD control.
- **Shape**: `[num_envs, num_dofs]` (target position per DOF)
- **Access**: `self.action_processor.current_targets`

### **Observation Buffer (obs_buf)**
Concatenated observations sent to the RL policy for training.
- **Shape**: `[num_envs, obs_dim]` (observation vector per environment)
- **Access**: `self.obs_buf` via `ObservationEncoder`

## Configuration and Training

### **Hydra Configuration**
The hierarchical configuration system used for managing parameters. Supports composition, inheritance, and command-line overrides.

### **Configuration Composition**
Hydra's ability to combine multiple YAML files (base configs, task configs, training configs) into a single configuration at runtime.

### **Checkpoint**
A saved model state that can be loaded for testing or resumed training. Supports smart resolution (e.g., `checkpoint=latest` automatically finds the most recent checkpoint).

### **Test Mode**
A training script mode (`train.test=true`) that loads a checkpoint and runs the policy without further training, typically with rendering enabled.

### **Hot Reload**
A test mode feature that automatically reloads checkpoints at specified intervals, useful for debugging during training.

## System States

### **Reset Buffer (reset_buf)**
A boolean tensor indicating which environments need to be reset. Set to `True` when episodes terminate and cleared to `False` after reset completion.
- **Shape**: `[num_envs]` (boolean per environment)
- **Access**: `self.reset_buf` via environment

### **Termination**
The end of an episode due to success, failure, or timeout. Handled by the `TerminationManager` component with separate logic for different termination types.

### **Success/Failure Criteria**
Task-specific conditions that determine episode termination, such as reaching a target position or collision with the ground.

## Development Concepts

### **Scientific Computing Mindset**
An approach to code design that prioritizes:
- Vectorized tensor operations over loops
- Mathematical elegance over defensive programming
- Pure functional computations
- Precomputed indices and transformations

### **Property Decorator Pattern**
A Python pattern used throughout the codebase to provide access to shared state without creating direct references:
```python
@property
def control_dt(self):
    return self.parent.physics_manager.control_dt
```

### **Vectorization**
Writing computations that operate on entire batches of environments simultaneously rather than using loops over individual environments.

## Research Workflow

### **Domain Randomization (DexHand-Specific)**
Systematic variation of simulation parameters in the DexHand environment, typically applied through the ResetManager component with configurable randomization ranges for hand position, orientation, and object properties.

This glossary serves as the authoritative reference for terminology used throughout the DexHand project documentation and codebase.

## See Also

- **[System Architecture](ARCHITECTURE.md)** - How these concepts fit together in the overall system
- **[Task Creation Guide](guide-task-creation.md)** - Practical application of these concepts
- **[TRAINING.md](../TRAINING.md)** - Usage of terminology in training workflows
- **[README.md](../README.md)** - Complete documentation hub and project overview
