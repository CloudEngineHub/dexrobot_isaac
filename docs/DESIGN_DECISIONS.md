# 🚨 Critical Design Decisions

**READ THIS FIRST** - Essential design caveats that affect physics, modeling, and control. These decisions solve fundamental challenges in simulating dexterous hands on GPUs.

## The Problem: Simulating Floating Hands on GPUs

Traditional robot simulations assume robots are mounted to a base or table. But dexterous manipulation research needs hands that can:
1. Move freely in 3D space without complex arm kinematics
2. Run thousands of copies in parallel on GPUs
3. Maintain stable, deterministic physics behavior
4. Avoid coordinate system singularities and gimbal lock

These requirements conflict with standard approaches and lead to several non-obvious design decisions.

## Critical Design Decision 1: Fixed Base with Internal DOF Control

### The Problem
Isaac Gym's GPU pipeline struggles with free-floating objects that use 6-DOF rigid body dynamics. The physics solver becomes unstable when thousands of floating hands interact with objects simultaneously.

### The Solution
Use `fix_base_link = True` with internal articulated DOFs for movement:
- Hand is anchored to world coordinates (won't fall under gravity)
- Movement via internal ARTx/y/z/Rx/y/z DOFs, not actor translation
- **CRITICAL**: All motion is relative to spawn position

### Why This Matters
```python
# What you might expect (WRONG):
hand.set_world_position([1.0, 2.0, 3.0])  # Doesn't work with fixed base

# What actually happens (CORRECT):
# If spawned at [0, 0, 0.5]:
ARTz = 0.0   # Hand stays at spawn position (world Z = 0.5)
ARTz = +0.1  # Hand moves to world Z = 0.6 (spawn + 0.1)
ARTz = -0.1  # Hand moves to world Z = 0.4 (spawn - 0.1)
```

### Trade-offs
- **Benefit**: Rock-solid physics stability with thousands of environments
- **Cost**: Must think in relative coordinates, not absolute positions
- **Alternative rejected**: Free-floating rigid bodies cause non-deterministic physics

## Critical Design Decision 2: Built-in 90° Y-Axis Rotation

### The Problem
The floating hand model needs a coordinate system that:
1. Aligns finger forward direction with +X (robotics convention)
2. Keeps palm facing down with zero rotation (natural grasp orientation)
3. Avoids gimbal lock in common hand orientations

### The Solution
The model has a built-in 90° Y-axis rotation in its kinematic tree:
- When ARRx=ARRy=ARRz=0, quaternion is [0, 0.707, 0, 0.707] not [0, 0, 0, 1]
- This pre-rotation optimizes the usable workspace
- `hand_pose_arr_aligned` observation compensates for this rotation

### Why This Matters
```python
# Raw hand orientation includes built-in rotation:
hand_quat = obs["hand_pose"][3:7]  # [0, 0.707, 0, 0.707] when ARR=0

# ARR-aligned orientation for intuitive control:
aligned_quat = obs["hand_pose_arr_aligned"][3:7]  # [0, 0, 0, 1] when ARR=0

# Converting to Euler angles:
# Raw: [0°, 90°, 0°] - confusing!
# Aligned: [0°, 0°, 0°] - intuitive!
```

### Trade-offs
- **Benefit**: Optimal workspace coverage, avoids gimbal lock in common poses
- **Cost**: Must understand the coordinate transform for proper control
- **Alternative rejected**: Standard orientation hits gimbal lock during grasping

## Critical Design Decision 3: Joint Limits Must Be Explicit

### The Problem
Isaac Gym's GPU pipeline pre-compiles physics constraints. It ignores MJCF joint ranges unless explicitly marked as limited, leading to joints that violate their physical limits during simulation.

### The Solution
Every joint with range limits must have `limited="true"`:
```xml
<!-- WRONG - limits ignored, joint can spin 360° -->
<joint name="finger_joint" range="0 1.3" />

<!-- CORRECT - limits enforced by physics -->
<joint name="finger_joint" range="0 1.3" limited="true" />
```

### Why This Matters
Without explicit limits:
- Fingers bend backwards through the palm
- Joints exceed mechanical stops
- Learned policies exploit non-physical configurations
- Real robot deployment fails catastrophically

### Trade-offs
- **Benefit**: Physically realistic joint constraints
- **Cost**: Must regenerate models when changing limits
- **Alternative rejected**: Runtime limit enforcement too slow for GPU pipeline

## Critical Design Decision 4: Physics Properties from MJCF Only

### The Problem
Runtime modification of physics properties (stiffness, damping) breaks GPU kernel compilation and causes non-deterministic behavior across parallel environments.

### The Solution
All physics properties come directly from MJCF files:
- Actuator `kp` → PD controller stiffness
- Joint `damping` → velocity damping
- **No runtime overrides** - edit MJCF files and regenerate

### Implementation
```bash
# After editing MJCF properties:
cd assets/dexrobot_mujoco/scripts
./regenerate_all.sh
```

### Trade-offs
- **Benefit**: Deterministic physics across all GPU environments
- **Cost**: Must regenerate models for any physics changes
- **Alternative rejected**: Runtime property changes cause GPU synchronization stalls

## Critical Design Decision 5: GPU Pipeline Constraints

### The Problem
Isaac Gym's GPU pipeline requires specific settings and call sequences. Deviating causes silent failures, wrong physics, or crashes after hours of training.

### The Solution
Strict requirements for GPU stability:
```yaml
# Minimum physics iterations for joint stability
num_position_iterations: 32

# Contact collection mode for GPU pipeline
contact_collection: 1  # CC_LAST_SUBSTEP

# Critical call sequence:
# 1. Create all actors
# 2. Acquire tensor handles
# 3. gym.prepare_sim()
# ANY other order causes silent corruption
```

### Why This Matters
These aren't arbitrary choices - they're discovered through painful debugging:
- Lower iterations: joints drift apart over time
- Wrong contact mode: contacts missed or duplicated
- Wrong call order: tensors contain garbage data

### Trade-offs
- **Benefit**: Stable physics with 4096+ parallel environments
- **Cost**: Inflexible initialization sequence
- **Alternative rejected**: CPU pipeline 1000x slower

## Summary: Living with These Decisions

These design decisions create a stable, efficient platform for dexterous manipulation research:

1. **Think in relative coordinates** - All positions relative to spawn point
2. **Understand the built-in rotation** - Use ARR-aligned observations
3. **Respect MJCF as single source of truth** - No runtime physics changes
4. **Follow GPU pipeline requirements exactly** - Order matters

The constraints may seem restrictive, but they enable training policies with thousands of parallel hands at 60+ FPS on a single GPU. That speed difference transforms what research is possible.

## Related Documentation

- **DOF/Action Reference**: [`reference-dof-control-api.md`](reference-dof-control-api.md) - Understanding the DOF structure
- **Observation System**: [`guide-observation-system.md`](guide-observation-system.md) - Working with coordinate transforms
- **Component Initialization**: [`guide-component-initialization.md`](guide-component-initialization.md) - Why initialization order matters
- **Physics Tuning**: [`guide-physics-tuning.md`](guide-physics-tuning.md) - Tuning within MJCF constraints
- **Implementation Details**: [`reference-physics-implementation.md`](reference-physics-implementation.md) - Deep dive into physics pipeline
