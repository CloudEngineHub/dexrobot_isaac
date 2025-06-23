# Actor Creation Guide

This guide explains the proper pattern for creating actors in DexHand environments, particularly when tasks need to add objects alongside the robot hand.

## Overview

The DexHand environment follows a specific actor creation order to ensure consistent indexing and proper simulation setup. This is critical for multi-actor environments where tasks add objects like cubes, spheres, or other interactive elements.

## Actor Creation Order

The creation flow follows this sequence:

1. **Task Initialization** - Task is created with placeholder sim/gym references
2. **Environment Creation** - Basic environment setup
3. **Task Asset Loading** - Task loads its assets (meshes, primitives) early
4. **Hand Creation** - Hands are created FIRST (always actor index 0)
5. **Task Object Creation** - Task objects are created AFTER hands

## Implementation Details

### 1. Factory Pattern (factory.py)

```python
# Create task with placeholder references
task = DexGraspTask(None, None, device, num_envs, cfg)

# Create environment (which will set up simulation)
env = DexHandBase(cfg, task, ...)

# Environment handles asset loading and actor creation internally
```

### 2. Environment Initialization (dexhand_base.py)

```python
def _init_components(self):
    # Create simulation
    self.sim = self.create_sim()

    # Update task with real sim/gym instances
    self.task.sim = self.sim
    self.task.gym = self.gym

    # Load task assets BEFORE creating any actors
    self.task.load_task_assets()

    # Create environments
    self._create_envs()

    # Create hands FIRST (ensures hand is actor 0)
    handles = self.hand_initializer.create_hands(self.envs, self.hand_asset)

    # Create task objects AFTER hands
    for i in range(self.num_envs):
        if hasattr(self.task, "create_task_objects"):
            self.task.create_task_objects(self.gym, self.sim, self.envs[i], i)
```

### 3. Task Implementation (e.g., dex_grasp_task.py)

```python
class DexGraspTask(DexTask):
    def load_task_assets(self):
        """Load assets early, before any actors are created."""
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False

        if self.object_type == "cube":
            self.object_asset = self.gym.create_box(
                self.sim,
                self.object_size, self.object_size, self.object_size,
                asset_options
            )

    def create_task_objects(self, gym, sim, env_ptr, env_id):
        """Create task objects AFTER hands."""
        # Hand is already created as actor 0
        # Object will be actor 1
        object_handle = self.gym.create_actor(
            env_ptr,
            self.object_asset,
            object_pose,
            f"object_{env_id}",
            env_id,
            0  # collision group
        )
        self.object_handles.append(object_handle)
```

## Actor Indexing

With the proper creation order:

- **Hand**: Always actor index 0 in each environment
- **Task Objects**: Start from actor index 1

This consistent indexing is critical for:
- DOF control (hand DOFs are accessed via actor 0)
- State tensors (root states indexed by actor)
- Reset operations

## Common Patterns

### Accessing Actor States

```python
def reset_task_state(self, env_ids):
    # Hand is actor 0
    hand_states = self.root_state_tensor[env_ids, 0]

    # Object is actor 1
    object_states = self.root_state_tensor[env_ids, 1]
```

### Multiple Objects

```python
def create_task_objects(self, gym, sim, env_ptr, env_id):
    # Create multiple objects
    # Actor 0: Hand (already created)
    # Actor 1: Object
    # Actor 2: Target visualization
    # Actor 3: Obstacle

    for i, asset in enumerate(self.object_assets):
        actor_idx = i + 1  # Starting from 1 (hand is 0)
        handle = self.gym.create_actor(...)
```

## Best Practices

1. **Always Load Assets First**: Call `load_task_assets()` before creating any actors
2. **Consistent Naming**: Use clear actor names like `"hand_0"`, `"object_0"`, `"target_0"`
3. **Track Actor Counts**: Tasks should define `num_task_actors` for clarity
4. **Document Indices**: Comment which actor has which index in your task

## Troubleshooting

### Issue: Actions not applied in multi-environment setup
**Cause**: Actor indices might be incorrect when objects are created before hands
**Solution**: Ensure hands are created first using the pattern above

### Issue: DOF control affects wrong actor
**Cause**: DOF indices assume hand is actor 0
**Solution**: Follow the creation order to ensure hand is always first

### Issue: Reset operations fail
**Cause**: Tensor indexing assumes specific actor order
**Solution**: Use consistent actor indices as documented

## Example Task Setup

Here's a complete example of a task with proper actor creation:

```python
class MyGraspTask(DexTask):
    def __init__(self, sim, gym, device, num_envs, cfg):
        super().__init__(sim, gym, device, num_envs, cfg)
        self.object_handles = []
        self.num_task_actors = 1  # Just the object
        self.root_state_tensor = None  # Will be set by environment

    def load_task_assets(self):
        """Load assets before actor creation."""
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False

        self.object_asset = self.gym.create_box(
            self.sim, 0.05, 0.05, 0.05, asset_options
        )

    def create_task_objects(self, gym, sim, env_ptr, env_id):
        """Create objects after hand."""
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 0.1)

        handle = gym.create_actor(
            env_ptr, self.object_asset, pose,
            f"object_{env_id}", env_id, 0
        )
        self.object_handles.append(handle)

    def reset_task_state(self, env_ids):
        """Reset with proper actor indices."""
        # Object is actor 1 (hand is 0)
        self.root_state_tensor[env_ids, 1, 0:3] = self.initial_pos[env_ids]
        self.root_state_tensor[env_ids, 1, 3:7] = self.initial_rot[env_ids]
```

## Related Documentation

- [`guide_component_initialization.md`](guide_component_initialization.md) - Component initialization order
- [`reference_physics_implementation.md`](reference_physics_implementation.md) - Physics and tensor management
- [`api_dof_control.md`](api_dof_control.md) - DOF control reference
