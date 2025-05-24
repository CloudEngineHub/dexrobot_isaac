# DexHand Tasks

This directory contains the implementation of the dexterous hand tasks using the new component-based architecture.

## Task Structure

The tasks follow a component-based architecture:

- `dex_hand_base.py`: Base class for all dexterous hand tasks
- `task_interface.py`: Interface that all task implementations must implement
- `base/vec_task.py`: Base class for vector tasks (inherited by DexHandBase)

## Key Features

### Auto-detection of Physics Steps Per Control Step

The environment now automatically detects how many physics steps are required between each control step based on reset stability requirements. This eliminates the need for manually configuring `controlFrequencyInv` and ensures that the environment is stable regardless of the physics setup.

Usage:
```python
# The environment will automatically detect the number of physics steps per control step
env = create_dex_env(...) 
```

### Configurable Action Space

The action space is now configurable, allowing you to specify which DOFs are controlled by the policy:

- `controlHandBase`: Whether the policy controls the hand base (6 DOFs)
- `controlFingers`: Whether the policy controls the finger joints (12 DOFs)

For DOFs not controlled by the policy, you can specify default targets or implement a custom control policy in the task.

Example configuration:
```yaml
env:
  controlHandBase: false      # Base not controlled by policy
  controlFingers: true        # Fingers controlled by policy
  defaultBaseTargets: [0.0, 0.0, 0.5, 0.0, 0.0, 0.0]  # Default targets for base
```

For task-specific control of uncontrolled DOFs, implement the `get_task_dof_targets` method in your task class:

```python
def get_task_dof_targets(self, num_envs, device, base_controlled, fingers_controlled):
    # Return targets for DOFs not controlled by the policy
    targets = {}
    
    if not base_controlled:
        # Example: Make the base follow a circular trajectory
        t = self.progress_buf.float() * 0.01
        base_targets = torch.zeros((num_envs, 6), device=device)
        base_targets[:, 0] = 0.3 * torch.sin(t)  # x position
        base_targets[:, 1] = 0.3 * torch.cos(t)  # y position
        base_targets[:, 2] = 0.5  # z position
        targets["base_targets"] = base_targets
    
    if not fingers_controlled and hasattr(self, 'object_pos'):
        # Example: Make fingers dynamically respond to object position
        finger_targets = self._compute_grasp_targets(self.object_pos)
        targets["finger_targets"] = finger_targets
    
    return targets
```

This allows for complex scenarios such as:
- **Dynamic trajectories**: The base or fingers can follow time-varying trajectories
- **State-dependent control**: Targets can depend on the state of the environment (e.g., object positions)
- **Task-phase control**: Different control strategies can be used during different phases of a task
- **Hybrid control**: Some DOFs can be controlled by the policy while others follow programmed behaviors

The task has complete control over what targets are returned, and can implement arbitrarily complex control laws. If the task returns `None` or omits a key from the targets dictionary, the environment will use the default targets specified in the configuration.