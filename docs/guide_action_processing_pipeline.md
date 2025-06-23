# Action Processing Pipeline Guide

This guide explains the functional rule pipeline for action processing in DexRobot Isaac.

## Overview

The action processing system transforms policy actions into robot DOF targets through a clean functional pipeline:

```
active_prev_targets → pre_action_rule → active_rule_targets → action_rule → active_raw_targets → post_action_filters → active_next_targets → coupling_rule → full_dof_targets
```

## Data Flow

### 1. Pre-Action Rule
- **Input**: `active_prev_targets` (18D: 6 base + 12 finger), state dict with observations
- **Output**: `active_rule_targets` (18D)
- **Purpose**: Generate baseline targets before policy actions are applied
- **Examples**: Gravity compensation, default poses, trajectory following

### 2. Action Rule
- **Input**: `active_prev_targets`, `active_rule_targets`, policy `actions`, config dict
- **Output**: `active_raw_targets` (18D)
- **Purpose**: Apply policy actions according to control mode and masking
- **Built-in modes**:
  - `position`: Scale actions to DOF limits with velocity clamping
  - `position_delta`: Add scaled actions to previous targets

### 3. Post-Action Filters
- **Input**: `active_prev_targets`, `active_rule_targets`, `active_raw_targets`
- **Output**: `active_next_targets` (18D)
- **Purpose**: Apply safety constraints and limits
- **Built-in filters**:
  - `velocity_clamp`: Limit maximum velocity between timesteps
  - `position_clamp`: Clamp to DOF position limits

### 4. Coupling Rule
- **Input**: `active_next_targets` (18D)
- **Output**: `full_dof_targets` (26D)
- **Purpose**: Map active DOFs to full DOF space with physical coupling
- **Example**: Finger joint coupling for underactuated joints

## Integration with Observations

The pipeline integrates with the observation system in two stages:

1. **Before pre-action rule**: Compute observations excluding `active_rule_targets`
2. **After pre-action rule**: Add `active_rule_targets` to observations for policy

This allows pre-action rules to use observations while providing rule targets to the policy.

## Usage Example

```python
# Define a pre-action rule for gravity compensation
def gravity_compensation_rule(active_prev_targets, state):
    """Apply gravity compensation based on hand orientation."""
    env = state['env']
    obs_dict = state['obs_dict']

    targets = active_prev_targets.clone()

    # Use hand orientation from observations
    if 'hand_pose_quat' in obs_dict:
        hand_quat = obs_dict['hand_pose_quat']
        # Apply compensation logic...

    return targets

# Register the rule
env.action_processor.set_pre_action_rule(gravity_compensation_rule)

# Define a custom post-action filter
def workspace_limit_filter(prev, rule, raw):
    """Limit hand position to workspace bounds."""
    filtered = raw.clone()
    # Apply workspace constraints to base DOFs...
    return filtered

# Register the filter
env.action_processor.register_post_action_filter("workspace_limit", workspace_limit_filter)

# Enable via config:
# postActionFilters: ["velocity_clamp", "position_clamp", "workspace_limit"]
```

## Migration from Rule-Based Control

The old `set_rule_based_controllers` method is replaced by pre-action rules:

**Old approach:**
```python
env.set_rule_based_controllers(
    base_controller=base_controller,
    finger_controller=finger_controller
)
```

**New approach:**
```python
def rule_based_pre_action(active_prev_targets, state):
    env = state['env']
    targets = active_prev_targets.clone()

    if base_controller and not env.policy_controls_hand_base:
        targets[:, :6] = base_controller(env)

    if finger_controller and not env.policy_controls_fingers:
        targets[:, 6:] = finger_controller(env)

    return targets

env.action_processor.set_pre_action_rule(rule_based_pre_action)
```

## Key Benefits

1. **Composability**: Each rule is a pure function that can be tested independently
2. **Flexibility**: Easy to add new rules without modifying core code
3. **Clarity**: Clear data flow with explicit naming (active vs full DOF)
4. **Safety**: Post-action filters guarantee constraints are enforced
5. **Extensibility**: Support for custom rules and filters

## Implementation Details

### Pure Functions
All rules use pure functions (no side effects) for better composability:
- Input tensors are not modified
- Output is always a new tensor
- No hidden state changes

### Dimension Consistency
- Rules operate on 18D active DOF space (6 base + 12 finger)
- Coupling maps to 26D full DOF space
- Clear naming distinguishes active vs full DOF tensors

### Configuration
Post-action filters are configured via YAML:
```yaml
postActionFilters: ["velocity_clamp", "position_clamp"]
```

Existing control mode configuration is preserved:
```yaml
controlMode: "position"  # or "position_delta"
policyControlsHandBase: false
policyControlsFingers: true
```

### Post-Action Filter Registry

The system uses a registry pattern for post-action filters:

1. **Built-in filters** are automatically registered:
   - `velocity_clamp`: Enforces velocity limits
   - `position_clamp`: Enforces position limits

2. **Custom filters** can be registered:
   ```python
   # Register a custom filter
   env.action_processor.register_post_action_filter("my_filter", my_filter_fn)
   ```

3. **Enabling filters** is controlled by configuration:
   - Via YAML: `postActionFilters: ["velocity_clamp", "my_filter"]`
   - Only filters listed in config are applied

4. **Filter order** matters - filters are applied in the order listed in configuration

This registry pattern separates filter implementation from configuration, making it easy to add new filters without modifying core code.
