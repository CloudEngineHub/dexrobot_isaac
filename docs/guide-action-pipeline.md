# Action Processing Pipeline Guide

This guide explains the functional rule pipeline for action processing in DexRobot Isaac.

**Related Documentation:**
- [DOF and Action Control API](reference-dof-control-api.md) - For DOF mappings and coupling details
- [Component Initialization](guide-component-initialization.md) - For two-stage initialization pattern
- [Physics Implementation](reference-physics-implementation.md) - For control_dt measurement
- [Observation System](guide-observation-system.md) - For observation-dependent rules

## Overview

The action processing system transforms policy actions into robot DOF targets through a clean functional pipeline implemented in `ActionProcessor` (`dexhand_env/components/action_processor.py`):

```
active_prev_targets → pre_action_rule → active_rule_targets → action_rule → active_raw_targets → post_action_filters → active_next_targets → coupling_rule → full_dof_targets
```

## Data Flow

### 1. Pre-Action Rule
- **Input**: `active_prev_targets` (18D: 6 base + 12 finger), state dict with observations
- **Output**: `active_rule_targets` (18D)
- **Purpose**: Generate baseline targets before policy actions are applied
- **Implementation**: `ActionProcessor.apply_pre_action_rule()`
- **Examples**: Gravity compensation, default poses, trajectory following

### 2. Action Rule
- **Input**: `active_prev_targets`, `active_rule_targets`, policy `actions`, config dict
- **Output**: `active_raw_targets` (18D)
- **Purpose**: Apply policy actions according to control mode and masking
- **Implementation**: `ActionProcessor.apply_action_rule()`
- **Built-in modes**:
  - `position`: Scale actions to DOF limits with velocity clamping
  - `position_delta`: Add scaled actions to previous targets

### 3. Post-Action Filters
- **Input**: `active_prev_targets`, `active_rule_targets`, `active_raw_targets`
- **Output**: `active_next_targets` (18D)
- **Purpose**: Apply safety constraints and limits
- **Implementation**: `ActionProcessor.apply_post_action_filters()`
- **Built-in filters**:
  - `velocity_clamp`: Limit maximum velocity between timesteps
  - `position_clamp`: Clamp to DOF position limits

### 4. Coupling Rule
- **Input**: `active_next_targets` (18D)
- **Output**: `full_dof_targets` (26D)
- **Purpose**: Map active DOFs to full DOF space with physical coupling
- **Implementation**: `ActionProcessor.apply_coupling_rule()`
- **Example**: Finger joint coupling for underactuated joints

## Two-Stage Observation Initialization

The system uses a two-stage initialization process to resolve a circular dependency between observations and pre-action rules:

### The Problem
- Pre-action rules need observations to make decisions (e.g., gravity compensation based on hand orientation)
- The policy needs to observe the output of pre-action rules (`active_rule_targets`) to understand the baseline behavior
- This creates a circular dependency: observations → pre-action rule → active_rule_targets → observations

### The Solution
The observation system is updated in two stages during each timestep:

1. **Stage 1 - Partial Observations**:
   - Compute all observations EXCEPT `active_rule_targets`
   - This includes robot state, contact forces, previous actions, etc.
   - Pass these partial observations to the pre-action rule

2. **Stage 2 - Complete Observations**:
   - Apply the pre-action rule using partial observations
   - Add the resulting `active_rule_targets` to the observation dictionary
   - Concatenate all observations for the policy

### Implementation in DexHandBase
```python
# Stage 1: Compute partial observations (exclude active_rule_targets)
obs_dict = self.observation_encoder.compute_observations(
    exclude_components=['active_rule_targets']
)

# Stage 2: Apply pre-action rule with partial observations
state = {'obs_dict': obs_dict, 'env': self}
active_rule_targets = self.action_processor.apply_pre_action_rule(
    self.action_processor.active_prev_targets, state
)

# Stage 3: Update observation with rule targets
obs_dict['active_rule_targets'] = active_rule_targets
self.obs_buf = self.observation_encoder.concatenate_observations(obs_dict)
```

This design ensures that:
- Pre-action rules have access to all necessary state information
- The policy can observe and learn from the baseline behavior set by rules
- There are no circular dependencies in the computation graph

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

## Setting Up Action Rules

The action rule is required and must be set explicitly. There is no default action rule.

```python
# Define action rule for position control mode
def position_action_rule(active_prev_targets, active_rule_targets, actions, config):
    """Apply actions in position control mode."""
    # Start with rule targets to preserve uncontrolled DOFs
    targets = active_rule_targets.clone()

    # Get action processor reference
    ap = env.action_processor

    # Apply actions only to policy-controlled DOFs
    if config['policy_controls_base'] and config['policy_controls_fingers']:
        # Full control - scale actions to DOF limits
        scaled = ap._scale_actions_to_limits(actions)
        # Apply with velocity clamping...
    elif config['policy_controls_base']:
        # Base only - update first 6 DOFs
        targets[:, :6] = # ... scaled base actions
    elif config['policy_controls_fingers']:
        # Fingers only - update DOFs 6-18
        targets[:, 6:] = # ... scaled finger actions

    return targets

# Register the action rule (required!)
env.action_processor.set_action_rule(position_action_rule)
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
