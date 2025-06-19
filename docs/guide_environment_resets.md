# Environment Reset System Guide

This guide explains the environment reset system in DexHand Isaac environments.

## Overview

The reset system follows Isaac Gym's standard pattern for managing episode termination and environment resets in vectorized environments.

## Key Components

### reset_buf (Reset Buffer)
- **Type**: `torch.Tensor` of shape `(num_envs,)` with dtype `torch.long`
- **Purpose**: Flags which environments need to be reset (1 = needs reset, 0 = no reset needed)
- **Standard Isaac Gym Pattern**: Used across all IsaacGymEnvs tasks

### episode_step_count (Progress Buffer)
- **Type**: `torch.Tensor` of shape `(num_envs,)`
- **Purpose**: Tracks the number of steps in each environment's current episode
- **Reset**: Automatically reset to 0 when environment resets

## Reset Flow

1. **Termination Check** (`check_termination()`):
   - Sets `reset_buf[env_id] = 1` when `episode_step_count >= max_episode_length - 1`
   - Can also include task-specific termination conditions

2. **Reset Execution** (`reset_idx(env_ids)`):
   - Resets DOF states and positions for specified environments
   - Resets episode step counter: `episode_step_count[env_ids] = 0`
   - **CRITICAL**: Must clear reset buffer: `reset_buf[env_ids] = 0`

3. **Post-Reset**:
   - Environment continues normally until next termination condition

## Common Pitfall

**Forgetting to clear reset_buf** causes continuous resets every step:
```python
# WRONG - Missing reset_buf clearing
def reset_idx(self, env_ids):
    self.episode_step_count[env_ids] = 0
    # Reset DOF states...

# CORRECT - Following Isaac Gym standard
def reset_idx(self, env_ids):
    self.episode_step_count[env_ids] = 0
    # Reset DOF states...
    self.reset_buf[env_ids] = 0  # CRITICAL!
```

## Example Usage

```python
# In post_physics_step():
# Check for terminations
self.reset_buf = self.reset_manager.check_termination()

# Reset environments that need it
if torch.any(self.reset_buf):
    env_ids = torch.nonzero(self.reset_buf).flatten()
    self.reset_idx(env_ids)
    # reset_buf is cleared inside reset_idx
```

## Testing Episode Length

You can test episode length behavior with:
```bash
python examples/dexhand_test.py --episode-length 10
```

This should reset the environment every 10 steps, visible in the logs as:
```
Environment reset detected at step 10
Environment reset detected at step 20
Environment reset detected at step 30
...
```

## References
- Isaac Gym's VecTask base class: `isaacgymenvs/tasks/base/vec_task.py`
- Example implementation: `isaacgymenvs/tasks/cartpole.py`
- DexHand reset implementation: `dex_hand_env/components/reset_manager.py`
