# TerminationManager Component Guide

This guide explains the TerminationManager component, which handles episode termination decisions in DexHand environments.

## Overview

The TerminationManager is responsible for evaluating when episodes should end and providing the appropriate termination type and rewards. It replaces the previous SuccessFailureTracker with a cleaner architecture focused purely on termination decisions.

## Core Concept: Three Termination Types

The TerminationManager recognizes three mutually exclusive termination types:

### 1. Success Termination
- **When**: Task objectives are successfully completed
- **Reward**: Positive reward (configurable via `successReward`)
- **Examples**: Object successfully grasped, target position reached

### 2. Failure Termination
- **When**: Task constraints are violated or unrecoverable state reached
- **Reward**: Negative reward/penalty (configurable via `failurePenalty`)
- **Examples**: Hand drops below ground, object falls out of bounds

### 3. Timeout Termination
- **When**: Episode reaches maximum length without success or failure
- **Reward**: Neutral reward (configurable via `timeoutReward`, default: 0.0)
- **Purpose**: Prevents infinite episodes, enables RL training progress

## Key Methods

### `evaluate(episode_step_count, builtin_success, task_success, builtin_failure, task_failure)`

**Purpose**: Evaluate all termination conditions and determine which environments should reset

**Parameters**:
- `episode_step_count`: Current step count for each environment
- `builtin_success`: Dictionary of built-in success criteria (empty for BaseTask)
- `task_success`: Dictionary of task-specific success criteria
- `builtin_failure`: Dictionary of built-in failure criteria (empty for BaseTask)
- `task_failure`: Dictionary of task-specific failure criteria

**Returns**:
- `should_reset`: Boolean tensor indicating which environments should reset
- `termination_info`: Dictionary with detailed termination information
- `episode_rewards`: Dictionary with reward tensors for each termination type

### `reset_tracking(env_ids)`

**Purpose**: Reset termination tracking state for specified environments after they are reset

**Parameters**:
- `env_ids`: Environment indices that were reset

### `update_consecutive_successes(success_tensor)`

**Purpose**: Update consecutive success tracking for curriculum learning

**Parameters**:
- `success_tensor`: Boolean tensor indicating which environments had success

## Configuration

Configure termination behavior in your task's YAML file:

```yaml
# BaseTask.yaml
env:
  # Episode length before timeout
  episodeLength: 300

  # Termination rewards
  successReward: 10.0     # Positive reward for success
  failurePenalty: 5.0     # Penalty for failure (applied as negative)
  timeoutReward: 0.0      # Neutral reward for timeout

  # Active criteria (empty means use all available)
  activeSuccessCriteria: []   # Use all success criteria
  activeFailureCriteria: []   # Use all failure criteria

  # Curriculum learning
  maxConsecutiveSuccesses: 50
```

## Usage Example

### Basic Implementation in a Task

```python
class MyTask(DexTask):
    def check_task_success_criteria(self):
        """Define when the task is successfully completed."""
        return {
            "object_grasped": self._check_grasp(),
            "target_reached": self._check_target_distance(),
        }

    def check_task_failure_criteria(self):
        """Define when the task has failed unrecoverably."""
        return {
            "hand_too_low": self._check_hand_height(),
            "object_dropped": self._check_object_dropped(),
        }

    def _check_grasp(self):
        # Return boolean tensor of shape (num_envs,)
        contact_forces = self.parent_env.contact_forces
        return torch.norm(contact_forces, dim=-1) > self.grasp_threshold

    def _check_target_distance(self):
        # Return boolean tensor of shape (num_envs,)
        hand_pos = self.parent_env.rigid_body_states[:, hand_idx, :3]
        distance = torch.norm(hand_pos - self.target_pos, dim=-1)
        return distance < self.target_threshold
```

### Integration in BaseTask

The TerminationManager is automatically integrated in DexHandBase:

```python
# In DexHandBase.post_physics_step():

# 1. Get termination criteria from task
task_success = self.task.check_task_success_criteria() if hasattr(self.task, "check_task_success_criteria") else {}
task_failure = self.task.check_task_failure_criteria() if hasattr(self.task, "check_task_failure_criteria") else {}

# 2. Evaluate termination conditions
should_reset, termination_info, episode_rewards = self.termination_manager.evaluate(
    self.episode_step_count, {}, task_success, {}, task_failure
)

# 3. Apply termination rewards
for reward_type, reward_tensor in episode_rewards.items():
    self.rew_buf += reward_tensor

# 4. Reset environments and tracking
if torch.any(should_reset):
    env_ids_to_reset = torch.nonzero(should_reset).flatten()
    self.reset_manager.reset_idx(env_ids_to_reset)
    self.termination_manager.reset_tracking(env_ids_to_reset)
```

## Termination Information Output

The `termination_info` dictionary contains detailed information for logging and analysis:

```python
termination_info = {
    # Termination type indicators (mutually exclusive boolean tensors)
    "success": success_termination,      # Shape: (num_envs,)
    "failure": failure_termination,      # Shape: (num_envs,)
    "timeout": timeout_termination,      # Shape: (num_envs,)

    # Aggregated statistics (scalars)
    "success_rate": success_count / num_envs,
    "failure_rate": failure_count / num_envs,
    "timeout_rate": timeout_count / num_envs,

    # Specific reason tracking (per criterion)
    "success_reason_object_grasped": grasp_success_mask,
    "failure_reason_hand_too_low": hand_low_failure_mask,
    # ... (one for each active criterion)
}
```

## Advanced Features

### Filtering Active Criteria

Control which criteria are evaluated:

```yaml
env:
  activeSuccessCriteria: ["object_grasped"]  # Only check grasping
  activeFailureCriteria: ["hand_too_low"]    # Only check hand height
```

Empty lists mean use all available criteria.

### Consecutive Success Tracking

The TerminationManager tracks consecutive successes for curriculum learning:

```python
# Check current consecutive success count
consecutive = self.termination_manager.consecutive_successes

# Use for curriculum progression
if consecutive > self.curriculum_threshold:
    self.increase_difficulty()
```

### Reason-Specific Tracking

Track which specific criteria triggered termination:

```python
# Check if any environment succeeded due to grasping
if torch.any(termination_info["success_reason_object_grasped"]):
    logger.info("Grasping success achieved!")

# Check timeout rate for training monitoring
if termination_info["timeout_rate"] > 0.8:
    logger.warning("High timeout rate - consider adjusting task difficulty")
```

## Benefits Over Previous SuccessFailureTracker

1. **Clear Separation**: Only handles termination decisions, not reset execution
2. **Three-Type Model**: Explicit success/failure/timeout classification
3. **Better RL Integration**: Proper episode completion signals for rl_games
4. **Extensible**: Easy to add new termination types or criteria
5. **No Duplication**: Single timeout check (was duplicated in old architecture)

## Common Patterns

### Simple Task (Timeout Only)
```python
# For basic tasks, just rely on timeout termination
# No need to implement success/failure criteria methods
# Episodes will terminate after episodeLength steps
```

### Success-Only Task
```python
def check_task_success_criteria(self):
    return {"goal_reached": self._check_goal()}

# No failure criteria - only success or timeout
```

### Complex Multi-Objective Task
```python
def check_task_success_criteria(self):
    return {
        "primary_objective": self._check_primary_goal(),
        "secondary_objective": self._check_secondary_goal(),
        "bonus_objective": self._check_bonus_goal(),
    }

def check_task_failure_criteria(self):
    return {
        "safety_violation": self._check_safety_limits(),
        "workspace_exit": self._check_workspace_bounds(),
        "collision": self._check_collisions(),
    }
```

## Debugging Tips

### Log Termination Rates
```python
# Add to your training logs
logger.info(f"Success: {termination_info['success_rate']:.2%}, "
           f"Failure: {termination_info['failure_rate']:.2%}, "
           f"Timeout: {termination_info['timeout_rate']:.2%}")
```

### Visualize Termination Reasons
Use the detailed reason tracking to understand why episodes are ending:

```python
for key, value in termination_info.items():
    if key.startswith("success_reason_") and torch.any(value):
        reason = key.replace("success_reason_", "")
        count = value.sum().item()
        logger.info(f"Success reason '{reason}': {count} environments")
```

## References
- Implementation: `dexhand_env/components/termination_manager.py`
- Integration: `dexhand_env/tasks/dexhand_base.py`
- Task interface: `dexhand_env/tasks/task_interface.py`
- Reset system: `docs/guide-environment-resets.md`
