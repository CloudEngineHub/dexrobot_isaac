# Task Creation Guide

This guide walks through creating a new task for the DexHand environment.

**Related Documentation:**
- [System Architecture](ARCHITECTURE.md) - For overall design and component relationships
- [Terminology Glossary](GLOSSARY.md) - For key concepts and definitions
- [Component Initialization](guide-component-initialization.md) - For two-stage initialization details
- [Observation System](guide-observation-system.md) - For adding task-specific observations
- [TRAINING.md](TRAINING.md) - For using your task in training workflows

## Overview

Tasks in this environment define:
- Scene setup (objects, spawn positions)
- Reward functions
- Termination conditions
- Reset behavior
- Task-specific observations

## Two-Stage Initialization Pattern

**CRITICAL:** Tasks must follow the two-stage initialization pattern used throughout the DexHand environment.

**→ See [Component Initialization Guide](guide-component-initialization.md) for the complete explanation of this pattern.**

### Quick Summary for Task Implementation

**Stage 1: Construction** - Store configuration, create data structures
```python
def __init__(self, cfg, *args, **kwargs):
    # Store task-specific config (no dependencies on other components)
    self.target_position = torch.tensor(cfg["task"]["target_position"], dtype=torch.float32)
    self.success_threshold = cfg["task"]["success_threshold"]
    super().__init__(cfg, *args, **kwargs)
```

**Stage 2: Finalization** - Complete setup after `control_dt` is available
```python
def finalize_setup(self):
    """Complete initialization after all components are ready."""
    super().finalize_setup()
    # Now safe to access control_dt and other component dependencies
    # Setup any velocity-based computations that need control_dt here
```

## Step-by-Step Task Creation

### 1. Create Task Configuration

Create a new YAML file in `dexhand_env/cfg/task/YourTask.yaml`:

```yaml
# YourTask Configuration
name: "YourTask"

# Inherit from BaseTask for common settings
defaults:
  - BaseTask

# Task-specific environment settings
env:
  episodeLength: 1000

  # Custom spawn positions
  initialHandPos: [0.0, 0.0, 0.2]
  initialHandRot: [0.0, 0.0, 0.0]

# Task-specific parameters
task:
  # Your task parameters here
  target_position: [0.1, 0.1, 0.1]
  success_threshold: 0.05

# Reward configuration
reward:
  # Define reward components and weights
  reach_target: 1.0
  distance_penalty: -0.1
  success_bonus: 10.0

  # Override base reward weights if needed
  alive: 1.0
  height_safety: -10.0

# Observations (in addition to base observations)
observations:
  policy:
    - target_position
    - distance_to_target
```

### 2. Create Task Implementation

Create `dexhand_env/tasks/your_task.py`:

```python
import torch
from dexhand_env.tasks.base_task import BaseTask

class YourTask(BaseTask):
    """Custom task implementation."""

    def __init__(self, cfg, *args, **kwargs):
        # Stage 1: Store task-specific config (no dependencies on other components)
        self.target_position = torch.tensor(cfg["task"]["target_position"], dtype=torch.float32)
        self.success_threshold = cfg["task"]["success_threshold"]

        super().__init__(cfg, *args, **kwargs)

    def finalize_setup(self):
        """Stage 2: Complete initialization after all components are ready."""
        super().finalize_setup()
        # Now safe to access control_dt and other component dependencies
        # Setup any velocity-based computations that need control_dt here

    def create_task_objects(self, gym, sim, env_ptr, env_id: int):
        """Create task-specific objects in the environment."""
        # Add objects like target spheres, boxes, etc.
        # Example: create target visualization
        pass

    def reset_task_state(self, env_ids):
        """Reset task-specific state for given environments."""
        # Reset object positions, randomize targets, etc.
        pass

    def compute_task_observations(self):
        """Compute task-specific observations."""
        obs = {}

        # Calculate distance to target
        hand_pos = self.rigid_body_states[:, self.hand_rigid_body_index, :3]
        target_pos = self.target_position.to(self.device)
        distance = torch.norm(hand_pos - target_pos, dim=1, keepdim=True)

        obs["distance_to_target"] = distance
        obs["target_position"] = target_pos.repeat(self.num_envs, 1)

        return obs

    def compute_task_reward_terms(self):
        """Compute task-specific reward components."""
        rewards = {}

        # Get hand position
        hand_pos = self.rigid_body_states[:, self.hand_rigid_body_index, :3]
        target_pos = self.target_position.to(self.device)

        # Distance-based reward
        distance = torch.norm(hand_pos - target_pos, dim=1)
        rewards["reach_target"] = torch.exp(-distance * 5.0)  # Exponential reward
        rewards["distance_penalty"] = -distance

        # Success bonus
        success = distance < self.success_threshold
        rewards["success_bonus"] = success.float()

        return rewards

    def compute_task_terminations(self):
        """Compute task-specific termination conditions."""
        # Return dict with termination conditions
        # Example: terminate on success
        hand_pos = self.rigid_body_states[:, self.hand_rigid_body_index, :3]
        target_pos = self.target_position.to(self.device)
        distance = torch.norm(hand_pos - target_pos, dim=1)

        return {
            "success": distance < self.success_threshold
        }
```

### 3. Register Task

Add your task to the factory in `dexhand_env/factory.py`:

```python
# In the task creation section
elif task_name == "YourTask":
    from dexhand_env.tasks.your_task import YourTask
    task = YourTask(cfg=task_cfg, **task_kwargs)
```

### 4. Create Training Configuration (Optional)

Create `dexhand_env/cfg/train/YourTaskPPO.yaml` if you need task-specific training parameters:

```yaml
# Inherit from base training config
defaults:
  - BaseTaskPPO

# Override specific parameters for your task
params:
  config:
    name: "YourTask"
    # Task-specific training parameters
    learning_rate: 5e-4
    horizon_length: 32
    # etc.
```

### 5. Test Your Task

```bash
# Test script only supports BaseTask - use training script for testing custom tasks
python train.py task=YourTask steps=100 headless=false

# Train your task
python train.py task=YourTask

# Test with custom training config
python train.py task=YourTask train=YourTaskPPO
```

## Complete Example: BlindGrasping Task

The BlindGrasping task provides a real-world example of all the concepts above. Here are the key implementation patterns you can adapt for your own tasks.

### Task Configuration Example

**File: `dexhand_env/cfg/task/BlindGrasping.yaml`**
```yaml
# @package _global_
defaults:
  - BaseTask
  - _self_

task:
  name: BlindGrasping

env:
  episodeLength: 300  # 10 seconds at 30 Hz

# Reward configuration - critical for learning
reward:
  # Base rewards (from BaseTask)
  alive: 1.0
  height_safety: -10.0
  finger_velocity: -0.01

  # Task-specific rewards
  object_height: 10.0           # Reward for lifting object
  grasp_approach: 2.0           # Reward for getting close to object
  contact_duration: 5.0         # Reward for maintaining contact
  grasp_success: 100.0          # Large bonus for successful grasp

  # Penalties
  object_fall: -50.0            # Penalty for dropping object
  finger_ground_contact: -20.0  # Penalty for finger touching ground
```

### Core Task Implementation Patterns

**Key patterns from `dexhand_env/tasks/box_grasping_task.py`:**

```python
class BlindGraspingTask(DexTask):
    """Box grasping with tactile-only feedback (no vision)."""

    def __init__(self, sim, gym, device, num_envs, cfg):
        # Always call parent constructor first
        super().__init__(sim, gym, device, num_envs, cfg)

        # Task-specific parameters from config
        self.box_size = 0.05  # 5cm cube
        self.success_height = 0.20  # 20cm lift requirement
        self.min_contact_duration = 2.0  # seconds

        # Task state tensors (one per environment)
        self.object_heights = torch.zeros(num_envs, device=device)
        self.contact_durations = torch.zeros(num_envs, device=device)
        self.grasp_success = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def setup_task(self) -> None:
        """Create box objects in each environment."""
        # Create box asset
        box_options = gymapi.AssetOptions()
        box_options.density = 400.0  # kg/m³ (wood-like)
        box_asset = self.gym.create_box(self.sim, self.box_size, self.box_size, self.box_size, box_options)

        # Spawn boxes in each environment
        for i in range(self.num_envs):
            box_handle = self.gym.create_actor(
                self.envs[i], box_asset,
                gymapi.Transform(p=gymapi.Vec3(0.0, 0.0, 0.1)),  # 10cm above table
                f"box_{i}", i, 0
            )
            # Store actor indices for later reference
            self.box_actor_indices[i] = self.gym.get_actor_index(self.envs[i], box_handle, gymapi.DOMAIN_SIM)

    def compute_task_observations(self) -> Dict[str, torch.Tensor]:
        """Compute tactile-only observations (no vision)."""
        # Get contact forces between fingers and objects
        contact_forces = self.rigid_body_contact_forces[self.fingertip_indices]

        # Binary contact detection (touch/no-touch)
        finger_contacts = (torch.norm(contact_forces, dim=-1) > 0.1).float()

        # Contact duration tracking
        self.contact_durations += (torch.sum(finger_contacts, dim=1) >= 2) * self.control_dt

        return {
            "finger_contacts": finger_contacts,           # [num_envs, 5] - one per fingertip
            "contact_duration": self.contact_durations,   # [num_envs] - seconds of contact
            "object_relative_height": self.object_heights - 0.1,  # [num_envs] - height above spawn
        }

    def compute_task_reward_terms(self) -> Dict[str, torch.Tensor]:
        """Compute raw reward terms (no weighting)."""
        # Object height reward (encourage lifting)
        height_reward = torch.clamp(self.object_heights - 0.1, 0.0, 0.2) / 0.2

        # Contact duration reward (encourage sustained grasp)
        duration_reward = torch.clamp(self.contact_durations / self.min_contact_duration, 0.0, 1.0)

        # Success bonus (binary: 0 or 1)
        success_reward = self.grasp_success.float()

        return {
            "object_height": height_reward,
            "contact_duration": duration_reward,
            "grasp_success": success_reward,
        }

    def compute_task_terminations(self) -> Dict[str, torch.Tensor]:
        """Determine when episodes should end."""
        # Success: object lifted high enough with sustained contact
        success = (self.object_heights > self.success_height) & (self.contact_durations > self.min_contact_duration)

        # Failure: object falls too low
        failure = self.object_heights < 0.05  # 5cm below table

        return {
            "success": success,
            "failure": failure,
        }

    def reset_task_state(self, env_ids: torch.Tensor) -> None:
        """Reset task-specific state for given environments."""
        # Reset object positions to random spawn locations
        reset_positions = self.initial_box_positions[env_ids] + torch.randn_like(self.initial_box_positions[env_ids]) * 0.02

        # Apply reset to simulation
        self.root_state_tensor[self.box_actor_indices[env_ids], :3] = reset_positions
        self.root_state_tensor[self.box_actor_indices[env_ids], 3:7] = torch.tensor([0, 0, 0, 1])  # identity quaternion
        self.root_state_tensor[self.box_actor_indices[env_ids], 7:] = 0  # zero velocities

        # Reset task state tracking
        self.object_heights[env_ids] = 0.1  # 10cm (initial spawn height)
        self.contact_durations[env_ids] = 0.0
        self.grasp_success[env_ids] = False
```

### Key Implementation Insights

**1. Reward Design Philosophy:**
```python
# ✅ CORRECT: Raw reward terms in task, weighting in config
def compute_task_reward_terms(self):
    return {
        "object_height": height_above_table,  # Raw value 0.0-0.2
        "contact_duration": normalized_duration,  # Raw value 0.0-1.0
    }

# ❌ WRONG: Don't apply weights in task implementation
def compute_task_reward_terms(self):
    return {
        "object_height": height_above_table * 10.0,  # No hardcoded scaling!
    }
```

**2. State Management:**
```python
# ✅ CORRECT: Always reset ALL task state
def reset_task_state(self, env_ids):
    self.object_heights[env_ids] = initial_value
    self.contact_durations[env_ids] = 0.0
    self.any_other_state[env_ids] = default_value

# ❌ WRONG: Forgetting to reset state leads to bugs
def reset_task_state(self, env_ids):
    self.object_heights[env_ids] = initial_value
    # Missing: self.contact_durations[env_ids] = 0.0  # Bug!
```

**3. Observation Consistency:**
```python
# ✅ CORRECT: Always return same observation structure
def compute_task_observations(self):
    return {
        "finger_contacts": contacts,      # Always [num_envs, 5]
        "contact_duration": durations,    # Always [num_envs]
    }

# ❌ WRONG: Conditional observations break training
def compute_task_observations(self):
    obs = {"finger_contacts": contacts}
    if self.include_duration:  # Don't do this!
        obs["contact_duration"] = durations
    return obs
```

## Task Implementation Checklist

### Required Task Methods
```python
# These methods MUST be implemented in your task:
def initialize_task_buffers(self):
    """Create task-specific tensors"""
    self.target_positions = torch.zeros((self.num_envs, 3), device=self.device)

def compute_task_reward_terms(self):
    """Return dict of raw reward values (no weights)"""
    return {"reach_distance": -torch.norm(hand_pos - target_pos, dim=-1)}

def compute_task_terminations(self):
    """Return success/failure dictionaries"""
    return {"target_reached": distance < 0.05}, {"timeout": False}

def reset_task_state(self, env_indices):
    """Reset task-specific state for given environments"""
    self.target_positions[env_indices] = torch.rand(...) * 0.2 - 0.1
```

### Repository-Specific Patterns

**Reward Terms**: Always return raw values in `compute_task_reward_terms()`:
```python
# ✅ CORRECT - raw distance value
return {"reach_distance": -distance}

# ❌ WRONG - never apply weights in task
return {"reach_distance": -distance * 5.0}  # Weight applied in wrong place!
```

**Tensor Operations**: Use batch operations for all environments:
```python
# ✅ CORRECT - vectorized for all environments
distances = torch.norm(self.hand_positions - self.targets, dim=-1)

# ❌ WRONG - loop over environments
for i in range(self.num_envs):
    distances[i] = np.linalg.norm(...)  # CPU operation!
```

**Debug with Repository Tools**:
```bash
# Test task with visualization
python train.py task=YourTask test=true viewer=true numEnvs=1

# Log reward components
python train.py task=YourTask train.logging.logLevel=DEBUG

# Validate observations
python examples/dexhand_test.py task=YourTask --episode-length 10
```

## Common Patterns

### Distance-Based Rewards

```python
distance = torch.norm(hand_pos - target_pos, dim=1)
reward = torch.exp(-distance * scale_factor)
```

### Success Detection

```python
success = distance < threshold
reward = success.float() * bonus_value
```

### Contact Rewards

```python
contact_force = self.contact_forces[:, finger_indices].sum(dim=-1)
contact_reward = (contact_force > contact_threshold).float()
```

### Multi-Stage Tasks

```python
# Track task stage per environment
stage_1_complete = distance_to_object < 0.05
stage_2_complete = stage_1_complete & (grasp_force > min_force)
# Provide stage-specific rewards
```

## Component Interaction Patterns

Understanding how tasks interact with the broader system:

```
[ DexHandBase ]
    |
    +-- owns --> [ PhysicsManager ]
    +-- owns --> [ TensorManager ]
    +-- owns --> [ ActionProcessor ]
    +-- owns --> [ ObservationEncoder ]
    +-- owns --> [ RewardCalculator ]
    +-- owns --> [ TerminationManager ]
    +-- owns --> [ ResetManager ]
    +-- owns --> [ ViewerController ]
    |
    +-- owns --> [ YourTask ]
                    |
                    +-- implements --> compute_task_observations()
                    +-- implements --> compute_task_reward_terms()
                    +-- implements --> compute_task_terminations()
                    +-- implements --> reset_task_state()
```

**Data Flow During Simulation Step:**
1. `DexHandBase.pre_physics_step()` - Apply actions via ActionProcessor
2. `PhysicsManager.step()` - Advance physics simulation
3. `TensorManager.refresh_tensors()` - Update simulation state tensors
4. `DexHandBase.post_physics_step()` - Compute observations, rewards, terminations
5. `YourTask.compute_*()` methods called during post_physics_step

## Testing and Debugging

### Integration Testing
Create a standalone test script to verify your task works correctly:

```python
# test_your_task.py
from dexhand_env.factory import create_env

def test_your_task():
    cfg = {"task": "YourTask", "env": {"numEnvs": 4, "episodeLength": 100}}
    env = create_env(cfg)

    # Run for several episodes to check for crashes
    for episode in range(5):
        obs = env.reset()
        for step in range(100):
            action = env.action_space.sample()  # Random actions
            obs, reward, done, info = env.step(action)
            if done.any():
                print(f"Episode {episode}, Step {step}: Some envs terminated")
                break
    print("Task integration test passed!")

if __name__ == "__main__":
    test_your_task()
```

### Debugging Tips

1. **Use `pdb` for interactive debugging:**
   ```python
   import pdb; pdb.set_trace()
   ```
   Launch with `python examples/dexhand_test.py` and the debugger will trigger in your terminal.

2. **Visual debugging with Isaac Gym:**
   - Enable contact force visualization in ViewerController
   - Use rigid body visualization to check object states
   - Add debug prints to track tensor shapes and values

3. **Tensor debugging:**
   ```python
   # Add shape assertions for new tensors
   assert hand_pos.shape == (self.num_envs, 3), f"Expected shape {(self.num_envs, 3)}, got {hand_pos.shape}"

   # Check for NaN values
   assert not torch.isnan(rewards["my_reward"]).any(), "NaN values in reward computation"
   ```

4. **Start small:**
   - Test with `num_envs=1` first
   - Use simple reward functions initially
   - Add complexity incrementally

This guide should get you started with creating custom tasks. Refer to `BlindGraspingTask` for a complete example implementation.

## See Also

- **[System Architecture](ARCHITECTURE.md)** - Component interaction patterns and design principles
- **[Component Debugging](guide-debugging.md)** - Troubleshooting component interaction issues
- **[Coordinate Systems](reference-coordinate-systems.md)** - Understanding fixed base motion and coordinate quirks
- **[TRAINING.md](TRAINING.md)** - Training your custom task with RL algorithms
- **[Terminology Glossary](GLOSSARY.md)** - Definitions of concepts used in this guide
- **[Component Initialization](guide-component-initialization.md)** - Detailed two-stage initialization documentation
