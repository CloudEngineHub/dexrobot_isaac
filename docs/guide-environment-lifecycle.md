# Environment Lifecycle Overview

This guide provides a high-level overview of the DexHand environment simulation loop, showing how the key components interact during training and execution.

## Simulation Loop Flow

The DexHand environment follows a standard RL environment pattern with specific component responsibilities:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Environment   │    │  RL Algorithm    │    │   Task Logic    │
│     Reset       │    │  (PPO/SAC/etc)   │    │ (Rewards/Done)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Simulation Step                             │
│                                                                 │
│  1. Action Processing  →  2. Physics Step  →  3. Observation   │
│     (ActionProcessor)     (PhysicsManager)     (ObservationEnc) │
│                                                                 │
│  4. Reward Computation  →  5. Termination  →  6. Reset Check   │
│     (Task + RewardCalc)     (TerminationMgr)  (ResetManager)   │
└─────────────────────────────────────────────────────────────────┘
```

## Detailed Step Breakdown

### 1. Action Processing
**Component**: `ActionProcessor`
**Purpose**: Convert policy actions to simulation targets

```python
# Entry point: env.step(actions)
def pre_physics_step(self, actions):
    # Convert normalized actions [-1,1] to physical targets
    self.action_processor.process_actions(actions)

    # Apply rule-based control overrides if configured
    self._apply_rule_based_control()

    # Send targets to Isaac Gym
    self.gym.set_dof_position_target_tensor(self.sim, self.dof_targets)
```

**Key Functions**:
- Action scaling based on control mode (`position`, `position_delta`, `velocity`, `effort`)
- DOF mapping (fingers, base, coupled joints)
- Rule-based control integration

**See**: [Action Processing Pipeline Guide](guide-action-pipeline.md)

### 2. Physics Simulation
**Component**: `PhysicsManager`
**Purpose**: Advance the physics simulation

```python
def step_physics(self):
    # Step Isaac Gym physics
    for _ in range(self.physics_steps_per_control_step):
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

    # Refresh tensor cache
    self.tensor_manager.refresh_tensors()
```

**Key Functions**:
- Multi-substep physics integration
- Tensor synchronization
- Contact force computation

**See**: [Physics Implementation Reference](reference-physics-implementation.md)

### 3. Observation Encoding
**Component**: `ObservationEncoder`
**Purpose**: Construct observation vector for policy

```python
def get_observations(self):
    # Collect base observations (hand state, DOF positions, etc.)
    obs = self.observation_encoder.encode_observations()

    # Add task-specific observations
    task_obs = self.task.compute_task_observations()
    obs.update(task_obs)

    return obs
```

**Key Functions**:
- Hand pose and DOF state encoding
- Contact force processing
- Velocity computation
- Task-specific observations

**See**: [Observation System Guide](guide-observation-system.md)

### 4. Reward Computation
**Components**: `Task` + `RewardCalculator`
**Purpose**: Compute scalar reward signal

```python
def compute_rewards(self):
    # Task computes raw reward terms
    task_rewards = self.task.compute_task_reward_terms()

    # RewardCalculator applies weights and aggregates
    total_reward = self.reward_calculator.compute_total_reward(task_rewards)

    return total_reward
```

**Key Functions**:
- Task-specific reward logic (distance, contact, success)
- Reward component weighting
- Reward aggregation and scaling

**See**: [Task Creation Guide](guide-task-creation.md)

### 5. Termination Detection
**Component**: `TerminationManager`
**Purpose**: Determine when episodes should end

```python
def check_terminations(self):
    # Check base termination conditions
    timeouts = self.progress_buf >= self.max_episode_length

    # Check task-specific terminations
    task_terms = self.task.compute_task_terminations()

    # Combine all termination sources
    done = timeouts | task_terms["success"] | task_terms["failure"]

    return done
```

**Key Functions**:
- Timeout detection
- Success criteria evaluation
- Failure condition checking

**See**: [Termination Logic Guide](guide-termination-logic.md)

### 6. Environment Reset
**Component**: `ResetManager`
**Purpose**: Reset terminated environments

```python
def reset_idx(self, env_ids):
    # Reset task-specific state
    self.task.reset_task_state(env_ids)

    # Reset hand to initial pose
    self.reset_manager.reset_hand_state(env_ids)

    # Apply physics reset
    self.gym.set_actor_root_state_tensor(self.sim, self.root_state_tensor)

    # Clear progress counters
    self.reset_buf[env_ids] = 0
    self.progress_buf[env_ids] = 0
```

**Key Functions**:
- Selective environment reset
- State randomization
- Physics state application

**See**: [Environment Resets Guide](guide-environment-resets.md)

## Component Interaction Patterns

### Data Flow Dependencies

```
TensorManager ──→ ObservationEncoder ──→ Observations
     │                                        │
     ▼                                        ▼
ActionProcessor ──→ Physics Step ──→ TensorManager.refresh()
     │                    │                   │
     ▼                    ▼                   ▼
Rule Controllers    Contact Forces    Reward Computation
```

### Timing Dependencies

1. **Initialization**: Components created in dependency order
2. **Control Cycle**: `control_dt` measured and distributed
3. **Simulation**: Fixed timestep execution
4. **Reset**: Coordinated state restoration

## Common Lifecycle Issues

### Action Not Applied
**Symptoms**: Robot doesn't respond to policy actions
**Debug Steps**:
1. Check action processing logs
2. Verify control mode configuration
3. Check DOF target tensor application

### Observations Invalid
**Symptoms**: NaN values or unexpected observation ranges
**Debug Steps**:
1. Check tensor refresh timing
2. Verify observation encoding logic
3. Validate task-specific observations

### Rewards Not Learning
**Symptoms**: Policy doesn't improve over time
**Debug Steps**:
1. Check reward component weights
2. Verify reward scaling
3. Examine task success criteria

### Reset Failures
**Symptoms**: Environments don't reset properly
**Debug Steps**:
1. Check reset buffer management
2. Verify state tensor application
3. Examine task reset logic

## Performance Characteristics

### Bottlenecks
- **Physics simulation**: Scales with environment count and physics steps
- **Tensor operations**: GPU memory bandwidth dependent
- **Observation encoding**: CPU-GPU transfer overhead

### Optimization Points
- **Vectorized operations**: Batch processing across environments
- **Tensor caching**: Minimize GPU memory allocations
- **Component efficiency**: Targeted optimization based on profiling

## Related Documentation

- **[Component Initialization](guide-component-initialization.md)** - How components are created and initialized
- **[Action Processing Pipeline](guide-action-pipeline.md)** - Detailed action processing flow
- **[Observation System](guide-observation-system.md)** - Observation encoding and management
- **[Environment Resets](guide-environment-resets.md)** - Reset logic and state management
- **[Termination Logic](guide-termination-logic.md)** - Episode termination criteria
- **[Physics Implementation](reference-physics-implementation.md)** - Low-level physics details

This lifecycle overview provides the mental model needed to understand how all DexHand components work together during training and execution.
