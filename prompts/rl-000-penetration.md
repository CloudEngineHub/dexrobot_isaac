# RL-000: Prevent Policy from Exploiting Penetration

## Problem Statement and Context

**Critical Issue**: The current reward system in the DexHand box grasping task allows policies to exploit physics simulation artifacts by achieving rewards through object interpenetration rather than learning proper manipulation skills. This fundamentally undermines the training objective and prevents the development of realistic grasping behaviors.

**Why This Matters**:
- Policies that exploit penetration cannot transfer to real-world robotics applications
- Penetration-based strategies bypass the fundamental challenges of dexterous manipulation
- Training rewards become misleading indicators of actual manipulation competency
- The learned behaviors violate basic physical constraints that real robots must respect

**Training Impact**: This issue blocks effective RL training because:
1. Policies receive high rewards for unrealistic behaviors
2. Legitimate contact strategies are disadvantaged compared to penetration exploits
3. Training convergence leads to physically impossible manipulation strategies
4. Policy evaluation metrics become meaningless for real-world deployment

## Analysis of Current Reward System

I've analyzed the `/home/yiwen/dexrobot_isaac/dexhand_env/tasks/box_grasping_task.py` file and identified how rewards are calculated and where potential penetration exploits could occur.

## Main Reward/Penalty Methods Found

### 1. compute_task_reward_terms() (Lines 970-1040)
- Primary reward computation method with stage-based reward system
- Delegates to stage-specific reward methods based on current stage (1, 2, or 3)
- Includes stage completion bonuses for successful transitions

### 2. Stage-Specific Reward Methods

**_compute_stage1_rewards() (Lines 1042-1099)**
- Pre-grasp positioning rewards
- No penetration penalties identified

**_compute_stage2_rewards() (Lines 1101-1132)**
- Contact establishment rewards
- **CRITICAL EXPLOIT RISK**: s2_fingerpad_proximity reward uses exponential decay based on distance to object
- Could incentivize penetration to get closer to object center

**_compute_stage3_rewards() (Lines 1134-1158)**
- Lifting rewards including s3_object_height and s3_grasp_maintenance
- No direct penetration penalties

### 3. Contact Detection Logic

**_detect_finger_box_contacts() (Lines 763-823)**
- Uses heuristic combining finger contact, box contact, and proximity
- **POTENTIAL EXPLOIT**: Proximity threshold is sqrt(3) * box_size/2 * 1.2
- Could allow registering "contact" while partially penetrating the box

## Identified Penetration Exploit Risks

### Critical Risk: Stage 2 Proximity Rewards
- **Location**: s2_fingerpad_proximity reward (Line 1119)
- **Issue**: Exponential reward for getting closer to object center
- **Weight**: 0.5 in config (Line 135)
- **Exploit**: Policy could learn to penetrate box to minimize distance to center

### Moderate Risk: Contact Detection Heuristic
- **Location**: _detect_finger_box_contacts() proximity check (Lines 807-810)
- **Issue**: 20% margin on proximity threshold could allow penetration
- **Impact**: Could register successful "grasps" while fingers are inside object

### No Penetration Penalties Found
- **Issue**: No explicit penalties for excessive contact forces or penetration
- **Impact**: Physics violations go unpunished, potentially rewarded

## Approved Implementation Plan

### Primary Solution: Tensor-Based Penetration Detection

Use Isaac Gym's tensor API for maximum performance with dual penetration indicators:

**1. Excessive Force Magnitude Detection**
```python
def detect_excessive_forces(self):
    """Detect penetration via excessive contact forces using tensor API."""
    contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
    contact_force_tensor = gymtorch.wrap_tensor(contact_forces)

    # Extract forces for hand bodies (fingertips)
    hand_forces = contact_force_tensor[self.hand_body_indices]
    hand_force_norms = torch.norm(hand_forces, dim=1)

    # Configurable excessive force threshold
    excessive_forces = hand_force_norms > self.penetration_force_threshold
    return excessive_forces.any()
```

**2. Geometric Penetration Detection**
```python
def detect_geometric_penetration(self, obs_dict):
    """Detect fingertip penetration inside box volume."""
    fingertip_positions = obs_dict["fingertip_poses_world"].view(-1, 5, 7)[:, :, :3]
    box_center = self.box_center_positions  # Shape: (num_envs, 3)
    half_box_size = self.box_size / 2.0     # Shape: (3,)

    # Calculate distance from each fingertip to box center
    distances = torch.norm(fingertip_positions - box_center.unsqueeze(1), dim=-1)
    min_distances = distances.min(dim=1)[0]  # Closest fingertip per env

    # Configurable geometric penetration threshold
    penetration_threshold = half_box_size.min() * self.geometric_penetration_factor
    penetration_detected = min_distances < penetration_threshold
    return penetration_detected, min_distances
```

### Reward System Modifications

**1. Add Penetration Penalty Component**
```python
def compute_penetration_penalty(self, obs_dict):
    """Strong penalty for penetration via forces or geometry."""
    force_penetration = self.detect_excessive_forces()
    geometric_penetration, _ = self.detect_geometric_penetration(obs_dict)

    penetration_detected = force_penetration | geometric_penetration
    penalty = torch.where(penetration_detected,
                         self.penetration_penalty_magnitude,  # Configurable penalty
                         torch.zeros_like(penetration_detected, dtype=torch.float))
    return penalty
```

**2. Modify Proximity Reward Logic**
```python
def compute_proximity_reward_safe(self, obs_dict):
    """Proximity reward with penetration protection."""
    _, min_distances = self.detect_geometric_penetration(obs_dict)
    half_box_width = self.box_size.min() / 2.0

    # Configurable minimum distance clamp
    min_reward_distance = half_box_width * self.proximity_min_distance_factor
    safe_distances = torch.clamp(min_distances, min=min_reward_distance)
    proximity_reward = torch.exp(self.proximity_reward_scale * (safe_distances - min_reward_distance))
    return proximity_reward
```

### Implementation Steps

**Phase 1: Core Penetration Detection (Steps 1-2)**
1. Implement tensor-based excessive force detection using Isaac Gym's contact force tensor API
2. Add geometric penetration detection using fingertip-to-box-center distance calculations

**Phase 2: Reward Integration (Steps 3-4)**
3. Add `penetration_penalty` component to reward calculation with strong negative weight (-10.0)
4. Modify `s2_fingerpad_proximity` reward to clamp minimum distance at half box width

**Phase 3: Configuration and Testing (Steps 5-6)**
5. Add penetration penalty weight to task configuration files
6. Test and validate that legitimate grasping behaviors remain rewarded while penetration is prevented

### Configuration Parameters

**YAML Configuration Section** (to be added to task config files):
```yaml
# Penetration prevention parameters
penetrationPrevention:
  # Force-based detection
  forceThreshold: 50.0              # Newtons - excessive force indicates penetration

  # Geometric detection
  geometricPenetrationFactor: 1.0   # Multiplier of half box size (1.0 = at box surface)

  # Proximity reward protection
  proximityMinDistanceFactor: 1.0   # Multiplier of half box width (1.0 = at surface)
  proximityRewardScale: -2.0        # Exponential decay rate for proximity reward

# Reward weights (existing section)
rewardWeights:
  penetration_penalty: -10.0        # Penalty intensity for penetration behavior
```

**Parameter Loading in Task Constructor**:
```python
def __init__(self, cfg, *args, **kwargs):
    # Load penetration prevention parameters
    penetration_cfg = cfg.get("penetrationPrevention", {})

    self.penetration_force_threshold = penetration_cfg.get("forceThreshold", 50.0)
    self.geometric_penetration_factor = penetration_cfg.get("geometricPenetrationFactor", 1.0)
    self.proximity_min_distance_factor = penetration_cfg.get("proximityMinDistanceFactor", 1.0)
    self.proximity_reward_scale = penetration_cfg.get("proximityRewardScale", -2.0)
```

**Updated Penetration Penalty Component**:
```python
def compute_penetration_penalty(self, obs_dict):
    """Binary penalty for penetration via forces or geometry."""
    force_penetration = self.detect_excessive_forces()
    geometric_penetration, _ = self.detect_geometric_penetration(obs_dict)

    penetration_detected = force_penetration | geometric_penetration
    # Return binary penalty (intensity controlled by rewardWeights)
    penalty = torch.where(penetration_detected,
                         torch.ones_like(penetration_detected, dtype=torch.float),
                         torch.zeros_like(penetration_detected, dtype=torch.float))
    return penalty
```

### Technical Specifications

**Configurable Parameters**:
- **`forceThreshold`**: 50.0N (default) - excessive forces indicate collision resolution/penetration
- **`geometricPenetrationFactor`**: 1.0 (default) - fraction of half box size for penetration detection
- **`proximityMinDistanceFactor`**: 1.0 (default) - fraction of half box width for minimum reward distance
- **`proximityRewardScale`**: -2.0 (default) - exponential decay rate for proximity reward
- **`penetration_penalty`**: -10.0 (default in rewardWeights) - penalty intensity for penetration behavior

**Benefits of Configurable Parameters**:
- Easy hyperparameter tuning without code changes
- Task-specific threshold adjustment
- Penalty intensity controlled through standard reward weights system
- Environment-specific force threshold tuning

## Expected Outcomes

**Primary Objective**: Ensure policies learn physically realistic manipulation strategies that can transfer to real-world robotic systems.

**Success Criteria**:
1. **Elimination of Penetration Exploits**: Policies cannot achieve high rewards through object interpenetration
2. **Preservation of Legitimate Behaviors**: Proper contact-based grasping strategies remain rewarded
3. **Training Stability**: Policy training converges to realistic manipulation skills
4. **Real-World Transferability**: Learned behaviors respect physical constraints applicable to real robots

**Performance Metrics**:
- Penetration detection rate during policy evaluation
- Comparison of reward curves before/after implementation
- Success rate on grasping tasks using only legitimate contact strategies
- Force magnitude distributions during successful grasps (should remain within realistic bounds)

**Long-term Impact**: This implementation establishes a foundation for physically grounded RL training that prioritizes realistic skill acquisition over simulation-specific exploits, enabling more reliable sim-to-real transfer for dexterous manipulation tasks.

This tensor-based approach with configurable parameters provides maximum performance and flexibility for preventing both force-based and geometric penetration exploitation while maintaining the research integrity of the dexterous manipulation learning process.
