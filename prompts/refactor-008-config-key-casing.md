# refactor-008-config-key-casing.md

Unify configuration key naming to snake_case under task section for code consistency.

## Context

The configuration files have inconsistent naming conventions, particularly in the `task:` section where some keys use camelCase while Python code conventions prefer snake_case. This creates cognitive friction when working between config files and Python code.

**Design Decision**: Keep other sections (env, sim, train) as camelCase for CLI usability, but unify the `task:` section to snake_case for code consistency since these keys are primarily accessed by Python code rather than CLI overrides.

## Current State

**BaseTask.yaml - 16 camelCase keys in task section:**
- `policyControlsHandBase` → `policy_controls_hand_base`
- `policyControlsFingers` → `policy_controls_fingers`
- `defaultBaseTargets` → `default_base_targets`
- `defaultFingerTargets` → `default_finger_targets`
- `maxFingerJointVelocity` → `max_finger_joint_velocity`
- `maxBaseLinearVelocity` → `max_base_linear_velocity`
- `maxBaseAngularVelocity` → `max_base_angular_velocity`
- `activeSuccessCriteria` → `active_success_criteria`
- `activeFailureCriteria` → `active_failure_criteria`
- `rewardWeights` → `reward_weights`
- `enableComponentDebugLogs` → `enable_component_debug_logs`
- `maxConsecutiveSuccesses` → `max_consecutive_successes`
- `contactForceBodies` → `contact_force_bodies`
- `contactBinaryThreshold` → `contact_binary_threshold`
- `contactVisualization` → `contact_visualization`
- `policyObservationKeys` → `policy_observation_keys`

**BlindGrasping.yaml - 9 camelCase keys in task section:**
- `maxBaseLinearVelocity` → `max_base_linear_velocity`
- `maxBaseAngularVelocity` → `max_base_angular_velocity`
- `maxFingerJointVelocity` → `max_finger_joint_velocity`
- `contactBinaryThreshold` → `contact_binary_threshold`
- `penetrationPrevention` → `penetration_prevention`
- `policyObservationKeys` → `policy_observation_keys`
- `activeSuccessCriteria` → `active_success_criteria`
- `activeFailureCriteria` → `active_failure_criteria`
- `rewardWeights` → `reward_weights`

## Desired Outcome

1. **Configuration Files**: All task section keys use consistent snake_case naming
2. **Code References**: All Python code references updated to use new snake_case keys
3. **No Breaking CLI**: env/sim/train sections keep camelCase for CLI usability
4. **Zero Backward Compatibility**: Clean break, no legacy support needed

## Code References Requiring Updates

**9 Python files with 17 key references:**

### `/home/yiwen/dexrobot_isaac/dexhand_env/tasks/dexhand_base.py` (11 references)
- Line 362, 364: `contactForceBodies` → `contact_force_bodies`
- Line 468: `policyControlsHandBase` → `policy_controls_hand_base`
- Line 469: `policyControlsFingers` → `policy_controls_fingers`
- Line 470: `maxFingerJointVelocity` → `max_finger_joint_velocity`
- Line 471: `maxBaseLinearVelocity` → `max_base_linear_velocity`
- Line 472: `maxBaseAngularVelocity` → `max_base_angular_velocity`
- Line 480, 482: `defaultBaseTargets` → `default_base_targets`
- Line 484, 486: `defaultFingerTargets` → `default_finger_targets`
- Line 1103: `enableComponentDebugLogs` → `enable_component_debug_logs`

### `/home/yiwen/dexrobot_isaac/dexhand_env/components/termination/termination_manager.py` (4 references)
- Line 48: `activeSuccessCriteria` → `active_success_criteria`
- Line 49: `activeFailureCriteria` → `active_failure_criteria`
- Line 59: `rewardWeights` → `reward_weights`
- Line 71: `maxConsecutiveSuccesses` → `max_consecutive_successes`

### `/home/yiwen/dexrobot_isaac/dexhand_env/tasks/blind_grasping_task.py` (3 references)
- Line 185: `penetrationPrevention` → `penetration_prevention`
- Line 797: `contactBinaryThreshold` → `contact_binary_threshold`
- Line 1245: `activeFailureCriteria` → `active_failure_criteria` (in comment)

### `/home/yiwen/dexrobot_isaac/dexhand_env/components/reward/reward_calculator.py` (1 reference)
- Line 37: `rewardWeights` → `reward_weights`

### `/home/yiwen/dexrobot_isaac/dexhand_env/components/observation/observation_encoder.py` (1 reference)
- Line 712: `contactBinaryThreshold` → `contact_binary_threshold`

### `/home/yiwen/dexrobot_isaac/dexhand_env/components/graphics/viewer_controller.py` (1 reference)
- Line 76: `contactVisualization` → `contact_visualization`

### `/home/yiwen/dexrobot_isaac/dexhand_env/components/initialization/initialization_manager.py` (1 reference)
- Line 71: `policyObservationKeys` → `policy_observation_keys`

### `/home/yiwen/dexrobot_isaac/dexhand_env/components/initialization/hand_initializer.py` (1 reference)
- Line 557: `contactForceBodies` → `contact_force_bodies` (in comment)

### `/home/yiwen/dexrobot_isaac/examples/dexhand_test.py` (1 reference)
- Line 1044: `policyControlsHandBase` → `policy_controls_hand_base` (in comment)

## Implementation Notes

**Architecture Compliance:**
- Follows fail-fast philosophy - no backward compatibility, clean break
- Maintains single source of truth - no dual naming support
- Preserves CLI usability for frequently-used env/sim/train keys

**Testing Strategy:**
- Test both BaseTask and BlindGrasping task loading
- Verify test script and training pipeline work correctly
- Confirm all config key access patterns function properly

**Breaking Change Protocol:**
- No backward compatibility required per CLAUDE.md guidelines
- Clean architectural improvement with immediate effect
- All references must be updated atomically

## Constraints

- **Scope Limited**: Only `task:` section keys, leave env/sim/train as camelCase
- **No Legacy Support**: Clean break, all references must be updated
- **Architecture Boundaries**: Respect component separation during updates
- **Testing Required**: Both test script and training pipeline must work

## Dependencies

None - standalone refactoring task with no external dependencies.
