"""
Step Processor component for DexHand environment.

This module handles the complex post-physics step processing logic for the DexHand environment.
It coordinates reward computation, termination evaluation, and state updates.
"""

from loguru import logger
import torch


class StepProcessor:
    """
    Handles post-physics step processing for the DexHand environment.

    This component coordinates:
    - Reward computation
    - Termination evaluation
    - State updates
    - Reset handling
    """

    def __init__(self, parent):
        """Initialize StepProcessor with parent environment reference."""
        self.parent = parent

    @property
    def device(self):
        """Access device from parent (single source of truth)."""
        return self.parent.device

    def process_physics_step(self):
        """Process state after physics simulation step."""
        try:
            # Refresh tensors from simulation
            self.parent.tensor_manager.refresh_tensors(
                self.parent.contact_force_local_body_indices
            )

            # Observations were already computed in pre_physics_step
            # We just need to return them here
            # The obs_buf and obs_dict are already set from pre_physics_step

            # Update episode progress directly first
            self.parent.episode_step_count += 1

            # Check for episode termination using TerminationManager
            termination_results = self._evaluate_termination()
            (
                should_reset,
                termination_info,
                termination_rewards,
                raw_termination_rewards,
            ) = termination_results

            self.parent.reset_buf = should_reset

            # Compute rewards
            reward_components = self._compute_rewards()
            self.parent.last_reward_components = reward_components

            # Track successes for curriculum learning
            if "success" in termination_info:
                self.parent.termination_manager.update_consecutive_successes(
                    termination_info["success"]
                )

            # Add termination rewards to total rewards AND tracked components
            self._add_termination_rewards(
                termination_rewards, raw_termination_rewards, reward_components
            )

            # Reset environments that completed episodes
            if torch.any(self.parent.reset_buf):
                env_ids_to_reset = torch.nonzero(self.parent.reset_buf).flatten()
                self.parent.reset_idx(env_ids_to_reset)

            # Physics step count tracking for auto-detecting steps per control
            self.parent.physics_manager.mark_control_step()

            # Update extras
            self._update_extras(termination_info)

            return (
                self.parent.obs_buf,
                self.parent.rew_buf,
                self.parent.reset_buf,
                self.parent.extras,
            )

        except Exception as e:
            logger.critical(f"CRITICAL ERROR in post_physics_step: {e}")
            import traceback

            traceback.print_exc()
            raise

    def _evaluate_termination(self):
        """Evaluate termination conditions."""
        builtin_success = {}
        task_success = {}
        builtin_failure = {}
        task_failure = {}

        # Implement ground collision detection
        if "height_safety" in self.parent.cfg["termination"]:
            height_thresholds = self.parent.cfg["termination"]["height_safety"]

            # Check hand base height
            hand_base_z = self.parent.rigid_body_states[
                :, self.parent.hand_local_rigid_body_index, 2
            ]
            handbase_hitting_ground = (
                hand_base_z < height_thresholds["handbase_threshold"]
            )

            # Check fingertip heights (already in obs_dict)
            fingertip_heights = self.parent.obs_dict["fingertip_poses_world"].view(
                self.parent.num_envs, 5, 7
            )[:, :, 2]
            min_fingertip_height = torch.min(fingertip_heights, dim=1)[0]
            fingertips_hitting_ground = (
                min_fingertip_height < height_thresholds["fingertip_threshold"]
            )

            # Combine both conditions
            builtin_failure["hitting_ground"] = (
                handbase_hitting_ground | fingertips_hitting_ground
            )

        # Get task-specific success/failure criteria
        task_success = self.parent.task.check_task_success_criteria()
        task_failure = self.parent.task.check_task_failure_criteria()

        # Evaluate termination conditions
        return self.parent.termination_manager.evaluate(
            self.parent.episode_step_count,
            builtin_success,
            task_success,
            builtin_failure,
            task_failure,
        )

    def _compute_rewards(self):
        """Compute task rewards using centralized orchestration."""
        # Get common reward terms
        common_rewards = self.parent.reward_calculator.compute_common_reward_terms(
            self.parent.obs_dict, self.parent
        )

        # Get task-specific reward terms
        task_rewards = self.parent.task.compute_task_reward_terms(self.parent.obs_dict)

        # Combine rewards using reward calculator
        (
            self.parent.rew_buf[:],
            reward_components,
        ) = self.parent.reward_calculator.compute_total_reward(
            common_rewards=common_rewards,
            task_rewards=task_rewards,
        )

        return reward_components

    def _add_termination_rewards(
        self, termination_rewards, raw_termination_rewards, reward_components
    ):
        """Add termination rewards to total rewards and tracked components."""
        for reward_type, reward_tensor in termination_rewards.items():
            self.parent.rew_buf += reward_tensor
            # Track both raw and weighted termination rewards
            # Raw values
            raw_key = f"termination_{reward_type}"
            reward_components[raw_key] = raw_termination_rewards[reward_type]
            # Weighted values
            weighted_key = f"termination_{reward_type}_weighted"
            reward_components[weighted_key] = reward_tensor

        # Update last_reward_components to include termination rewards
        self.parent.last_reward_components = reward_components

    def _update_extras(self, termination_info):
        """Update extras dictionary with relevant information."""
        self.parent.extras = {
            "consecutive_successes": self.parent.termination_manager.consecutive_successes,
            "episode_length": self.parent.episode_step_count.clone(),  # Add episode length for logging
        }

        # Add termination info to extras for logging
        self.parent.extras.update(termination_info)

        # Add reward components to extras for logging
        self.parent.extras["reward_components"] = self.parent.last_reward_components
