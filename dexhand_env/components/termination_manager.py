"""
Termination manager component for DexHand environment.

This module provides functionality to evaluate episode termination conditions
and determine when environments should be reset.
"""

import torch


class TerminationManager:
    """
    Manages episode termination decisions for the DexHand environment.

    This component provides functionality to:
    - Evaluate termination conditions (success/failure/timeout)
    - Generate termination signals and rewards
    - Track termination statistics for logging

    Three types of termination:
    - Success: Task completed successfully (positive reward)
    - Failure: Task failed due to violation (negative reward)
    - Timeout: Episode reached max length (neutral reward)
    """

    def __init__(self, parent, cfg):
        """
        Initialize the termination manager.

        Args:
            parent: Parent object (typically DexHandBase) that provides shared properties
            cfg: Configuration dictionary containing termination settings
        """
        self.parent = parent
        self.cfg = cfg

        # Episode status tracking
        self.episode_success = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.episode_failure = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )

        # Active criteria lists (which criteria to use)
        self.active_success_criteria = cfg["env"].get("activeSuccessCriteria", [])
        self.active_failure_criteria = cfg["env"].get(
            "activeFailureCriteria", ["hitting_ground"]
        )

        # Track which specific success/failure criteria triggered for each environment
        self.success_reasons = {}
        self.failure_reasons = {}

        # Maximum episode length for timeout termination
        self.max_episode_length = cfg["env"]["episodeLength"]

        # Termination rewards
        self.success_reward = cfg["env"].get("successReward", 10.0)
        self.failure_penalty = cfg["env"].get("failurePenalty", 5.0)
        self.timeout_reward = cfg["env"].get("timeoutReward", 0.0)

        # Track consecutive successes for curriculum learning
        self.consecutive_successes = 0
        self.max_consecutive_successes = cfg["env"].get("maxConsecutiveSuccesses", 50)

    @property
    def num_envs(self):
        """Get number of environments from parent."""
        return self.parent.num_envs

    @property
    def device(self):
        """Get device from parent."""
        return self.parent.device

    def evaluate(
        self,
        episode_step_count,
        builtin_success,
        task_success,
        builtin_failure,
        task_failure,
    ):
        """
        Evaluate termination conditions and determine which environments should reset.

        Args:
            episode_step_count: Buffer tracking episode step count
            builtin_success: Dictionary of built-in success criteria
            task_success: Dictionary of task-specific success criteria
            builtin_failure: Dictionary of built-in failure criteria
            task_failure: Dictionary of task-specific failure criteria

        Returns:
            Tuple containing:
                should_reset: Boolean tensor indicating which environments should reset
                termination_info: Dictionary with termination type information
                episode_rewards: Dictionary with reward components
        """
        # Initialize termination info
        termination_info = {}

        # Track active criteria and their results
        active_success = {}
        active_failure = {}

        # Process built-in success criteria
        for name, criterion in builtin_success.items():
            if name in self.active_success_criteria or not self.active_success_criteria:
                active_success[name] = criterion
                termination_info[f"success_{name}"] = criterion.float().mean().item()

        # Process task-specific success criteria
        for name, criterion in task_success.items():
            if name in self.active_success_criteria or not self.active_success_criteria:
                active_success[name] = criterion
                termination_info[f"success_{name}"] = criterion.float().mean().item()

        # Process built-in failure criteria
        for name, criterion in builtin_failure.items():
            if name in self.active_failure_criteria or not self.active_failure_criteria:
                active_failure[name] = criterion
                termination_info[f"failure_{name}"] = criterion.float().mean().item()

        # Process task-specific failure criteria
        for name, criterion in task_failure.items():
            if name in self.active_failure_criteria or not self.active_failure_criteria:
                active_failure[name] = criterion
                termination_info[f"failure_{name}"] = criterion.float().mean().item()

        # Initialize termination type tensors
        episode_success = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        episode_failure = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )

        # Check for success conditions
        for name, criterion in active_success.items():
            # Initialize tracking tensor for this reason if it doesn't exist
            if name not in self.success_reasons:
                self.success_reasons[name] = torch.zeros(
                    self.num_envs, device=self.device, dtype=torch.bool
                )

            # Identify new successes (environments that weren't successful before but now satisfy this criterion)
            new_successes = ~episode_success & criterion

            # Update tracking for this specific reason
            self.success_reasons[name] = new_successes | self.success_reasons[name]

            # Update overall success status
            episode_success = episode_success | criterion

        # Check for failure conditions
        for name, criterion in active_failure.items():
            # Initialize tracking tensor for this reason if it doesn't exist
            if name not in self.failure_reasons:
                self.failure_reasons[name] = torch.zeros(
                    self.num_envs, device=self.device, dtype=torch.bool
                )

            # Identify new failures
            new_failures = ~episode_failure & criterion

            # Update tracking for this specific reason
            self.failure_reasons[name] = new_failures | self.failure_reasons[name]

            # Update overall failure status
            episode_failure = episode_failure | criterion

        # Check for timeout termination
        timeout = episode_step_count >= self.max_episode_length - 1

        # Store episode outcomes
        self.episode_success = episode_success
        self.episode_failure = episode_failure

        # Determine which environments should reset
        should_reset = episode_success | episode_failure | timeout

        # Create termination type indicators (mutually exclusive)
        success_termination = episode_success & should_reset
        failure_termination = episode_failure & ~episode_success & should_reset
        timeout_termination = (
            timeout & ~episode_success & ~episode_failure & should_reset
        )

        # Add termination type information
        termination_info["success"] = success_termination
        termination_info["failure"] = failure_termination
        termination_info["timeout"] = timeout_termination

        # Track specific success/failure reasons
        for name, reason_mask in self.success_reasons.items():
            termination_info[f"success_reason_{name}"] = reason_mask

        for name, reason_mask in self.failure_reasons.items():
            termination_info[f"failure_reason_{name}"] = reason_mask

        # Calculate termination statistics
        success_count = success_termination.sum().item()
        failure_count = failure_termination.sum().item()
        timeout_count = timeout_termination.sum().item()

        termination_info["success_rate"] = success_count / self.num_envs
        termination_info["failure_rate"] = failure_count / self.num_envs
        termination_info["timeout_rate"] = timeout_count / self.num_envs

        # Generate episode rewards
        episode_rewards = self._get_termination_rewards(
            success_termination, failure_termination, timeout_termination
        )

        return should_reset, termination_info, episode_rewards

    def _get_termination_rewards(
        self, success_termination, failure_termination, timeout_termination
    ):
        """
        Get rewards based on termination type.

        Args:
            success_termination: Boolean tensor for success terminations
            failure_termination: Boolean tensor for failure terminations
            timeout_termination: Boolean tensor for timeout terminations

        Returns:
            Dictionary with reward components
        """
        rewards = {}

        # Success rewards
        success_reward = torch.zeros(self.num_envs, device=self.device)
        if torch.any(success_termination):
            success_reward[success_termination] = self.success_reward
        rewards["success"] = success_reward

        # Failure penalties
        failure_penalty = torch.zeros(self.num_envs, device=self.device)
        if torch.any(failure_termination):
            failure_penalty[failure_termination] = -self.failure_penalty
        rewards["failure"] = failure_penalty

        # Timeout rewards (usually neutral)
        timeout_reward = torch.zeros(self.num_envs, device=self.device)
        if torch.any(timeout_termination):
            timeout_reward[timeout_termination] = self.timeout_reward
        rewards["timeout"] = timeout_reward

        return rewards

    def update_consecutive_successes(self, success_tensor):
        """
        Update consecutive success tracking.

        Args:
            success_tensor: Boolean tensor indicating success for each environment
        """
        # If at least one environment had a success
        if torch.any(success_tensor):
            self.consecutive_successes += 1
        else:
            self.consecutive_successes = 0

        # Cap at max value
        self.consecutive_successes = min(
            self.consecutive_successes, self.max_consecutive_successes
        )

    def reset_tracking(self, env_ids):
        """
        Reset termination tracking for specified environments.

        Args:
            env_ids: Environment indices to reset
        """
        # Reset success/failure flags
        self.episode_success[env_ids] = False
        self.episode_failure[env_ids] = False

        # Reset success/failure reason trackers
        for name in self.success_reasons:
            self.success_reasons[name][env_ids] = False

        for name in self.failure_reasons:
            self.failure_reasons[name][env_ids] = False

        # Note: We don't reset consecutive_successes here as it tracks across episodes
