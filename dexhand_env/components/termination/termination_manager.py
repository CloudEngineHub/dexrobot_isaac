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

    def __init__(self, parent, task_cfg):
        """
        Initialize the termination manager.

        Args:
            parent: Parent object (typically DexHandBase) that provides shared properties
            task_cfg: Task-specific configuration dictionary
        """
        self.parent = parent
        self.task_cfg = task_cfg

        # Episode status tracking
        self.episode_success = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.episode_failure = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )

        # Active criteria lists - read from config, no hardcoded defaults
        # If not specified in config, use empty list (means use all available)
        termination_cfg = task_cfg.get("termination", {})
        self.active_success_criteria = termination_cfg.get("activeSuccessCriteria", [])
        self.active_failure_criteria = termination_cfg.get("activeFailureCriteria", [])

        # Pre-allocate tensors for all possible success/failure reasons
        # This prevents dynamic dictionary growth during runtime which causes memory leaks
        self._initialize_reason_tracking()

        # Maximum episode length for timeout termination
        self.max_episode_length = parent.env_cfg["episodeLength"]

        # Termination rewards - read from reward weights config
        reward_weights = task_cfg["rewardWeights"]
        self.success_reward = reward_weights["termination_success"]
        self.failure_penalty = reward_weights["termination_failure_penalty"]
        self.timeout_penalty = reward_weights["termination_timeout_penalty"]

        # Raw values for unweighted tracking (before scaling)
        self.RAW_SUCCESS = 1.0
        self.RAW_FAILURE = 1.0
        self.RAW_TIMEOUT = 1.0

        # Track consecutive successes for curriculum learning
        self.consecutive_successes = 0
        self.max_consecutive_successes = task_cfg["maxConsecutiveSuccesses"]

    @property
    def num_envs(self):
        """Get number of environments from parent."""
        return self.parent.num_envs

    @property
    def device(self):
        """Get device from parent."""
        return self.parent.device

    def _initialize_reason_tracking(self):
        """
        Pre-allocate tensors for all possible success/failure reasons.

        This prevents dynamic dictionary growth during runtime which can cause memory leaks.
        We pre-allocate for all commonly used criteria names.
        """
        # Initialize reason tracking dictionaries (will be populated dynamically)
        self.success_reasons = {}
        self.failure_reasons = {}

    def validate_criteria(self, builtin_dict, task_dict, active_list, criteria_type):
        """
        Fail-fast if any active criterion is missing.

        Args:
            builtin_dict: Dictionary of builtin criteria
            task_dict: Dictionary of task-specific criteria
            active_list: List of active criteria names
            criteria_type: String describing the type (for error messages)
        """
        if not active_list:  # Empty list means use all available
            return

        all_available = set(builtin_dict.keys()) | set(task_dict.keys())
        for criterion in active_list:
            if criterion not in all_available:
                raise RuntimeError(
                    f"{criteria_type} criterion '{criterion}' is configured as active but not implemented! "
                    f"Available criteria: {sorted(all_available)}. "
                    f"Either implement '{criterion}' or remove it from active{criteria_type}Criteria."
                )

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
                termination_rewards: Dictionary with one-time termination bonuses/penalties
        """
        # Validate that all active criteria are implemented
        self.validate_criteria(
            builtin_success, task_success, self.active_success_criteria, "Success"
        )
        self.validate_criteria(
            builtin_failure, task_failure, self.active_failure_criteria, "Failure"
        )

        # Initialize termination info
        termination_info = {}

        # Track active criteria and their results
        active_success = {}
        active_failure = {}

        # Process built-in success criteria
        for name, criterion in builtin_success.items():
            if name in self.active_success_criteria or not self.active_success_criteria:
                active_success[name] = criterion
                # Keep as tensor for logging - logger will handle conversion
                termination_info[f"success_{name}"] = criterion.float().mean()

        # Process task-specific success criteria
        for name, criterion in task_success.items():
            if name in self.active_success_criteria or not self.active_success_criteria:
                active_success[name] = criterion
                # Keep as tensor for logging - logger will handle conversion
                termination_info[f"success_{name}"] = criterion.float().mean()

        # Process built-in failure criteria
        for name, criterion in builtin_failure.items():
            if name in self.active_failure_criteria or not self.active_failure_criteria:
                active_failure[name] = criterion
                # Keep as tensor for logging - logger will handle conversion
                termination_info[f"failure_{name}"] = criterion.float().mean()

        # Process task-specific failure criteria
        for name, criterion in task_failure.items():
            if name in self.active_failure_criteria or not self.active_failure_criteria:
                active_failure[name] = criterion
                # Keep as tensor for logging - logger will handle conversion
                termination_info[f"failure_{name}"] = criterion.float().mean()

        # Initialize termination type tensors
        episode_success = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        episode_failure = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )

        # Check for success conditions
        for name, criterion in active_success.items():
            # Create tracking tensor on first use
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
            # Create tracking tensor on first use
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

        # Calculate termination statistics - keep as tensors
        success_count = success_termination.sum()
        failure_count = failure_termination.sum()
        timeout_count = timeout_termination.sum()

        # Keep rates as tensors for logging - avoid .item() calls
        termination_info["success_rate"] = success_count.float() / self.num_envs
        termination_info["failure_rate"] = failure_count.float() / self.num_envs
        termination_info["timeout_rate"] = timeout_count.float() / self.num_envs

        # Generate termination rewards (one-time bonuses/penalties)
        termination_rewards, raw_termination_rewards = self._get_termination_rewards(
            success_termination, failure_termination, timeout_termination
        )

        return (
            should_reset,
            termination_info,
            termination_rewards,
            raw_termination_rewards,
        )

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
            Dictionary with both raw and weighted reward components
        """
        rewards = {}
        raw_rewards = {}

        # Success rewards
        success_reward = torch.zeros(self.num_envs, device=self.device)
        success_raw = torch.zeros(self.num_envs, device=self.device)
        success_reward[success_termination] = self.success_reward
        success_raw[success_termination] = self.RAW_SUCCESS
        rewards["success"] = success_reward
        raw_rewards["success"] = success_raw

        # Failure penalties (note: stored as positive in config, applied as negative)
        failure_penalty = torch.zeros(self.num_envs, device=self.device)
        failure_raw = torch.zeros(self.num_envs, device=self.device)
        failure_penalty[failure_termination] = -self.failure_penalty
        failure_raw[failure_termination] = self.RAW_FAILURE
        rewards["failure_penalty"] = failure_penalty
        raw_rewards["failure_penalty"] = failure_raw

        # Timeout penalties (note: stored as positive in config, applied as negative)
        timeout_penalty = torch.zeros(self.num_envs, device=self.device)
        timeout_raw = torch.zeros(self.num_envs, device=self.device)
        timeout_penalty[timeout_termination] = -self.timeout_penalty
        timeout_raw[timeout_termination] = self.RAW_TIMEOUT
        rewards["timeout_penalty"] = timeout_penalty
        raw_rewards["timeout_penalty"] = timeout_raw

        return rewards, raw_rewards

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
