"""
Success/Failure tracker component for DexHand environment.

This module provides functionality to track success and failure criteria
for dexterous manipulation tasks.
"""

# Import PyTorch
import torch


class SuccessFailureTracker:
    """
    Tracks success and failure criteria for dexterous manipulation tasks.
    
    This component provides functionality to:
    - Evaluate success and failure criteria
    - Track which criteria triggered for each environment
    - Generate episode termination signals and information
    """
    
    def __init__(self, num_envs, device, cfg):
        """
        Initialize the success/failure tracker.
        
        Args:
            num_envs: Number of environments
            device: PyTorch device
            cfg: Configuration dictionary
        """
        self.num_envs = num_envs
        self.device = device
        
        # Episode status tracking
        self.episode_success = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.episode_failure = torch.zeros(num_envs, device=device, dtype=torch.bool)
        
        # Active criteria lists (which criteria to use)
        self.active_success_criteria = cfg["env"].get("activeSuccessCriteria", [])
        self.active_failure_criteria = cfg["env"].get("activeFailureCriteria", ["hitting_ground"])
        
        # Track which specific success/failure criteria triggered for each environment
        self.success_reasons = {}
        self.failure_reasons = {}
        
        # Maximum episode length
        self.max_episode_length = cfg["env"]["episodeLength"]
        
        # Success and failure rewards
        self.success_reward = cfg["env"].get("successReward", 10.0)
        self.failure_penalty = cfg["env"].get("failurePenalty", 5.0)
        
        # Track consecutive successes for curriculum learning
        self.consecutive_successes = 0
        self.max_consecutive_successes = cfg["env"].get("maxConsecutiveSuccesses", 50)
    
    def evaluate(self, progress_buf, builtin_success, task_success, builtin_failure, task_failure):
        """
        Evaluate success and failure criteria and update episode status.
        
        This method checks all active success and failure criteria and determines
        which environments have completed episodes (success or failure).
        
        Args:
            progress_buf: Buffer tracking episode progress
            builtin_success: Dictionary of built-in success criteria
            task_success: Dictionary of task-specific success criteria
            builtin_failure: Dictionary of built-in failure criteria
            task_failure: Dictionary of task-specific failure criteria
            
        Returns:
            Tuple containing:
                done_buf: Tensor indicating which environments are done
                info: Dictionary with episode information (success/failure criteria)
        """
        # Initialize episode info
        info = {}
        
        # Track active criteria and their results
        active_success = {}
        active_failure = {}
        
        # Process built-in success criteria
        for name, criterion in builtin_success.items():
            if name in self.active_success_criteria or not self.active_success_criteria:
                active_success[name] = criterion
                info[f"success_{name}"] = criterion.float().mean().item()
        
        # Process task-specific success criteria
        for name, criterion in task_success.items():
            if name in self.active_success_criteria or not self.active_success_criteria:
                active_success[name] = criterion
                info[f"success_{name}"] = criterion.float().mean().item()
        
        # Process built-in failure criteria
        for name, criterion in builtin_failure.items():
            if name in self.active_failure_criteria or not self.active_failure_criteria:
                active_failure[name] = criterion
                info[f"failure_{name}"] = criterion.float().mean().item()
        
        # Process task-specific failure criteria
        for name, criterion in task_failure.items():
            if name in self.active_failure_criteria or not self.active_failure_criteria:
                active_failure[name] = criterion
                info[f"failure_{name}"] = criterion.float().mean().item()
        
        # Initialize success and failure tensors
        episode_success = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        episode_failure = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        # Check for any success or failure conditions
        for name, criterion in active_success.items():
            # Initialize tracking tensor for this reason if it doesn't exist
            if name not in self.success_reasons:
                self.success_reasons[name] = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            
            # Identify new successes (environments that weren't successful before but now satisfy this criterion)
            new_successes = ~episode_success & criterion
            
            # Update tracking for this specific reason
            self.success_reasons[name] = new_successes | self.success_reasons[name]
            
            # Update overall success status
            episode_success = episode_success | criterion
        
        for name, criterion in active_failure.items():
            # Initialize tracking tensor for this reason if it doesn't exist
            if name not in self.failure_reasons:
                self.failure_reasons[name] = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            
            # Identify new failures
            new_failures = ~episode_failure & criterion
            
            # Update tracking for this specific reason
            self.failure_reasons[name] = new_failures | self.failure_reasons[name]
            
            # Update overall failure status
            episode_failure = episode_failure | criterion
        
        # Store episode outcomes
        self.episode_success = episode_success
        self.episode_failure = episode_failure
        
        # Episode is done if it's a success or failure, or if max steps reached
        max_steps_done = progress_buf >= self.max_episode_length - 1
        done_buf = episode_success | episode_failure | max_steps_done
        
        # Create tensors indicating termination reasons
        timeout_done = max_steps_done & ~episode_success & ~episode_failure
        
        # Track termination reasons as tensors (num_envs,)
        info["success"] = episode_success
        info["failure"] = episode_failure 
        info["timeout"] = timeout_done
        
        # Track specific success/failure reasons
        for name, reason_mask in self.success_reasons.items():
            info[f"success_reason_{name}"] = reason_mask
                
        for name, reason_mask in self.failure_reasons.items():
            info[f"failure_reason_{name}"] = reason_mask
        
        # Record overall statistics for easy logging
        success_count = episode_success.sum().item()
        failure_count = episode_failure.sum().item()
        timeout_count = (max_steps_done & ~episode_success & ~episode_failure).sum().item()
        
        info["success_rate"] = success_count / self.num_envs
        info["failure_rate"] = failure_count / self.num_envs
        info["timeout_rate"] = timeout_count / self.num_envs
        
        return done_buf, info
    
    def get_rewards(self):
        """
        Get success and failure rewards based on episode status.
        
        Returns:
            Dictionary with success and failure rewards
        """
        rewards = {}
        
        # Add success reward
        success_reward = torch.zeros(self.num_envs, device=self.device)
        if torch.any(self.episode_success):
            success_reward[self.episode_success] = self.success_reward
        rewards["success"] = success_reward
        
        # Add failure penalty
        failure_penalty = torch.zeros(self.num_envs, device=self.device)
        if torch.any(self.episode_failure):
            failure_penalty[self.episode_failure] = -self.failure_penalty
        rewards["failure"] = failure_penalty
        
        return rewards
    
    def update(self, success_tensor):
        """
        Update success tracking.
        
        Args:
            success_tensor: Boolean tensor indicating success for each environment
        """
        # If at least one environment had a success
        if torch.any(success_tensor):
            self.consecutive_successes += 1
        else:
            self.consecutive_successes = 0
            
        # Cap at max value
        self.consecutive_successes = min(self.consecutive_successes, self.max_consecutive_successes)
        
    def reset(self, env_ids):
        """
        Reset success/failure tracking for specified environments.
        
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