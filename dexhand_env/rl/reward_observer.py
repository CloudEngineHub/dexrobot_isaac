"""
Reward component observer for logging individual reward terms to TensorBoard.
"""

import torch
from rl_games.common.algo_observer import AlgoObserver


class RewardComponentObserver(AlgoObserver):
    """
    Custom observer that tracks and logs individual reward components to TensorBoard.

    This observer:
    - Accumulates reward components during episodes
    - Logs per-episode totals and means when episodes complete
    - Tracks both weighted and unweighted values
    - Provides separate statistics by termination type (success/failure/timeout)

    TensorBoard organization:
    - rewards/: Total rewards (logged by rl_games) and termination rates
    - reward_breakdown/: Detailed component analysis with 5-level hierarchy
      - Level 1: reward_breakdown
      - Level 2: termination_type (all, success, failure, timeout)
      - Level 3: weight_type (raw, weighted)
      - Level 4: aggregation (episode, step)
      - Level 5: component_name (alive, height_safety, finger_velocity, etc.)

    Example keys:
    - reward_breakdown/all/raw/episode/alive
    - reward_breakdown/all/raw/step/alive
    - reward_breakdown/success/weighted/episode/height_safety
    - reward_breakdown/timeout/raw/step/finger_velocity
    """

    def __init__(self):
        super().__init__()

        # Episode accumulators
        self.episode_reward_sums = {}  # component_name -> tensor[num_envs]
        self.episode_lengths = None  # tensor[num_envs]
        self.num_envs = None
        self.device = None
        self.initialized = False

        # Track total episodes for global statistics
        self.total_episodes = 0
        self.episodes_by_type = {"success": 0, "failure": 0, "timeout": 0}

        # Store reference to algorithm and writer
        self.algo = None
        self.writer = None

    def after_init(self, algo):
        """Store reference to the algorithm for accessing data."""
        self.algo = algo
        self.writer = algo.writer if hasattr(algo, "writer") else None

        # Get environment info for initialization
        if hasattr(algo, "env") and hasattr(algo.env, "num_envs"):
            self._initialize(algo.env.num_envs, algo.device)

    def _initialize(self, num_envs, device):
        """Initialize tensors based on environment info."""
        self.num_envs = num_envs
        self.device = device
        self.episode_lengths = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.initialized = True

    def process_infos(self, infos, done_indices):
        """
        Process info dictionaries from completed episodes.

        This is the main entry point called by rl_games when episodes complete.

        Args:
            infos: Info dictionaries from the environment
            done_indices: Indices of environments that are done
        """
        if not infos or len(done_indices) == 0:
            return

        # Initialize if needed
        if not self.initialized and hasattr(self, "algo"):
            if hasattr(self.algo, "vec_env"):
                num_envs = self.algo.vec_env.env.num_envs
                device = self.algo.device
                self._initialize(num_envs, device)

        # Process done environments and log their rewards
        self._process_done_episodes(infos, done_indices)

    def after_steps(self):
        """
        Called after each training step to accumulate rewards.
        """
        if not self.initialized:
            return

        # Increment episode lengths
        self.episode_lengths += 1

        # Try to get current reward components from the environment
        if hasattr(self.algo, "vec_env") and hasattr(self.algo.vec_env, "env"):
            env = self.algo.vec_env.env
            if hasattr(env, "extras") and "reward_components" in env.extras:
                self._accumulate_reward_components(env.extras["reward_components"])

    def _accumulate_reward_components(self, reward_components):
        """Accumulate reward components across steps."""
        for component_name, values in reward_components.items():
            if component_name not in self.episode_reward_sums:
                if isinstance(values, torch.Tensor):
                    self.episode_reward_sums[component_name] = torch.zeros_like(values)
                else:
                    self.episode_reward_sums[component_name] = torch.zeros(
                        self.num_envs, device=self.device
                    )

            # Accumulate values
            if isinstance(values, torch.Tensor):
                self.episode_reward_sums[component_name] += values
            else:
                self.episode_reward_sums[component_name] += values

    def _process_done_episodes(self, infos, done_indices):
        """Process and log rewards for completed episodes."""
        # Get termination types
        termination_types = self._get_termination_types(infos, done_indices)

        # Process each done environment
        for idx in done_indices:
            env_id = idx.item()
            # Use our tracked episode length, not the one from infos (which might be reset)
            episode_length = self.episode_lengths[env_id].item()
            termination_type = termination_types.get(env_id, "unknown")

            if episode_length > 0 and self.episode_reward_sums:
                self._log_accumulated_rewards(env_id, episode_length, termination_type)

            # Reset accumulators for this environment
            self.episode_lengths[env_id] = 0
            for component_name in self.episode_reward_sums:
                if isinstance(self.episode_reward_sums[component_name], torch.Tensor):
                    self.episode_reward_sums[component_name][env_id] = 0

            # Update episode counters
            self.total_episodes += 1
            if termination_type in self.episodes_by_type:
                self.episodes_by_type[termination_type] += 1

    def _get_termination_types(self, infos, done_indices):
        """Extract termination type for each done environment."""
        termination_types = {}

        # Check for termination type indicators in infos
        for key in ["success", "failure", "timeout"]:
            if key in infos:
                values = infos[key]
                if isinstance(values, torch.Tensor):
                    for idx in done_indices:
                        env_id = idx.item()
                        if env_id < len(values) and values[env_id]:
                            termination_types[env_id] = key

        return termination_types

    def _log_accumulated_rewards(self, env_id, episode_length, termination_type):
        """Log accumulated reward components for a completed episode."""
        if self.writer is None:
            return

        frame = self.algo.frame if hasattr(self.algo, "frame") else self.total_episodes

        # Log each accumulated component
        for component_name, accumulated_values in self.episode_reward_sums.items():
            # Skip 'total' component as it's already logged by rl_games
            if component_name == "total":
                continue

            # Get accumulated value for this environment
            if isinstance(accumulated_values, torch.Tensor):
                if accumulated_values.dim() > 0 and env_id < len(accumulated_values):
                    episode_total = accumulated_values[env_id].item()
                else:
                    episode_total = accumulated_values.item()
            else:
                episode_total = accumulated_values

            # Calculate per-step average from episode total
            step_average = episode_total / episode_length if episode_length > 0 else 0.0

            # Determine if this is a weighted component
            is_weighted = component_name.endswith("_weighted")
            base_name = (
                component_name.replace("_weighted", "")
                if is_weighted
                else component_name
            )

            # Determine weight type
            weight_type = "weighted" if is_weighted else "raw"

            # Log for "all" episodes
            episode_key = f"reward_breakdown/all/{weight_type}/episode/{base_name}"
            step_key = f"reward_breakdown/all/{weight_type}/step/{base_name}"
            self.writer.add_scalar(episode_key, episode_total, frame)
            self.writer.add_scalar(step_key, step_average, frame)

            # Also log by specific termination type
            if termination_type in ["success", "failure", "timeout"]:
                episode_key = f"reward_breakdown/{termination_type}/{weight_type}/episode/{base_name}"
                step_key = f"reward_breakdown/{termination_type}/{weight_type}/step/{base_name}"
                self.writer.add_scalar(episode_key, episode_total, frame)
                self.writer.add_scalar(step_key, step_average, frame)

        # Log termination type rates periodically
        if self.total_episodes > 0 and self.total_episodes % 100 == 0:
            for term_type, count in self.episodes_by_type.items():
                rate = count / self.total_episodes
                self.writer.add_scalar(
                    f"rewards/termination_rates/{term_type}", rate, frame
                )
