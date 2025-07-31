"""
Reward component observer for logging individual reward terms to TensorBoard.
"""

import torch
from rl_games.common.algo_observer import AlgoObserver
from rl_games.algos_torch.torch_ext import AverageMeter


class RewardComponentObserver(AlgoObserver):
    """
    Custom observer that tracks and logs individual reward components to TensorBoard.

    This observer:
    - Accumulates reward components during episodes
    - Logs per-episode totals and means when episodes complete
    - Tracks both weighted and unweighted values
    - Provides separate statistics by termination type (success/failure/timeout)

    TensorBoard organization:
    - rewards/: Total rewards (logged by rl_games)
    - training/: Training metrics including termination rates
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

    def __init__(self, log_interval=10):
        super().__init__()

        # Episode accumulators
        self.episode_reward_sums = {}  # component_name -> tensor[num_envs]
        self.episode_lengths = None  # tensor[num_envs]
        self.num_envs = None
        self.device = None
        self.initialized = False

        # Meters for tracking episode means by termination type
        # Structure: episode_meters[termination_type][component_name] = AverageMeter
        self.episode_meters = {
            "all": {},
            "success": {},
            "failure": {},
            "timeout": {},
        }

        # Cumulative sums for computing step averages
        # Structure: cumulative_sums[termination_type][component_name] = {"rewards": float, "steps": int}
        self.cumulative_sums = {
            "all": {},
            "success": {},
            "failure": {},
            "timeout": {},
        }

        self.games_to_track = 100  # Same default as RL Games

        # Track total episodes for global statistics
        self.total_episodes = 0
        self.episodes_by_type = {"success": 0, "failure": 0, "timeout": 0}

        # Track windowed episodes for current logging interval
        self.windowed_total_episodes = 0
        self.windowed_episodes_by_type = {"success": 0, "failure": 0, "timeout": 0}

        # Store reference to algorithm and writer
        self.algo = None
        self.writer = None

        # Track which components we've discovered
        self.discovered_components = set()

        # Logging frequency control
        self.log_interval = log_interval  # Log every N done episodes
        self.episodes_since_last_log = 0

    def after_init(self, algo):
        """Store reference to the algorithm for accessing data."""
        self.algo = algo
        self.writer = algo.writer if hasattr(algo, "writer") else None

        # Get environment info for initialization
        if hasattr(algo, "vec_env") and hasattr(algo.vec_env, "env"):
            env = algo.vec_env.env
            if hasattr(env, "num_envs"):
                self._initialize(env.num_envs, algo.device)

        # Log configuration
        if self.writer:
            print(
                f"RewardComponentObserver: Logging every {self.log_interval} episodes"
            )

    def _initialize(self, num_envs, device):
        """Initialize tensors based on environment info."""
        self.num_envs = num_envs
        self.device = device
        self.episode_lengths = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.initialized = True

    def _ensure_component_exists(self, component_name):
        """
        Ensure meters and accumulators exist for a component.

        This is called dynamically as new components are discovered,
        avoiding the need to hardcode component names.
        """
        if component_name in self.discovered_components:
            return

        # Create accumulator
        self.episode_reward_sums[component_name] = torch.zeros(
            self.num_envs, device=self.device
        )

        # Create episode meters and cumulative sums for each termination type
        for term_type in self.episode_meters:
            self.episode_meters[term_type][component_name] = AverageMeter(
                1, self.games_to_track
            ).to(self.device)
            self.cumulative_sums[term_type][component_name] = {
                "rewards": 0.0,
                "steps": 0,
            }

        self.discovered_components.add(component_name)

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
                env = self.algo.vec_env.env
                if hasattr(env, "num_envs"):
                    self._initialize(env.num_envs, self.algo.device)

        # Process done environments
        self._process_done_episodes_vectorized(infos, done_indices)

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
            # Ensure component exists (dynamic discovery)
            self._ensure_component_exists(component_name)

            # Accumulate values
            if isinstance(values, torch.Tensor):
                self.episode_reward_sums[component_name] += values
            else:
                self.episode_reward_sums[component_name] += torch.tensor(
                    values, device=self.device
                )

    def _process_done_episodes_vectorized(self, infos, done_indices):
        """Process done episodes - update statistics and log if interval reached."""
        if not self.episode_reward_sums:
            return

        # Get termination type masks - vectorized approach
        success_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        failure_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        timeout_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Extract termination types from infos
        if "success" in infos and isinstance(infos["success"], torch.Tensor):
            success_mask = infos["success"].bool()
        if "failure" in infos and isinstance(infos["failure"], torch.Tensor):
            failure_mask = infos["failure"].bool()
        if "timeout" in infos and isinstance(infos["timeout"], torch.Tensor):
            timeout_mask = infos["timeout"].bool()

        # Get episode lengths for done environments
        done_lengths = self.episode_lengths[done_indices].float()

        # Update statistics for all components
        self._update_statistics(
            done_indices, done_lengths, success_mask, failure_mask, timeout_mask
        )

        # Reset accumulators for done environments
        self.episode_lengths[done_indices] = 0
        for component_name in self.episode_reward_sums:
            self.episode_reward_sums[component_name][done_indices] = 0

        # Update episode counters
        num_done = len(done_indices)
        success_count = success_mask[done_indices].sum().item()
        failure_count = failure_mask[done_indices].sum().item()
        timeout_count = timeout_mask[done_indices].sum().item()

        # Update cumulative counters
        self.total_episodes += num_done
        self.episodes_by_type["success"] += success_count
        self.episodes_by_type["failure"] += failure_count
        self.episodes_by_type["timeout"] += timeout_count

        # Update windowed counters
        self.windowed_total_episodes += num_done
        self.windowed_episodes_by_type["success"] += success_count
        self.windowed_episodes_by_type["failure"] += failure_count
        self.windowed_episodes_by_type["timeout"] += timeout_count

        # Track episodes for logging interval
        self.episodes_since_last_log += num_done

        # Log to TensorBoard if interval reached
        if self.writer and self.episodes_since_last_log >= self.log_interval:
            self._log_to_tensorboard()
            self.episodes_since_last_log = 0

    def _update_statistics(
        self, done_indices, done_lengths, success_mask, failure_mask, timeout_mask
    ):
        """Update all statistics without logging to TensorBoard."""
        # Process each component
        for component_name, accumulated_values in self.episode_reward_sums.items():
            # Skip 'total' component as it's already logged by rl_games
            if component_name == "total":
                continue

            # Get values for done environments
            done_values = accumulated_values[done_indices]

            # Update meters and cumulative sums for "all" episodes
            self.episode_meters["all"][component_name].update(done_values)

            # Update cumulative sums for step average calculation
            total_reward = done_values.sum().item()
            total_steps = done_lengths.sum().item()
            self.cumulative_sums["all"][component_name]["rewards"] += total_reward
            self.cumulative_sums["all"][component_name]["steps"] += total_steps

            # Process by termination type using masks
            for term_type, mask in [
                ("success", success_mask),
                ("failure", failure_mask),
                ("timeout", timeout_mask),
            ]:
                # Get indices of done environments with this termination type
                term_done_mask = mask[done_indices]
                if term_done_mask.any():
                    # Get values for this termination type
                    term_values = done_values[term_done_mask]
                    term_lengths = done_lengths[term_done_mask]

                    # Update episode meter for this termination type
                    self.episode_meters[term_type][component_name].update(term_values)

                    # Update cumulative sums
                    term_total_reward = term_values.sum().item()
                    term_total_steps = term_lengths.sum().item()
                    self.cumulative_sums[term_type][component_name][
                        "rewards"
                    ] += term_total_reward
                    self.cumulative_sums[term_type][component_name][
                        "steps"
                    ] += term_total_steps

    def _log_to_tensorboard(self):
        """Log all accumulated statistics to TensorBoard."""
        if not self.writer:
            return

        # Get current frame for logging
        frame = self.algo.frame if hasattr(self.algo, "frame") else self.total_episodes

        # Log each component
        for component_name in self.discovered_components:
            # Skip 'total' component as it's already logged by rl_games
            if component_name == "total":
                continue

            # Determine if this is a weighted component
            is_weighted = component_name.endswith("_weighted")
            base_name = (
                component_name.replace("_weighted", "")
                if is_weighted
                else component_name
            )
            weight_type = "weighted" if is_weighted else "raw"

            # Log for each termination type
            for term_type in ["all", "success", "failure", "timeout"]:
                if component_name in self.episode_meters[term_type]:
                    # Get episode mean from meter
                    episode_mean = self.episode_meters[term_type][
                        component_name
                    ].get_mean()

                    # Calculate step average from cumulative sums
                    cum_rewards = self.cumulative_sums[term_type][component_name][
                        "rewards"
                    ]
                    cum_steps = self.cumulative_sums[term_type][component_name]["steps"]
                    step_mean = cum_rewards / max(cum_steps, 1)

                    # Log to TensorBoard
                    episode_key = f"reward_breakdown/{term_type}/{weight_type}/episode/{base_name}"
                    step_key = (
                        f"reward_breakdown/{term_type}/{weight_type}/step/{base_name}"
                    )
                    self.writer.add_scalar(episode_key, episode_mean, frame)
                    self.writer.add_scalar(step_key, step_mean, frame)

        # Log termination type rates using windowed statistics
        if self.windowed_total_episodes > 0:
            for term_type, count in self.windowed_episodes_by_type.items():
                rate = count / self.windowed_total_episodes
                self.writer.add_scalar(
                    f"training/termination_rates/{term_type}", rate, frame
                )

        # Reset cumulative sums for next window
        for term_type in self.cumulative_sums:
            for component_name in self.cumulative_sums[term_type]:
                self.cumulative_sums[term_type][component_name] = {
                    "rewards": 0.0,
                    "steps": 0,
                }

        # Reset windowed episode counters for next window
        self.windowed_total_episodes = 0
        self.windowed_episodes_by_type = {"success": 0, "failure": 0, "timeout": 0}

    def after_clear_stats(self):
        """
        Called when stats are cleared.

        Following RL Games pattern: clear all meters and cumulative sums.
        """
        for term_type_meters in self.episode_meters.values():
            for meter in term_type_meters.values():
                meter.clear()

        # Reset cumulative sums
        for term_type in self.cumulative_sums:
            for component_name in self.cumulative_sums[term_type]:
                self.cumulative_sums[term_type][component_name] = {
                    "rewards": 0.0,
                    "steps": 0,
                }
