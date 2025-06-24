"""
Base class for vectorized tasks in Isaac Gym.

This module defines the VecTask abstract base class that all tasks should inherit from.
"""

from abc import ABC
from typing import Dict, Any

import numpy as np
import random

# Import IsaacGym first (before torch)
from isaacgym import gymapi

# Then import torch
import torch
from loguru import logger


# Global variable to store the simulation instance
EXISTING_SIM = None
SCREEN_CAPTURE_RESOLUTION = (1027, 768)


def _create_sim_once(gym, *args, **kwargs):
    global EXISTING_SIM

    # Check if GPU pipeline is enabled - always create a new sim for GPU pipeline
    if len(args) >= 4 and isinstance(args[3], gymapi.SimParams):
        sim_params = args[3]
        if hasattr(sim_params, "use_gpu_pipeline") and sim_params.use_gpu_pipeline:
            # With GPU pipeline, always create a new simulation
            logger.info("GPU pipeline enabled, creating new simulation instance")
            return gym.create_sim(*args, **kwargs)

    # Without GPU pipeline, can reuse existing simulation
    if EXISTING_SIM is not None:
        return EXISTING_SIM
    else:
        EXISTING_SIM = gym.create_sim(*args, **kwargs)
        return EXISTING_SIM


class Env(ABC):
    def __init__(
        self,
        config: Dict[str, Any],
        rl_device: str,
        sim_device: str,
        graphics_device_id: int,
        headless: bool,
    ):
        """Initialize the environment.

        Args:
            config: the configuration dictionary.
            sim_device: the device to simulate physics on. eg. 'cuda:0' or 'cpu'
            graphics_device_id: the device ID to render with.
            headless: Set to False to disable viewer rendering.
        """
        split_device = sim_device.split(":")
        self.device_type = split_device[0]
        self.device_id = int(split_device[1]) if len(split_device) > 1 else 0

        # Fail fast if deprecated use_gpu_pipeline key is present
        if "use_gpu_pipeline" in config.get("sim", {}):
            raise RuntimeError(
                "The 'use_gpu_pipeline' config key is deprecated and must be removed. "
                "GPU pipeline is now automatically determined from sim_device. "
                "Use sim_device='cuda:0' for GPU pipeline or sim_device='cpu' for CPU pipeline."
            )

        # Automatically determine GPU pipeline based on sim_device
        # This provides a single source of truth for device configuration
        if self.device_type.lower() == "cuda" or self.device_type.lower() == "gpu":
            # Enable GPU pipeline for CUDA devices
            self.use_gpu_pipeline = True
            self.device = sim_device
            logger.info(f"Using GPU pipeline with device {sim_device}")
        else:
            # Disable GPU pipeline for CPU devices
            self.use_gpu_pipeline = False
            self.device = "cpu"
            logger.info("Using CPU pipeline with device cpu")

        # Set the internal config for compatibility with existing code
        if "sim" not in config:
            config["sim"] = {}
        config["sim"]["use_gpu_pipeline"] = self.use_gpu_pipeline

        self.rl_device = rl_device
        self.graphics_device_id = graphics_device_id
        self.headless = headless
        self.sim_device_id = self.device_id
        self.cfg = config

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.num_envs = self.cfg["env"]["numEnvs"]
        self._num_observations = 0  # Use private variable
        self._num_actions = 0  # Use private variable
        self.num_states = 0

        # Observation and action spaces
        self.obs_space = {}
        self.act_space = {}

        # Clip observations and actions
        self.clip_obs = self.cfg["env"].get("clipObservations", np.Inf)
        self.clip_actions = self.cfg["env"].get("clipActions", np.Inf)

        # Set random seed
        self.seed(self.cfg.get("seed", 42))

    @property
    def num_observations(self):
        """Number of observations. Fail-fast if accessed before initialization."""
        if self._num_observations == 0:
            raise RuntimeError(
                "num_observations accessed before initialization. "
                "This value is only available after the environment is fully initialized "
                "and reset() has been called at least once."
            )
        return self._num_observations

    @num_observations.setter
    def num_observations(self, value):
        """Set the number of observations."""
        self._num_observations = value

    @property
    def num_actions(self):
        """Number of actions. Fail-fast if accessed before initialization."""
        if self._num_actions == 0:
            raise RuntimeError(
                "num_actions accessed before initialization. "
                "This value is only available after the environment is fully initialized."
            )
        return self._num_actions

    @num_actions.setter
    def num_actions(self, value):
        """Set the number of actions."""
        self._num_actions = value

    def seed(self, seed=None):
        if seed is None:
            return

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def step(self, actions):
        """Take a step in the environment."""
        raise NotImplementedError

    def reset(self):
        """Reset the environment."""
        raise NotImplementedError

    def render(self, mode="human"):
        """Render the environment."""
        raise NotImplementedError

    def close(self):
        """Close the environment."""
        raise NotImplementedError


class VecTask(Env):
    def __init__(
        self,
        config: Dict[str, Any],
        rl_device: str,
        sim_device: str,
        graphics_device_id: int,
        headless: bool,
        virtual_screen_capture: bool = False,
        force_render: bool = False,
    ):
        """Initialize the vectorized task.

        Args:
            config: the configuration dictionary.
            sim_device: the device to simulate physics on. eg. 'cuda:0' or 'cpu'
            graphics_device_id: the device ID to render with.
            headless: Set to False to disable viewer rendering.
            virtual_screen_capture: Set to True to allow the users get captured screen in RGB array.
            force_render: Set to True to always force rendering in the steps.
        """
        super().__init__(config, rl_device, sim_device, graphics_device_id, headless)
        self.virtual_screen_capture = virtual_screen_capture
        self.virtual_display = None
        if self.virtual_screen_capture:
            from pyvirtualdisplay.smartdisplay import SmartDisplay

            self.virtual_display = SmartDisplay(size=SCREEN_CAPTURE_RESOLUTION)
            self.virtual_display.start()
        self.force_render = force_render

        self.sim_params = self._parse_sim_params(
            self.cfg["physics_engine"], self.cfg["sim"], self.num_envs
        )
        if self.cfg["physics_engine"] == "physx":
            self.physics_engine = gymapi.SIM_PHYSX
        elif self.cfg["physics_engine"] == "flex":
            self.physics_engine = gymapi.SIM_FLEX
        else:
            msg = f"Invalid physics engine backend: {self.cfg['physics_engine']}"
            raise ValueError(msg)

        self.dt = self.sim_params.dt

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        self.gym = gymapi.acquire_gym()

        # Setup sim
        self.sim = None
        self.envs = []

        # Initialize buffers
        self.num_dof = 0
        self.obs_buf = None
        self.states_buf = None
        self.rew_buf = None
        self.reset_buf = None
        self.episode_step_count = None
        self.extras = {}

    def _parse_sim_params(
        self, physics_engine: str, config_sim: Dict[str, Any], num_envs: int
    ) -> gymapi.SimParams:
        """Parse the sim configuration.

        Args:
            physics_engine: which physics engine to use.
            config_sim: sim configuration dictionary.
            num_envs: number of parallel environments.
        Returns:
            SimParams object.
        """
        sim_params = gymapi.SimParams()

        # assign general sim parameters
        sim_params.dt = config_sim["dt"]
        sim_params.num_client_threads = config_sim.get("num_client_threads", 0)
        sim_params.use_gpu_pipeline = config_sim["use_gpu_pipeline"]
        sim_params.substeps = config_sim.get("substeps", 2)

        # Hard-code up-axis to Z (required by all DexHand environments)
        sim_params.up_axis = gymapi.UP_AXIS_Z

        # assign gravity
        sim_params.gravity = gymapi.Vec3(*config_sim["gravity"])

        # configure physics parameters
        if physics_engine == "physx":
            # set the parameters
            if "physx" in config_sim:
                for opt in config_sim["physx"].keys():
                    if opt == "contact_collection":
                        setattr(
                            sim_params.physx,
                            opt,
                            gymapi.ContactCollection(config_sim["physx"][opt]),
                        )
                    elif opt == "gpu_contact_pairs_per_env":
                        # Calculate max_gpu_contact_pairs automatically
                        pairs_per_env = config_sim["physx"][opt]
                        total_pairs = pairs_per_env * num_envs
                        setattr(sim_params.physx, "max_gpu_contact_pairs", total_pairs)
                        logger.info(
                            f"Auto-calculated max_gpu_contact_pairs: {pairs_per_env} per env * {num_envs} envs = {total_pairs}"
                        )
                    else:
                        setattr(sim_params.physx, opt, config_sim["physx"][opt])

            # Critical GPU pipeline parameters (as per troubleshooting guide)
            if config_sim["use_gpu_pipeline"]:
                # These parameters are essential for GPU pipeline stability
                if not hasattr(sim_params.physx, "contact_collection"):
                    sim_params.physx.contact_collection = (
                        gymapi.ContactCollection.CC_LAST_SUBSTEP
                    )
                if not hasattr(sim_params.physx, "default_buffer_size_multiplier"):
                    sim_params.physx.default_buffer_size_multiplier = 1.0
                if not hasattr(sim_params.physx, "max_gpu_contact_pairs"):
                    raise ValueError(
                        "max_gpu_contact_pairs not set. Please specify 'gpu_contact_pairs_per_env' in the physx config."
                    )
                if not hasattr(sim_params.physx, "always_use_articulations"):
                    sim_params.physx.always_use_articulations = True
        elif physics_engine == "flex":
            # set the parameters
            if "flex" in config_sim:
                for opt in config_sim["flex"].keys():
                    setattr(sim_params.flex, opt, config_sim["flex"][opt])

        # return the configured params
        return sim_params

    def create_sim(self):
        """Create the simulation."""
        self.sim = _create_sim_once(
            self.gym,
            self.sim_device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        if self.sim is None:
            logger.error("Failed to create sim")
            quit()

        return self.sim

    def set_viewer(self):
        """Create the viewer - overridden by subclasses that manage their own viewer."""
        pass

    def pre_physics_step(self, actions):
        """Apply actions before physics step."""
        raise NotImplementedError

    def post_physics_step(self):
        """Process observations, rewards, etc. after physics step."""
        raise NotImplementedError

    def render(self, mode="rgb_array"):
        """Draw the frame to the viewer, and check for keyboard events."""
        # Subclasses should implement viewer rendering if needed
        raise NotImplementedError(
            "Subclasses must implement render() if they support rendering"
        )

    def close(self):
        """Close the environment."""
        if self.sim:
            self.gym.destroy_sim(self.sim)
            self.sim = None
