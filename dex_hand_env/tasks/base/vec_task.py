"""
Base class for vectorized tasks in Isaac Gym.

This module defines the VecTask abstract base class that all tasks should inherit from.
"""

import os
import sys
import time
import abc
from abc import ABC
from datetime import datetime
from os.path import join
from collections import deque
from copy import deepcopy
from typing import Dict, Any, Tuple, List, Set

import gym
from gym import spaces
import numpy as np
import torch
import operator
import random
from isaacgym import gymtorch, gymapi


# Global variable to store the simulation instance
EXISTING_SIM = None
SCREEN_CAPTURE_RESOLUTION = (1027, 768)


def _create_sim_once(gym, *args, **kwargs):
    global EXISTING_SIM
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

        self.device = "cpu"
        if config["sim"]["use_gpu_pipeline"]:
            if self.device_type.lower() == "cuda" or self.device_type.lower() == "gpu":
                self.device = "cuda" + ":" + str(self.device_id)
            else:
                print("GPU Pipeline is enabled but the device is not CUDA. Using CPU instead.")
                self.device = "cpu"
        else:
            self.device = "cpu"

        self.rl_device = rl_device
        self.graphics_device_id = graphics_device_id
        self.headless = headless
        self.sim_device_id = self.device_id
        self.cfg = config

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.num_envs = self.cfg["env"]["numEnvs"]
        self.num_observations = 0
        self.num_actions = 0
        self.num_states = 0

        # Observation and action spaces
        self.obs_space = {}
        self.act_space = {}

        # Clip observations and actions
        self.clip_obs = self.cfg["env"].get("clipObservations", np.Inf)
        self.clip_actions = self.cfg["env"].get("clipActions", np.Inf)

        # Set random seed
        self.seed(self.cfg.get("seed", 42))

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
            self.cfg["physics_engine"], self.cfg["sim"]
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
        self.viewer = None

        # Control frequency vs physics simulation frequency
        self.control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)

        # Initialize buffers
        self.num_dof = 0
        self.obs_buf = None
        self.states_buf = None
        self.rew_buf = None
        self.reset_buf = None
        self.progress_buf = None
        self.extras = {}

        # Rendering
        self.enable_viewer_sync = True
        self.viewer = None
        self.last_frame_time = time.time()
        self.render_fps = -1

    def _parse_sim_params(self, physics_engine: str, config_sim: Dict[str, Any]) -> gymapi.SimParams:
        """Parse the sim configuration.

        Args:
            physics_engine: which physics engine to use.
            config_sim: sim configuration dictionary.
        Returns:
            SimParams object.
        """
        sim_params = gymapi.SimParams()

        # check correct up-axis
        if config_sim["up_axis"] not in ["z", "y"]:
            msg = f"Invalid up axis: {config_sim['up_axis']}"
            print(msg)
            raise ValueError(msg)

        # assign general sim parameters
        sim_params.dt = config_sim["dt"]
        sim_params.num_client_threads = config_sim.get("num_client_threads", 0)
        sim_params.use_gpu_pipeline = config_sim["use_gpu_pipeline"]
        sim_params.substeps = config_sim.get("substeps", 2)

        # assign up-axis
        if config_sim["up_axis"] == "z":
            sim_params.up_axis = gymapi.UP_AXIS_Z
        else:
            sim_params.up_axis = gymapi.UP_AXIS_Y

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
                    else:
                        setattr(sim_params.physx, opt, config_sim["physx"][opt])
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
            self.sim_params
        )
        if self.sim is None:
            print("*** Failed to create sim")
            quit()

        return self.sim

    def set_viewer(self):
        """Create the viewer."""
        # if running with a viewer, set up keyboard shortcuts and camera
        if not self.headless:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_R, "record_frames"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_SPACE, "toggle_random_actions"
            )

            # set the camera position based on up axis
            sim_params = self.gym.get_sim_params(self.sim)
            if sim_params.up_axis == gymapi.UP_AXIS_Z:
                cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
                cam_target = gymapi.Vec3(10.0, 15.0, 0.0)
            else:
                cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
                cam_target = gymapi.Vec3(10.0, 0.0, 15.0)

            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def pre_physics_step(self, actions):
        """Apply actions before physics step."""
        raise NotImplementedError

    def post_physics_step(self):
        """Process observations, rewards, etc. after physics step."""
        raise NotImplementedError

    def render(self, mode="rgb_array"):
        """Draw the frame to the viewer, and check for keyboard events."""
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                elif evt.action == "record_frames" and evt.value > 0:
                    self.record_frames = not self.record_frames
                elif evt.action == "toggle_random_actions" and evt.value > 0:
                    self.random_actions = not self.random_actions

            # fetch results
            if self.device != "cpu":
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)

                # Wait for dt to elapse in real time.
                # This synchronizes the physics simulation with the rendering rate.
                self.gym.sync_frame_time(self.sim)

                # Slow down rendering to real time
                now = time.time()
                delta = now - self.last_frame_time
                if self.render_fps < 0:
                    # render at control frequency
                    render_dt = self.dt * self.control_freq_inv
                else:
                    render_dt = 1.0 / self.render_fps

                if delta < render_dt:
                    time.sleep(render_dt - delta)

                self.last_frame_time = time.time()

            else:
                self.gym.poll_viewer_events(self.viewer)

            if self.virtual_display and mode == "rgb_array":
                img = self.virtual_display.grab()
                return np.array(img)

    def close(self):
        """Close the environment."""
        if self.viewer:
            self.gym.destroy_viewer(self.viewer)
            self.viewer = None
        
        if self.sim:
            self.gym.destroy_sim(self.sim)
            self.sim = None