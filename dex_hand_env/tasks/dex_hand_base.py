"""
Base class for DexHand tasks.

This module provides the DexHandBase class that uses composition
to delegate functionality to specialized components.
"""

import os
import sys
import time
import abc
import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Set
from abc import ABC
from datetime import datetime
from os.path import join
from collections import deque
from copy import deepcopy

# Import Gym
import gym
from gym import spaces

# Import IsaacGym first
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import to_torch, tensor_clamp, quat_mul, quat_conjugate

# Then import PyTorch
import torch

# Import our utilities
from dex_hand_env.utils.torch_jit_utils import quat_to_euler, quat_from_euler, axisangle2quat

# Import components
from dex_hand_env.components.camera_controller import CameraController
from dex_hand_env.components.fingertip_visualizer import FingertipVisualizer
from dex_hand_env.components.success_failure_tracker import SuccessFailureTracker
from dex_hand_env.components.reward_calculator import RewardCalculator

# Import utilities
from dex_hand_env.utils.coordinate_transforms import point_in_hand_frame

# Global variable to share sim instance
EXISTING_SIM = None

def _create_sim_once(gym, *args, **kwargs):
    global EXISTING_SIM
    if EXISTING_SIM is not None:
        return EXISTING_SIM
    else:
        EXISTING_SIM = gym.create_sim(*args, **kwargs)
        return EXISTING_SIM

# Import task interface
from dex_hand_env.tasks.task_interface import DexTask
# Import base task class
from dex_hand_env.tasks.base.vec_task import VecTask


class DexHandBase(VecTask):
    """
    Base class for DexHand tasks that implements common functionality for all dexterous hand tasks.
    
    This class provides a unified framework for dexterous manipulation tasks using the DexHand
    robot model. It uses composition to delegate functionality to specialized components:
    
    - CameraController: Handles camera control and keyboard shortcuts
    - FingertipVisualizer: Visualizes fingertip contacts with color
    - SuccessFailureTracker: Tracks success and failure criteria
    - RewardCalculator: Calculates rewards
    
    The task-specific functionality is provided by a DexTask implementation that is
    passed to the constructor.
    """
    
    def __init__(
        self,
        cfg,
        task: DexTask,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture=False,
        force_render=False,
    ):
        """
        Initialize the DexHandBase environment.
        
        Args:
            cfg: Configuration dictionary.
            task: Task implementation (must implement DexTask interface).
            rl_device: The device to use for PyTorch operations.
            sim_device: The device to use for simulation.
            graphics_device_id: The device ID to use for rendering.
            headless: If True, don't render to a window.
            virtual_screen_capture: If True, allow capturing screen for rgb_array mode.
            force_render: If True, always render in the steps.
        """
        print("\n===== DEXHAND ENVIRONMENT INITIALIZATION =====")
        print(f"RL Device: {rl_device}")
        print(f"Sim Device: {sim_device}")
        print(f"Graphics Device ID: {graphics_device_id}")
        print(f"Headless Mode: {headless}")
        print(f"Force Render: {force_render}")
        print("=============================================\n")
        self.cfg = cfg
        self.task = task
        self.asset_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets")
        
        # Set device early for tensor creation
        self.device = rl_device
        
        # Task specific parameters
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        
        # Number of physics steps per control step (auto-computed based on reset requirements)
        # Default to 1 physics step per control step, but this will be adjusted automatically
        # if needed for reset stability
        self.physics_steps_per_control_step = 1
        
        # Log deprecation warning if controlFrequencyInv is provided
        if "controlFrequencyInv" in self.cfg["env"]:
            import logging
            logging.warning("controlFrequencyInv is deprecated and will be ignored. "
                          "The environment will automatically compute the number of "
                          "physics steps per control step based on reset stability requirements.")
        
        # Action scaling and bias will be calculated automatically from joint limits
        
        # Physics parameters
        self.physics_dt = self.cfg["sim"].get("dt", 0.01)  # Physics simulation timestep
        self.control_dt = self.physics_dt * self.physics_steps_per_control_step  # Control timestep
        
        # Policy velocity limits (in rad/s for rotations and m/s for translations)
        # These limits restrict how fast the policy can command the robot to move,
        # which is often more conservative than the physical limits in the URDF/MJCF
        
        # Always use custom velocity limits to ensure safe operation
        self.use_custom_velocity_limits = True
        
        # Finger joint velocity limits (rad/s)
        self.policy_finger_velocity_limit = self.cfg["env"].get("maxFingerVelocity", 2.0)
        
        # Base DOF velocity limits (translations in m/s, rotations in rad/s)
        self.policy_base_lin_velocity_limit = self.cfg["env"].get("maxBaseLinearVelocity", 1.0)
        self.policy_base_ang_velocity_limit = self.cfg["env"].get("maxBaseAngularVelocity", 1.5)
        
        # PD controller parameters for position control
        # These determine how aggressively the robot tries to reach target positions
        self.base_stiffness = self.cfg["env"].get("baseStiffness", 400.0)
        self.base_damping = self.cfg["env"].get("baseDamping", 40.0)
        self.finger_stiffness = self.cfg["env"].get("fingerStiffness", 100.0)
        self.finger_damping = self.cfg["env"].get("fingerDamping", 10.0)
        
        # Initial hand pose parameters (configurable by derived tasks)
        # Default places hand at (0,0,0.5) with identity rotation, allowing tasks to place
        # tabletop/ground objects at z=0 without initial penetration
        self.initial_hand_pos = self.cfg["env"].get("initialHandPos", [0.0, 0.0, 0.5])
        self.initial_hand_rot = self.cfg["env"].get("initialHandRot", [0.0, 0.0, 0.0, 1.0])
        
        # Model paths
        self.hand_asset_file = "dexrobot_mujoco/dexrobot_mujoco/models/dexhand021_right_simplified_floating.xml"
        
        # Verify asset file exists
        asset_full_path = os.path.join(self.asset_root, self.hand_asset_file)
        if not os.path.exists(asset_full_path):
            print(f"ERROR: Hand asset file not found at {asset_full_path}")
            print(f"Asset directory contents:")
            if os.path.exists(self.asset_root):
                print(f"Asset root directory exists at {self.asset_root}")
                print(f"Contents of {self.asset_root}:")
                for root, dirs, files in os.walk(self.asset_root, topdown=True, followlinks=True):
                    for name in dirs:
                        print(os.path.join(root, name))
                    for name in files:
                        if name.endswith(".xml"):
                            print(os.path.join(root, name))
        
        # Define joint names
        self.base_joint_names = [
            "ARTx", "ARTy", "ARTz", "ARRx", "ARRy", "ARRz"
        ]
        
        self.finger_joint_names = [
            "r_f_joint1_1", "r_f_joint1_2", "r_f_joint1_3", "r_f_joint1_4",
            "r_f_joint2_1", "r_f_joint2_2", "r_f_joint2_3", "r_f_joint2_4",
            #"r_f_joint3_1",  # This is a fixed joint
            "r_f_joint3_2", "r_f_joint3_3", "r_f_joint3_4",
            "r_f_joint4_1", "r_f_joint4_2", "r_f_joint4_3", "r_f_joint4_4",
            "r_f_joint5_1", "r_f_joint5_2", "r_f_joint5_3", "r_f_joint5_4"
        ]
        
        # Map from hardware active DoFs to full finger joint space
        # based on HardwareMapping from dexhand_ros.py
        self.active_joint_mapping = {
            "th_dip": ["r_f_joint1_3", "r_f_joint1_4"],
            "th_mcp": ["r_f_joint1_2"],
            "th_rot": ["r_f_joint1_1"],
            "ff_spr": ["r_f_joint2_1", "r_f_joint4_1", "r_f_joint5_1"],
            "ff_dip": ["r_f_joint2_3", "r_f_joint2_4"],
            "ff_mcp": ["r_f_joint2_2"],
            "mf_dip": ["r_f_joint3_3", "r_f_joint3_4"],
            "mf_mcp": ["r_f_joint3_2"],
            "rf_dip": ["r_f_joint4_3", "r_f_joint4_4"],
            "rf_mcp": ["r_f_joint4_2"],
            "lf_dip": ["r_f_joint5_3", "r_f_joint5_4"],
            "lf_mcp": ["r_f_joint5_2"]
        }
        
        # Create reverse mapping from joint name to controller
        self.joint_to_control = {}
        for control, joints in self.active_joint_mapping.items():
            for joint in joints:
                self.joint_to_control[joint] = control
        
        # Active joint names (12 DoFs that can be controlled directly)
        self.active_joint_names = list(self.active_joint_mapping.keys())
        
        # Body names for fingertips and fingerpads in the MJCF model
        self.fingertip_body_names = [
            "r_f_link1_tip", "r_f_link2_tip", "r_f_link3_tip", 
            "r_f_link4_tip", "r_f_link5_tip"
        ]
        
        self.fingerpad_body_names = [
            "r_f_link1_pad", "r_f_link2_pad", "r_f_link3_pad", 
            "r_f_link4_pad", "r_f_link5_pad"
        ]
        
        # Configure observation and action spaces
        self.num_observations = 0
        self.obs_keys = []
        
        # Set control mode (read from either controlMode or controlType for backward compatibility)
        if "controlMode" in self.cfg["env"]:
            self.action_control_mode = self.cfg["env"]["controlMode"]  # position or position_delta
        elif "controlType" in self.cfg["env"]:
            # Map controlType to our internal control mode
            control_type = self.cfg["env"]["controlType"]
            if control_type == "joint_pos":
                self.action_control_mode = "position_delta"  # Most common mode
            else:
                self.action_control_mode = "position"  # Default for other modes
            print(f"Using controlType={control_type}, mapped to action_control_mode={self.action_control_mode}")
        else:
            self.action_control_mode = "position"  # Default
        
        # Define constants for action dimensions
        self.NUM_BASE_DOFS = 6
        self.NUM_ACTIVE_FINGER_DOFS = 12
        
        # Control options for hand base and fingers
        # These options determine which DOFs are controlled by the policy vs. by the environment
        # - If controlHandBase=True, the policy controls the 6 base DOFs (3 translation, 3 rotation)
        # - If controlFingers=True, the policy controls the 12 active finger DOFs
        # - For any DOFs not controlled by the policy, the environment will use default targets
        #   (defaultBaseTargets, defaultFingerTargets) or targets provided by the task
        self.control_hand_base = self.cfg["env"].get("controlHandBase", True)
        self.control_fingers = self.cfg["env"].get("controlFingers", True)
        
        # Default target positions for uncontrolled DOFs defined in config
        # Note: These are only used as fallbacks if the task doesn't provide targets
        # through the get_task_dof_targets method
        if "defaultBaseTargets" in self.cfg["env"]:
            self.default_base_targets = torch.tensor(
                self.cfg["env"]["defaultBaseTargets"],
                device=self.device
            )
        else:
            # Initial position as default
            self.default_base_targets = torch.tensor(
                [0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                device=self.device
            )
            
        if "defaultFingerTargets" in self.cfg["env"]:
            self.default_finger_targets = torch.tensor(
                self.cfg["env"]["defaultFingerTargets"],
                device=self.device
            )
        else:
            # Zeros as default
            self.default_finger_targets = torch.zeros(
                self.NUM_ACTIVE_FINGER_DOFS,
                device=self.device
            )
        
        # Validate control options - at least one must be True
        if not self.control_hand_base and not self.control_fingers:
            raise ValueError("At least one of controlHandBase or controlFingers must be True")
            
        # Save control settings for later (we'll set up action space after parent init)
        print(f"Control settings: base={self.control_hand_base}, fingers={self.control_fingers}")
        
        # Important: Initialize these variables before base class to avoid NoneType errors
        self.hand_handles = []
        self.fingertip_body_handles = []
        
        # Flag to track initialization state
        self._tensors_initialized = False
        
        # Initialize the base class
        print("Initializing VecTask parent class...")
        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )
        print("VecTask parent class initialized, creating simulation...")
        
        # Check if initialization has already been done by the parent class
        already_initialized = (hasattr(self, 'sim') and self.sim is not None and 
                              hasattr(self, 'dof_state') and self.dof_state is not None)
        
        if already_initialized:
            print("Environment already initialized by parent class")
        else:
            # We need to explicitly create and initialize everything
            print("Environment not fully initialized by parent class, initializing now...")
            if not hasattr(self, 'sim') or self.sim is None:
                self.sim = self.create_sim()
                
            # We should NOT create environments again - the parent class VecTask already did this
            # This was causing the duplicate assets and DOF count mismatch
            
            # Just make sure tensor handles are acquired and tensors are set up
            if not hasattr(self, 'dof_state') or self.dof_state is None:
                print("Acquiring tensor handles...")
                self._acquire_tensor_handles()
                
            if not hasattr(self, 'dof_pos') or self.dof_pos is None:
                print("Setting up tensors...")
                self._setup_tensors()
            
            print("Creating components...")
            self._create_components()
        
        # Verify tensors were properly initialized
        if not hasattr(self, 'dof_pos') or self.dof_pos is None or self.dof_pos.shape[1] == 0:
            raise ValueError("DOF position tensor was not properly initialized. The simulation initialization has failed.")
        
        # Mark tensors as initialized
        self._tensors_initialized = True
        
        # After parent initialization, define our action space
        # This ensures it won't be overridden by the parent class
        
        # Define action space based on control mode
        self.num_actions = 0
            
        # Calculate action space size
        if self.control_hand_base:
            print(f"Adding {self.NUM_BASE_DOFS} base DOFs to action space")
            self.num_actions += self.NUM_BASE_DOFS
        if self.control_fingers:
            print(f"Adding {self.NUM_ACTIVE_FINGER_DOFS} finger DOFs to action space")
            self.num_actions += self.NUM_ACTIVE_FINGER_DOFS
            
        print(f"Final action space size: {self.num_actions}")
        
        # Fail fast if num_actions is 0 (should never happen due to validation at start)
        if self.num_actions == 0:
            raise ValueError("Action space is empty (num_actions = 0). This should never happen as at least one control option must be enabled.")
        
        # Initialize sensor data
        self.contact_forces = torch.zeros(
            (self.num_envs, 5, 3), device=self.device, dtype=torch.float
        )
        
        # Previous action buffer for incremental control - always store active DOF target space (6+12=18D)
        # Initialize with the default base position for all environments to prevent downward drift
        self.prev_active_targets = torch.zeros(
            (self.num_envs, self.NUM_BASE_DOFS + self.NUM_ACTIVE_FINGER_DOFS), 
            device=self.device, dtype=torch.float
        )
        
        # Initialize base position targets with initial_hand_pos
        self.prev_active_targets[:, 0] = self.initial_hand_pos[0]  # ARTx
        self.prev_active_targets[:, 1] = self.initial_hand_pos[1]  # ARTy
        self.prev_active_targets[:, 2] = self.initial_hand_pos[2]  # ARTz - most critical for gravity issues
        # Initialize rotation targets - default is identity quaternion which is [0,0,0] in axis-angle
        self.prev_active_targets[:, 3:6] = 0.0  # ARRx, ARRy, ARRz
        
        # Initialize observation dict
        self.obs_dict = {}
        
        # Track physics steps to automatically determine steps per control
        self.physics_step_count = 0
        self.last_control_step_count = 0
        
        # Initialize action tracking for rewards
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.prev_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        
        # Initialize velocity and acceleration tracking for rewards
        self.prev_hand_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_hand_ang_vel = torch.zeros((self.num_envs, 3), device=self.device)
        
        # Use num_dof which should be properly set during tensor initialization
        # If dof_pos is not initialized yet, we should fail fast rather than using a fallback
        if not hasattr(self, 'dof_pos') or self.dof_pos is None:
            raise ValueError("DOF position tensor not initialized before creating tracking tensors")
            
        self.prev_dof_vel = torch.zeros((self.num_envs, self.dof_pos.shape[1]), device=self.device)
        
        # Initialize contact tracking (will be properly set during first observation)
        self.prev_contacts = torch.zeros((self.num_envs, 5), device=self.device, dtype=torch.bool)
        
        # Now that all tensors are initialized, compute initial observations to get the observation space
        self._compute_observations_buffer()
    
    def _create_components(self):
        """Create components for the environment."""
        # Create camera controller with proper error handling
        print("Creating camera controller...")
        # Only create camera controller in graphical mode
        if self.viewer is not None:
            self.camera_controller = CameraController(
                self.gym, 
                self.viewer, 
                self.envs, 
                self.num_envs, 
                self.device
            )
            
            # Explicitly subscribe keyboard events
            self.camera_controller.subscribe_keyboard_events()
            print("Camera controller created successfully!")
        else:
            # In headless mode, camera controller is not required
            print("No viewer available, running in headless mode")
            self.camera_controller = None
        
        # Create fingertip visualizer - required for fingertip tracking
        print("Creating fingertip visualizer...")
        if not hasattr(self, 'fingertip_body_handles') or len(self.fingertip_body_handles) == 0:
            raise ValueError("Fingertip body handles are not available. The hand model may not have loaded correctly.")
        
        self.fingertip_visualizer = FingertipVisualizer(
            self.gym,
            self.envs,
            self.hand_handles,
            self.fingertip_body_handles,
            self.device
        )
        print("Fingertip visualizer created successfully!")
        
        # Create success/failure tracker - required for episode termination
        print("Creating success/failure tracker...")
        self.success_failure_tracker = SuccessFailureTracker(
            self.num_envs,
            self.device,
            self.cfg
        )
        print("Success/failure tracker created successfully!")
        
        # Create reward calculator - required for reward computation
        print("Creating reward calculator...")
        self.reward_calculator = RewardCalculator(
            self.num_envs,
            self.device,
            self.cfg
        )
        print("Reward calculator created successfully!")
    
    def set_viewer(self):
        """Create the viewer with customized settings for a more appealing visualization.
        
        This method overrides the parent class method to:
        1. Set up a more suitable camera position for the hand
        2. Add keyboard shortcuts for controlling the environment
        3. Configure a blue background effect with lighting
        """
        # If running with a viewer, set up keyboard shortcuts and camera
        if not self.headless:
            # Create viewer with custom camera properties
            camera_props = gymapi.CameraProperties()
            # Set camera FOV, clipping planes, etc. for better visualization
            camera_props.horizontal_fov = 75.0
            camera_props.width = 1280
            camera_props.height = 720
            
            # Create the viewer with these properties
            self.viewer = self.gym.create_viewer(self.sim, camera_props)
            
            if self.viewer is None:
                print("Failed to create viewer")
                return
                
            # Subscribe to keyboard events
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "reset_all")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SPACE, "pause")
            
            # Set a good default camera position for viewing the hand
            # Position camera slightly above and to the front-right of the hand
            cam_pos = gymapi.Vec3(1.0, 0.5, 0.8)  # x: forward, y: right, z: up
            # Look at the origin (where the hand will be)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.5)  
            
            # Apply camera settings
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
            
            # We won't set debug visualization flags for aesthetic view
            # Just use the default rendering settings for better appearance
            
            print("Viewer created successfully with custom camera settings")
            
    def create_sim(self):
        """
        Create the simulation and initialize environments.
        
        This method overrides the base class method to properly sequence the initialization:
        1. Create simulation
        2. Create environments
        3. Acquire tensor handles
        4. Setup tensors
        5. Create components
        
        Returns:
            Simulation instance
        """
        # Check if simulation already exists
        if hasattr(self, 'sim') and self.sim is not None:
            print("Simulation already exists, skipping creation")
            return self.sim
            
        print("\n====== DETAILED INITIALIZATION SEQUENCE ======")
        print(f"Asset root: {self.asset_root}")
        print(f"Device: {self.device}")
        print(f"Number of environments: {self.num_envs}")
        print(f"Physics steps per control step: {self.physics_steps_per_control_step}")
        print(f"Physics timestep: {self.physics_dt}")
        print("Starting create_sim method...")
        
        # Print debug info about the asset path
        print(f"Asset root path: {self.asset_root}")
        print(f"Hand asset file: {self.hand_asset_file}")
        full_asset_path = os.path.join(self.asset_root, self.hand_asset_file)
        print(f"Full asset path: {full_asset_path}")
        print(f"Asset exists: {os.path.exists(full_asset_path)}")
        
        # Call parent to create simulation
        print("\n[1] Creating physics simulation...")
        try:
            # Create a new simulation instance instead of using shared one
            print("Creating new simulation instance")
            self.sim = self.gym.create_sim(
                self.sim_device_id,
                self.graphics_device_id,
                self.physics_engine,
                self.sim_params
            )
            print("Simulation created successfully!")
            print(f"Simulation handle: {self.sim}")
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to create simulation: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError("Simulation creation failed - cannot continue") from e
        
        # Create viewer first if not headless
        if not self.headless:
            print("\n[2] Creating viewer...")
            self.set_viewer()
            print("Viewer created!")
        
        # Load assets and create environments
        print("\n[3] Loading assets and creating environments...")
        try:
            # Check if environments already exist before creating them
            if hasattr(self, 'envs') and len(self.envs) == self.num_envs:
                print(f"Environments already exist (count: {len(self.envs)}), skipping creation")
            else:
                self._create_envs()
            print(f"Successfully created {len(self.envs)} environments")
            
            # Verify environment creation
            if len(self.envs) != self.num_envs:
                raise ValueError(f"Failed to create all environments. Created {len(self.envs)}, expected {self.num_envs}")
                
            # Verify hand actors were created
            if len(self.hand_handles) != self.num_envs:
                raise ValueError(f"Failed to create all hand actors. Created {len(self.hand_handles)}, expected {self.num_envs}")
        except Exception as e:
            print(f"CRITICAL ERROR in environment creation: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError("Failed to create environments properly - cannot continue initialization") from e
        
        # Step physics a few times to initialize tensors
        print("\n[4] Stepping physics to initialize tensors...")
        for i in range(4):
            print(f"Physics step {i+1}/4")
            try:
                self.gym.simulate(self.sim)
            except Exception as e:
                print(f"CRITICAL ERROR during initial physics steps: {e}")
                import traceback
                traceback.print_exc()
                raise RuntimeError("Failed to simulate physics - cannot continue initialization") from e
        print("Initial physics steps completed")
        
        # Acquire tensor handles after environments are created and simulation is stepped
        print("\n[5] Acquiring tensor handles...")
        try:
            self._acquire_tensor_handles()
            print("Tensor handles acquired successfully")
            
            # Verify handle acquisition
            if not hasattr(self, 'dof_state') or self.dof_state is None or self.dof_state.numel() == 0:
                raise ValueError("DOF state tensor is empty after acquisition")
        except Exception as e:
            print(f"CRITICAL ERROR acquiring tensor handles: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError("Failed to acquire tensor handles - cannot continue initialization") from e
        
        # Setup tensors (needed by components)
        print("\n[6] Setting up tensors...")
        try:
            self._setup_tensors()
            print("Tensors set up successfully")
            
            # Verify tensor setup
            if not hasattr(self, 'dof_pos') or self.dof_pos.shape[1] == 0:
                raise ValueError("DOF position tensor has invalid shape after tensor setup")
        except Exception as e:
            print(f"CRITICAL ERROR setting up tensors: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError("Failed to set up tensors - cannot continue initialization") from e
        
        # Create components after tensors are set up
        print("\n[7] Creating components...")
        try:
            self._create_components()
            print("Components created successfully")
            
            # Verify components were created
            # Camera controller is optional in headless mode
            if not self.headless and (not hasattr(self, 'camera_controller') or self.camera_controller is None):
                raise ValueError("Camera controller was not properly initialized in graphical mode")
                
            if not hasattr(self, 'fingertip_visualizer') or self.fingertip_visualizer is None:
                raise ValueError("Fingertip visualizer was not properly initialized")
                
            if not hasattr(self, 'success_failure_tracker') or self.success_failure_tracker is None:
                raise ValueError("Success/failure tracker was not properly initialized")
                
            if not hasattr(self, 'reward_calculator') or self.reward_calculator is None:
                raise ValueError("Reward calculator was not properly initialized")
        except Exception as e:
            print(f"CRITICAL ERROR creating components: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError("Failed to create components - cannot continue initialization") from e
            
        # Step physics again to ensure everything is properly initialized
        print("\n[8] Final physics step...")
        try:
            self.gym.prepare_sim(self.sim)
            self.gym.simulate(self.sim)
        except Exception as e:
            print(f"Error in final physics step: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError("Failed in final physics step - cannot continue initialization") from e
        
        # Refresh tensors to ensure they're up to date
        print("\n[9] Refreshing tensor data...")
        try:
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            print("Tensor data refreshed successfully")
            
            # Verify tensors after refresh
            if self.dof_pos.shape[1] == 0:
                raise ValueError("DOF position tensor still has invalid shape after tensor refresh")
        except Exception as e:
            print(f"CRITICAL ERROR refreshing tensor data: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError("Failed to refresh tensor data - cannot continue initialization") from e
        
        print("\n[10] Initialization completed successfully!")
        return self.sim
    
    def _create_envs(self):
        """Create environments and load assets.
        
        This method creates the environments and loads assets for the hand.
        It should be called after create_sim but before _acquire_tensor_handles.
        """
        # Load hand asset
        asset_options = gymapi.AssetOptions()
        # For a hand model with a fixed base (not affected by gravity):
        # - The base_link of the actor is fixed to the world at the specified pose
        # - The ARTx, ARTy, ARTz, ARRx, ARRy, ARRz DOFs still exist in the model
        #   and allow control of the base joint's position/orientation
        # - But since the base is fixed, the hand won't fall due to gravity
        # - This is correct for our simulation where we want to control the hand
        #   through its internal DOFs, not by moving the entire actor
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.replace_cylinder_with_capsule = True
        # Remove unsupported option: enable_mesh_static_collision
        
        # Print asset file path for debugging
        full_asset_path = os.path.join(self.asset_root, self.hand_asset_file)
        print(f"Loading hand asset from: {full_asset_path}")
        
        # Verify asset file exists - this is a critical check
        if not os.path.exists(full_asset_path):
            print(f"CRITICAL ERROR: Hand asset file not found at {full_asset_path}")
            
            # Search for alternative asset files
            asset_dir = os.path.dirname(full_asset_path)
            if os.path.exists(asset_dir):
                print(f"Directory {asset_dir} exists, but file {os.path.basename(full_asset_path)} not found")
                print(f"Available XML files in directory:")
                import glob
                xml_files = glob.glob(os.path.join(asset_dir, "*.xml"))
                for xml_file in xml_files:
                    print(f" - {os.path.basename(xml_file)}")
                
                if len(xml_files) > 0:
                    # Suggest alternative
                    alt_file = os.path.basename(xml_files[0])
                    print(f"Suggestion: Try using '{alt_file}' instead of '{os.path.basename(self.hand_asset_file)}'")
            else:
                print(f"Directory {asset_dir} does not exist")
                
                # Check the asset root
                if os.path.exists(self.asset_root):
                    print(f"Asset root directory exists at {self.asset_root}")
                    print(f"Contents of asset root:")
                    for item in os.listdir(self.asset_root):
                        if os.path.isdir(os.path.join(self.asset_root, item)):
                            print(f" - Dir: {item}")
                        else:
                            print(f" - File: {item}")
                else:
                    print(f"Asset root directory does not exist at {self.asset_root}")
                    
            # This is a critical error - we cannot continue without the asset file
            raise FileNotFoundError(f"Hand asset file not found: {full_asset_path}")
            
        # Verify the asset file is readable
        try:
            with open(full_asset_path, 'r') as f:
                print(f"Successfully opened asset file {full_asset_path}")
                content = f.read()
                print(f"Asset file size: {len(content)} bytes")
                
                # Check if file seems valid
                if "mujoco" not in content[:100].lower():
                    print(f"WARNING: File does not appear to be a valid MuJoCo XML file. First 100 chars: {content[:100]}")
        except Exception as e:
            print(f"Error reading asset file: {e}")
            raise RuntimeError(f"Cannot read asset file: {e}")
        
        # Load the hand MJCF asset
        try:
            hand_asset = self.gym.load_asset(
                self.sim, 
                self.asset_root, 
                self.hand_asset_file, 
                asset_options
            )
            
            # Verify asset loaded correctly
            if hand_asset:
                # Print DOF count from the asset
                dof_count = self.gym.get_asset_dof_count(hand_asset)
                print(f"Hand asset loaded successfully with {dof_count} DOFs")
                
                # Print DOF names if possible
                dof_names = [self.gym.get_asset_dof_name(hand_asset, i) for i in range(dof_count)]
                print(f"DOF names: {dof_names}")
            else:
                raise ValueError("Hand asset is null after loading")
        except Exception as e:
            raise RuntimeError(f"Failed to load hand asset: {e}")
        
        # Add ground plane with standard parameters
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)  # z-up
        plane_params.distance = 0.0  # at origin
        plane_params.static_friction = 0.5
        plane_params.dynamic_friction = 0.5
        plane_params.restitution = 0.0
        # Add the ground with these parameters
        self.gym.add_ground(self.sim, plane_params)
        
        # Define the spacing between environments
        env_spacing = 1.0
        env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        
        # Create environments
        num_per_row = int(np.sqrt(self.num_envs))
        self.envs = []
        self.hand_handles = []
        
        # No LightParams in this version of IsaacGym, so we'll use the default lighting
        # We'll rely on the blue color settings in the viewer instead
        
        for i in range(self.num_envs):
            # Create env
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)
            
            # Add hand actor
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(self.initial_hand_pos[0], self.initial_hand_pos[1], self.initial_hand_pos[2])
            pose.r = gymapi.Quat(self.initial_hand_rot[0], self.initial_hand_rot[1], self.initial_hand_rot[2], self.initial_hand_rot[3])
            
            hand_handle = self.gym.create_actor(env, hand_asset, pose, f"hand_{i}", i, 0)
            self.hand_handles.append(hand_handle)
            
            # Set DOF properties for PD control
            dof_props = self.gym.get_actor_dof_properties(env, hand_handle)
            
            # Debug: Print original DOF properties
            print(f"Original DOF properties for env {i}:")
            for j in range(min(6, len(dof_props))):  # Just print the first 6 DOFs (base DOFs)
                print(f"  DOF {j}: driveMode={dof_props['driveMode'][j]}, stiffness={dof_props['stiffness'][j]}, damping={dof_props['damping'][j]}")
            
            # Set position control mode and much higher PD gains for all DOFs
            for j in range(len(dof_props)):
                # Set control mode to position
                dof_props['driveMode'][j] = gymapi.DOF_MODE_POS
                
                # Base DOFs (first 6 DOFs)
                if j < self.NUM_BASE_DOFS:
                    # Higher stiffness and damping for base translation DOFs (especially Z for gravity)
                    if j < 3:  # ARTx, ARTy, ARTz - translation DOFs
                        # Much higher values for translation stiffness to overcome gravity
                        dof_props['stiffness'][j] = 5000.0  # Very high stiffness
                        dof_props['damping'][j] = 200.0     # Higher damping
                        
                        # Extra stiffness for ARTz (vertical axis) to combat gravity
                        if j == 2:  # ARTz
                            dof_props['stiffness'][j] = 10000.0  # Extremely high stiffness for Z
                            dof_props['damping'][j] = 500.0      # High damping for stability
                    else:  # ARRx, ARRy, ARRz - rotation DOFs
                        dof_props['stiffness'][j] = 1000.0  # High stiffness for rotation
                        dof_props['damping'][j] = 100.0     # Moderate damping
                else:  # Finger DOFs
                    dof_props['stiffness'][j] = self.finger_stiffness
                    dof_props['damping'][j] = self.finger_damping
            
            # Debug: Print modified DOF properties
            print(f"Modified DOF properties for env {i}:")
            for j in range(min(6, len(dof_props))):  # Just print the first 6 DOFs (base DOFs)
                print(f"  DOF {j}: driveMode={dof_props['driveMode'][j]}, stiffness={dof_props['stiffness'][j]}, damping={dof_props['damping'][j]}")
            
            # Apply the DOF properties to the actor
            self.gym.set_actor_dof_properties(env, hand_handle, dof_props)
            
            # Allow the task to load its assets (after basic hand is loaded)
            if hasattr(self.task, 'load_env_assets'):
                self.task.load_env_assets(env, i)
    
    def _acquire_tensor_handles(self):
        """Acquire tensor handles from the simulation.
        
        This method should be called after the simulation and environments are created.
        It acquires tensor handles for DOF state, actor root state, and rigid body state.
        """
        # Get DOF state tensor
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        
        # Verify DOF state tensor is valid
        if self.dof_state.numel() == 0:
            raise ValueError("DOF state tensor is empty. This indicates the hand model did not load properly.")
        
        print(f"Acquired dof_state tensor with shape {self.dof_state.shape}")
        
        # DEBUGGING: Verify DOF names and order
        try:
            # Get all DOF names for the first environment
            print(f"\n===== VERIFYING DOF NAMES AND ORDER =====")
            for i in range(min(10, self.gym.get_actor_dof_count(self.envs[0], self.hand_handles[0]))):
                dof_name = self.gym.get_actor_dof_name(self.envs[0], self.hand_handles[0], i)
                print(f"DOF index {i}: {dof_name}")
            print("============================================\n")
        except Exception as e:
            print(f"Error getting DOF names: {e}")
        
        # Get actor root state tensor
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)
        
        # Verify root state tensor
        if self.root_state_tensor.numel() == 0:
            raise ValueError("Root state tensor is empty. This indicates actors were not properly created.")
            
        print(f"Acquired root_state_tensor with shape {self.root_state_tensor.shape}")
        
        # Get rigid body state tensor
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor)
        
        # Verify rigid body tensor
        if self.rigid_body_states.numel() == 0:
            raise ValueError("Rigid body state tensor is empty. This indicates rigid bodies were not properly created.")
            
        print(f"Acquired rigid_body_states with shape {self.rigid_body_states.shape}")
        
        # Get DOF properties
        try:
            self.num_dof = self.gym.get_actor_dof_count(self.envs[0], self.hand_handles[0])
            
            # Verify num_dof is valid
            if self.num_dof == 0:
                raise ValueError("Actor has no DOFs. The hand model may be incorrect.")
                
            print(f"Acquired {self.num_dof} DOFs from actor")
        except Exception as e:
            raise ValueError(f"Failed to get DOF count: {e}")
    
    def _setup_tensors(self):
        """Set up tensor buffers for the environment.
        
        This method sets up tensor views and extracts specific state information.
        It should be called after _acquire_tensor_handles.
        """
        # Verify we have a valid dof_state tensor (should be verified in _acquire_tensor_handles)
        if not hasattr(self, 'dof_state') or self.dof_state is None or self.dof_state.numel() == 0:
            raise ValueError("DOF state tensor is invalid or empty. Cannot proceed with setup.")
        
        # Create DOF state tensor views
        try:
            # Get the actual number of DOFs from the hand asset
            expected_dofs = self.num_dof  # This was set during _acquire_tensor_handles
            
            # Reshape DOF state tensor to [num_envs, num_dofs, 2] for position/velocity
            dofs_per_env = self.dof_state.shape[0] // self.num_envs
            print(f"DOF state shape: {self.dof_state.shape}, dofs_per_env: {dofs_per_env}, expected: {expected_dofs}")
            
            # Fail fast if dofs_per_env doesn't match expected_dofs (duplicate loading)
            if dofs_per_env != expected_dofs:
                raise ValueError(f"DOF count mismatch: dof_state has {dofs_per_env} DOFs per environment, but actor reports {expected_dofs} DOFs. This indicates a critical initialization error, likely duplicate assets were loaded.")
                
            # Normal case - just reshape
            _dof_state_tensor = self.dof_state.view(self.num_envs, dofs_per_env, 2)
            self.dof_pos = _dof_state_tensor[..., 0]
            self.dof_vel = _dof_state_tensor[..., 1]
            
            # Verify DOF position and velocity tensors have valid shapes
            if self.dof_pos.shape[1] == 0:
                raise ValueError(f"DOF position tensor has invalid shape {self.dof_pos.shape}")
                
            print(f"Created DOF tensors: pos shape {self.dof_pos.shape}, vel shape {self.dof_vel.shape}")
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to set up DOF state tensors: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to set up DOF state tensors: {e}")
        
        # Get DOF properties - properly handle the structured numpy array format
        try:
            # Get DOF properties from IsaacGym
            _dof_props = self.gym.get_actor_dof_properties(self.envs[0], self.hand_handles[0])
            print(f"DOF properties type: {type(_dof_props)}")
            
            # For numpy structured array, extract the lower and upper limits directly
            if isinstance(_dof_props, np.ndarray) and hasattr(_dof_props, 'dtype') and _dof_props.dtype.names is not None:
                print(f"DOF properties fields: {_dof_props.dtype.names}")
                
                # Extract limits directly from the structured array
                if 'lower' in _dof_props.dtype.names and 'upper' in _dof_props.dtype.names:
                    # Convert directly to tensors
                    self.dof_lower_limits = torch.tensor(_dof_props['lower'], device=self.device)
                    self.dof_upper_limits = torch.tensor(_dof_props['upper'], device=self.device)
                    print(f"Successfully extracted DOF limits from structured array")
                else:
                    raise ValueError(f"DOF properties missing lower/upper fields: {_dof_props.dtype.names}")
            else:
                raise ValueError(f"Unexpected DOF properties format: {type(_dof_props)}")
            
            # Verify DOF limit tensors
            if self.dof_lower_limits.shape[0] == 0:
                raise ValueError(f"DOF limit tensors have invalid shape {self.dof_lower_limits.shape}")
                
            print(f"Created DOF limits: lower shape {self.dof_lower_limits.shape}, upper shape {self.dof_upper_limits.shape}")
            
            # Make sure DOF count matches the expected value - fail fast on mismatch
            if self.dof_pos.shape[1] != len(self.dof_lower_limits):
                raise ValueError(f"DOF count mismatch: dof_pos has {self.dof_pos.shape[1]} DOFs, but properties indicate {len(self.dof_lower_limits)} DOFs. This suggests duplicate assets were loaded.")
        except Exception as e:
            print(f"CRITICAL ERROR getting DOF properties: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to get DOF properties: {e}")
        
        # Set up DOF indices for base and fingers
        try:
            self.base_dof_indices = torch.arange(0, len(self.base_joint_names), device=self.device)
            self.finger_dof_indices = torch.arange(len(self.base_joint_names), 
                                                len(self.base_joint_names) + len(self.finger_joint_names),
                                                device=self.device)
            print(f"Set up DOF indices: base {self.base_dof_indices.shape}, fingers {self.finger_dof_indices.shape}")
        except Exception as e:
            raise RuntimeError(f"Failed to set up DOF indices: {e}")
        
        # Verify rigid_body_states tensor
        if not hasattr(self, 'rigid_body_states') or self.rigid_body_states is None or self.rigid_body_states.numel() == 0:
            raise ValueError("Rigid body state tensor is invalid or empty. Cannot proceed with setup.")
        
        # Create hand state tensor views
        try:
            # View rigid body states as [num_envs, num_bodies, 13] for position/rotation/velocity
            bodies_per_env = self.rigid_body_states.shape[0] // self.num_envs
            _rb_states = self.rigid_body_states.view(self.num_envs, bodies_per_env, 13)
            
            # Extract hand state (assuming base body is index 0)
            self.hand_pos = _rb_states[:, 0, 0:3].clone()  # Use .clone() to ensure we get a contiguous tensor
            self.hand_rot = _rb_states[:, 0, 3:7].clone()
            self.hand_vel = _rb_states[:, 0, 7:10].clone()
            self.hand_ang_vel = _rb_states[:, 0, 10:13].clone()
            
            # Verify hand state tensors
            if self.hand_pos.shape[1] != 3 or self.hand_rot.shape[1] != 4:
                raise ValueError(f"Hand state tensors have invalid shapes: pos {self.hand_pos.shape}, rot {self.hand_rot.shape}")
                
            print(f"Created hand state tensors: pos shape {self.hand_pos.shape}, rot shape {self.hand_rot.shape}")
        except Exception as e:
            raise RuntimeError(f"Failed to set up hand state tensors: {e}")
        
        # Extract fingertip states
        try:
            self.fingertip_state = torch.zeros((self.num_envs, len(self.fingertip_body_names), 13), 
                                           device=self.device)
            self.fingertip_body_handles = []
            
            for i, fingertip_body_name in enumerate(self.fingertip_body_names):
                fingertip_body_idx = self.gym.find_actor_rigid_body_handle(
                    self.envs[0], self.hand_handles[0], fingertip_body_name
                )
                
                if fingertip_body_idx == -1:
                    raise ValueError(f"Could not find fingertip body '{fingertip_body_name}' in the actor")
                    
                self.fingertip_body_handles.append(fingertip_body_idx)
                
                if fingertip_body_idx >= _rb_states.shape[1]:
                    raise ValueError(f"Fingertip body index {fingertip_body_idx} is out of bounds for rigid body states with shape {_rb_states.shape}")
                    
                self.fingertip_state[:, i] = _rb_states[:, fingertip_body_idx].clone()
            
            print(f"Created fingertip state tensor with shape {self.fingertip_state.shape}")
        except Exception as e:
            raise RuntimeError(f"Failed to set up fingertip state tensors: {e}")
        
        # Extract fingerpad states
        try:
            self.fingerpad_state = torch.zeros((self.num_envs, len(self.fingerpad_body_names), 13), 
                                           device=self.device)
                                           
            for i, fingerpad_body_name in enumerate(self.fingerpad_body_names):
                fingerpad_body_idx = self.gym.find_actor_rigid_body_handle(
                    self.envs[0], self.hand_handles[0], fingerpad_body_name
                )
                
                if fingerpad_body_idx == -1:
                    raise ValueError(f"Could not find fingerpad body '{fingerpad_body_name}' in the actor")
                
                if fingerpad_body_idx >= _rb_states.shape[1]:
                    raise ValueError(f"Fingerpad body index {fingerpad_body_idx} is out of bounds for rigid body states with shape {_rb_states.shape}")
                    
                self.fingerpad_state[:, i] = _rb_states[:, fingerpad_body_idx].clone()
            
            print(f"Created fingerpad state tensor with shape {self.fingerpad_state.shape}")
        except Exception as e:
            raise RuntimeError(f"Failed to set up fingerpad state tensors: {e}")
    
    def _compute_observations_buffer(self):
        """Compute observations and update the observation buffer."""
        # Check if tensors are initialized
        if not hasattr(self, '_tensors_initialized') or not self._tensors_initialized:
            print("WARNING: Attempting to compute observations before tensors are initialized.")
            if not hasattr(self, 'obs_buf'):
                print("Creating dummy observation buffer for initial setup")
                # Create a minimal observation buffer to avoid errors
                self.num_observations = 1
                self.obs_buf = torch.zeros((self.num_envs, self.num_observations), device=self.device)
            return
        
        # Initialize observations buffer if needed
        if not hasattr(self, 'obs_buf'):
            print("Computing initial observations...")
            # Verify critical tensors before computing observations
            if hasattr(self, 'dof_pos') and self.dof_pos.shape[1] == 0:
                raise ValueError("DOF position tensor has invalid shape. Cannot compute observations. This indicates the hand model did not load properly.")
            
            # Compute observations
            self._compute_observations()
            
            # Verify observation buffer was created properly
            if hasattr(self, 'obs_buf') and self.obs_buf is not None and self.obs_buf.numel() > 0:
                self.num_observations = self.obs_buf.shape[1]
                print(f"Setting num_observations to {self.num_observations} based on observation buffer")
            else:
                raise ValueError("Observation buffer creation failed. Cannot initialize environment properly.")
        else:
            # Verify critical tensors before updating observations
            if hasattr(self, 'dof_pos') and self.dof_pos.shape[1] == 0:
                raise ValueError("DOF position tensor has invalid shape. Cannot compute observations. This indicates the hand model did not load properly.")
            
            # Update observations
            self._compute_observations()
            
            # Update num_observations to match the observation buffer
            if self.obs_buf.numel() > 0:
                if self.num_observations != self.obs_buf.shape[1]:
                    print(f"Updating num_observations from {self.num_observations} to {self.obs_buf.shape[1]}")
                    self.num_observations = self.obs_buf.shape[1]
            else:
                raise ValueError("Observation buffer is empty after update. Cannot continue execution.")
    
    def _compute_observations(self):
        """Compute observations for the agent."""
        # Check if we're still in initialization
        if not hasattr(self, '_tensors_initialized') or not self._tensors_initialized:
            print("WARNING: Attempting to compute observations before tensors are initialized.")
            # Create a minimal observation buffer to avoid errors during initialization
            self.obs_buf = torch.zeros((self.num_envs, 1), device=self.device)
            return self.obs_buf
            
        # CRITICAL TENSOR CHECKS - these must be valid for the simulation to work
        critical_attrs = ['dof_pos', 'dof_vel', 'hand_pos', 'hand_rot']
        
        for attr in critical_attrs:
            if not hasattr(self, attr) or getattr(self, attr) is None:
                raise ValueError(f"Critical tensor {attr} is not initialized. This indicates the simulation failed to initialize properly.")
                
            # Special check for dof_pos shape
            if attr == 'dof_pos' and self.dof_pos.shape[1] == 0:
                raise ValueError(f"Critical tensor {attr} has invalid shape {self.dof_pos.shape}. This indicates the hand model did not load properly.")
                
        # SECONDARY TENSOR CHECKS - These can be initialized with defaults
        secondary_attrs = ['hand_vel', 'hand_ang_vel', 'fingertip_state', 'progress_buf',
                           'actions', 'prev_actions', 'contact_forces']
        
        for attr in secondary_attrs:
            if not hasattr(self, attr) or getattr(self, attr) is None:
                print(f"Warning: {attr} not initialized. Initializing with zeros.")
                if attr == 'hand_vel':
                    setattr(self, attr, torch.zeros((self.num_envs, 3), device=self.device))
                elif attr == 'hand_ang_vel':
                    setattr(self, attr, torch.zeros((self.num_envs, 3), device=self.device))
                elif attr == 'fingertip_state':
                    setattr(self, attr, torch.zeros((self.num_envs, 5, 13), device=self.device))
                elif attr == 'progress_buf':
                    setattr(self, attr, torch.zeros(self.num_envs, device=self.device, dtype=torch.long))
                elif attr == 'actions' or attr == 'prev_actions':
                    if not hasattr(self, 'num_actions') or self.num_actions == 0:
                        # This is a critical error - we should have num_actions defined by now
                        raise ValueError(f"num_actions is not initialized when trying to create {attr}. The environment was not properly initialized.")
                    setattr(self, attr, torch.zeros((self.num_envs, self.num_actions), device=self.device))
                elif attr == 'contact_forces':
                    setattr(self, attr, torch.zeros((self.num_envs, 5, 3), device=self.device))
        
        # Update obs_dict with current observations
        self.obs_dict = {
            # Hand pose
            "hand_pos": self.hand_pos.clone(),
            "hand_rot": self.hand_rot.clone(),
            "hand_vel": self.hand_vel.clone(),
            "hand_ang_vel": self.hand_ang_vel.clone(),
            
            # Joint state
            "dof_pos": self.dof_pos.clone(),
            "dof_vel": self.dof_vel.clone(),
            
            # Action history
            "actions": self.actions.clone(),
            "prev_actions": self.prev_actions.clone(),
            
            # Progress
            "progress": self.progress_buf.float() / self.max_episode_length,
            
            # Fingertip states
            "fingertip_pos": self.fingertip_state[:, :, 0:3],
            "fingertip_rot": self.fingertip_state[:, :, 3:7],
            
            # Contact forces (updated during post_physics_step)
            "contact_forces": self.contact_forces.clone()
        }
        
        # Add task-specific observations
        task_obs = self.task.get_task_observations()
        if task_obs:
            self.obs_dict.update(task_obs)
        
        # Flatten observations for RL
        obs_list = []
        self.obs_keys = []
        
        # Add core observations in a specific order
        for key in ["hand_pos", "hand_rot", "hand_vel", "hand_ang_vel", 
                   "dof_pos", "dof_vel", "actions", "prev_actions", "progress"]:
            if key in self.obs_dict:
                try:
                    # Use reshape instead of view to handle non-contiguous tensors
                    obs_list.append(self.obs_dict[key].reshape(self.num_envs, -1))
                    self.obs_keys.append(key)
                except Exception as e:
                    print(f"Error reshaping observation {key}: {e}")
        
        # Add task-specific observations
        for key in self.obs_dict:
            if key not in self.obs_keys:
                try:
                    # Use reshape instead of view to handle non-contiguous tensors
                    obs_list.append(self.obs_dict[key].reshape(self.num_envs, -1))
                    self.obs_keys.append(key)
                except Exception as e:
                    print(f"Error reshaping observation {key}: {e}")
        
        # Make sure we have at least one observation
        if len(obs_list) == 0:
            # Create a dummy observation if nothing else is available
            obs_list = [torch.zeros((self.num_envs, 1), device=self.device)]
            self.obs_keys = ["dummy"]
        
        # Concatenate all observations
        try:
            if len(obs_list) > 0:
                # Check that all observation tensors have valid shape (num_envs, *)
                valid_obs = []
                for i, obs in enumerate(obs_list):
                    if obs.shape[0] == self.num_envs and obs.numel() > 0:
                        valid_obs.append(obs)
                    else:
                        print(f"Warning: observation {self.obs_keys[i]} has invalid shape {obs.shape}, skipping")
                
                if len(valid_obs) > 0:
                    self.obs_buf = torch.cat(valid_obs, dim=1)
                    # Print success message
                    print(f"Successfully created observation buffer with shape {self.obs_buf.shape}")
                else:
                    # Create a fallback observation buffer
                    self.obs_buf = torch.zeros((self.num_envs, 1), device=self.device)
            else:
                # Create a fallback observation buffer
                self.obs_buf = torch.zeros((self.num_envs, 1), device=self.device)
        except Exception as e:
            print(f"Error concatenating observations: {e}")
            import traceback
            traceback.print_exc()
            # Create a fallback observation buffer
            self.obs_buf = torch.zeros((self.num_envs, 1), device=self.device)
        
        return self.obs_buf
    
    def check_builtin_success_criteria(self) -> Dict[str, torch.Tensor]:
        """
        Check built-in success criteria that apply to all tasks.
        
        Returns:
            Dictionary of built-in success criteria (name -> boolean tensor)
        """
        # Default implementation has no built-in success criteria
        return {}
    
    def check_builtin_failure_criteria(self) -> Dict[str, torch.Tensor]:
        """
        Check built-in failure criteria that apply to all tasks.
        
        Returns:
            Dictionary of built-in failure criteria (name -> boolean tensor)
        """
        # Default implementation has basic timeout criteria
        timeout = self.progress_buf >= self.max_episode_length - 1
        return {"timeout": timeout}
    
    def _reset_hand_state(self, env_ids):
        """Reset hand state for the specified environments."""
        # Reset DOF state - IMPORTANT: Set base DOFs to match initial position
        self.dof_pos[env_ids] = torch.zeros_like(self.dof_pos[env_ids])
        
        # Explicitly set the base DOF positions to match initial_hand_pos
        self.dof_pos[env_ids, 0] = self.initial_hand_pos[0]  # ARTx
        self.dof_pos[env_ids, 1] = self.initial_hand_pos[1]  # ARTy
        self.dof_pos[env_ids, 2] = self.initial_hand_pos[2]  # ARTz - most critical for gravity issues
        
        # Zero velocities
        self.dof_vel[env_ids] = torch.zeros_like(self.dof_vel[env_ids])
        
        # Reset hand pose to initial values
        initial_hand_pos = torch.tensor(self.initial_hand_pos, device=self.device)
        initial_hand_rot = torch.tensor(self.initial_hand_rot, device=self.device)
        
        # Reset targets for position control
        # Rather than zeroing targets, set them to match the desired initial position
        self.prev_active_targets[env_ids] = torch.zeros_like(self.prev_active_targets[env_ids])
        
        # Initialize base position targets with initial_hand_pos
        self.prev_active_targets[env_ids, 0] = self.initial_hand_pos[0]  # ARTx
        self.prev_active_targets[env_ids, 1] = self.initial_hand_pos[1]  # ARTy
        self.prev_active_targets[env_ids, 2] = self.initial_hand_pos[2]  # ARTz - most critical for gravity issues
        # Initialize rotation targets - default is identity quaternion which is [0,0,0] in axis-angle
        self.prev_active_targets[env_ids, 3:6] = 0.0  # ARRx, ARRy, ARRz
        
        # Debug the reset
        print(f"Reset hand state for envs {env_ids}:")
        print(f"  Initial DOF positions: {self.dof_pos[env_ids[0], :6]}")
        print(f"  Target positions: {self.prev_active_targets[env_ids[0], :6]}")
        
        # Set root body poses
        root_pos = torch.zeros((len(env_ids), 3), device=self.device)
        root_pos[:] = initial_hand_pos
        
        root_rot = torch.zeros((len(env_ids), 4), device=self.device)
        root_rot[:] = initial_hand_rot
        
        root_velocities = torch.zeros((len(env_ids), 6), device=self.device)
        
        # Set base actor state
        indices = torch.arange(len(env_ids), device=self.device).long()
        self.root_state_tensor[env_ids, 0, 0:3] = root_pos
        self.root_state_tensor[env_ids, 0, 3:7] = root_rot
        self.root_state_tensor[env_ids, 0, 7:13] = root_velocities
    
    def _apply_tensor_states(self, env_ids):
        """Apply tensor states to the physics simulation."""
        # Apply DOF state
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids.to(torch.int32)),
            len(env_ids)
        )
        
        # Apply root state
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(env_ids.to(torch.int32)),
            len(env_ids)
        )
    
    def _compute_point_in_hand_frame(self, pos_world, hand_pos, hand_rot):
        """Convert a point from world frame to hand frame."""
        return point_in_hand_frame(pos_world, hand_pos, hand_rot)
    
    def step(self, actions):
        """
        Apply actions, simulate physics, and return observations, rewards, resets, and info.
        
        Args:
            actions (torch.Tensor): Actions to apply to the environment
            
        Returns:
            Tuple of:
                observations (torch.Tensor): Observations after stepping
                rewards (torch.Tensor): Rewards after stepping
                dones (torch.Tensor): Done flags after stepping
                info (dict): Additional info
        """
        # Fail fast if num_actions is invalid
        if not hasattr(self, 'num_actions') or self.num_actions == 0:
            raise ValueError("Invalid num_actions = 0. Environment was not properly initialized.")
        
        # Validate action tensor
        if actions is None:
            raise ValueError("Action tensor cannot be None")
        
        # Validate action dimensions
        if actions.shape[1] != self.num_actions:
            raise ValueError(f"Action shape mismatch. Got {actions.shape}, expected (num_envs, {self.num_actions})")
            
        # Store actions safely
        try:
            self.actions = actions.clone()
        except Exception as e:
            print(f"Error storing actions: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to process actions: {e}")
        
        # Pre-physics step
        try:
            self.pre_physics_step(actions)
        except Exception as e:
            print(f"Error in pre_physics_step: {e}")
            import traceback
            traceback.print_exc()
        
        # Step physics
        try:
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
        except Exception as e:
            print(f"Error in physics simulation: {e}")
            import traceback
            traceback.print_exc()
        
        # Refresh state tensors
        try:
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
        except Exception as e:
            print(f"Error refreshing state tensors: {e}")
            import traceback
            traceback.print_exc()
        
        # Step graphics
        if not self.headless:
            try:
                self.gym.step_graphics(self.sim)
                
                if self.viewer is not None:
                    self.gym.draw_viewer(self.viewer, self.sim, True)
                    
                    # Camera controller events
                    if hasattr(self, 'camera_controller') and self.camera_controller is not None:
                        try:
                            self.camera_controller.check_keyboard_events(lambda env_ids: self.reset_idx(env_ids))
                            self.camera_controller.update_camera_position(self.hand_pos)
                        except Exception as e:
                            print(f"Error in camera controller: {e}")
            except Exception as e:
                print(f"Error in graphics update: {e}")
                import traceback
                traceback.print_exc()
        
        # Update progress buffer
        try:
            if not hasattr(self, 'progress_buf') or self.progress_buf is None:
                self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            self.progress_buf += 1
        except Exception as e:
            print(f"Error updating progress buffer: {e}")
            import traceback
            traceback.print_exc()
        
        # Compute observations
        try:
            self._compute_observations()
        except Exception as e:
            print(f"Error computing observations: {e}")
            import traceback
            traceback.print_exc()
        
        # Ensure observations exist
        if not hasattr(self, 'obs_buf') or self.obs_buf is None or self.obs_buf.shape[1] == 0:
            print("WARNING: obs_buf is not properly initialized, creating default")
            self.obs_buf = torch.zeros((self.num_envs, max(1, self.num_observations)), device=self.device)
            
        # Create default values for return
        rewards = torch.zeros(self.num_envs, device=self.device)
        
        # Create reset buffer if it doesn't exist
        if not hasattr(self, 'reset_buf') or self.reset_buf is None:
            self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            
        # Use reset_buf as dones
        dones = self.reset_buf.clone()
        
        # Initialize extras dict if needed
        if not hasattr(self, 'extras') or self.extras is None:
            self.extras = {}
        
        # Return extras as info
        info = self.extras.copy()  # Make a copy to avoid modifying the original
        
        # Update previous state variables
        if hasattr(self, 'actions'):
            self.prev_actions = self.actions.clone()
            
        if hasattr(self, 'dof_vel'):
            self.prev_dof_vel = self.dof_vel.clone()
            
        if hasattr(self, 'hand_vel'):
            self.prev_hand_vel = self.hand_vel.clone()
            
        if hasattr(self, 'hand_ang_vel'):
            self.prev_hand_ang_vel = self.hand_ang_vel.clone()
        
        # Update contact tracking
        try:
            if "contact_forces" in self.obs_dict:
                contact_force_norm = torch.norm(self.obs_dict["contact_forces"], dim=2)
                self.prev_contacts = contact_force_norm > 0.1
        except Exception as e:
            print(f"Error updating contact tracking: {e}")
            
        return self.obs_buf, rewards, dones, info
    
    def post_physics_step(self):
        """Process state after physics simulation step."""
        # Auto-detect physics_steps_per_control_step based on reset requirements
        # This allows the environment to automatically adjust the number of physics steps
        # per control step based on how many steps are needed for stable reset
        if not hasattr(self, 'auto_detected_physics_steps'):
            # Wait a few steps to ensure we've seen at least one reset cycle
            if self.physics_step_count > 10:
                measured_steps = self.physics_step_count - self.last_control_step_count
                
                # Only update if measured_steps is > 1 to avoid false detections
                if measured_steps > 1:
                    self.physics_steps_per_control_step = measured_steps
                    self.auto_detected_physics_steps = True
                    
                    import logging
                    logging.info(f"Auto-detected physics_steps_per_control_step: {measured_steps}. "
                                f"This means {measured_steps} physics steps occur between each policy action.")
                    
                    # Update control_dt to match the actual control frequency
                    self.control_dt = self.physics_dt * self.physics_steps_per_control_step
            
        # Compute observations
        self._compute_observations()
        
        # Create default reset_buf if not initialized
        if not hasattr(self, 'reset_buf') or self.reset_buf is None:
            self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Create default extras dict if not initialized
        if not hasattr(self, 'extras') or self.extras is None:
            self.extras = {}
        
        # Create default rewards if not initialized
        if not hasattr(self, 'rewards') or self.rewards is None:
            self.rewards = torch.zeros(self.num_envs, device=self.device)
        
        # Create default reward components if not initialized
        if not hasattr(self, 'reward_components') or self.reward_components is None:
            self.reward_components = {}
        
        # Verify that components are properly initialized
        if self.success_failure_tracker is None:
            raise ValueError("Success/failure tracker is not initialized. Component initialization failed.")
        
        if self.reward_calculator is None:
            raise ValueError("Reward calculator is not initialized. Component initialization failed.")
        
        try:
            # Get task-specific criteria
            task_success = self.task.check_task_success_criteria()
            task_failure = self.task.check_task_failure_criteria()
            
            # Get built-in criteria
            builtin_success = self.check_builtin_success_criteria()
            builtin_failure = self.check_builtin_failure_criteria()
            
            # Evaluate success and failure
            self.reset_buf, self.extras = self.success_failure_tracker.evaluate(
                self.progress_buf,
                builtin_success,
                task_success,
                builtin_failure,
                task_failure
            )
            
            # Get success/failure rewards
            success_failure_rewards = self.success_failure_tracker.get_rewards()
            
            # Compute common reward terms
            common_rewards = self.reward_calculator.compute_common_reward_terms(
                self.obs_dict,
                self.hand_vel,
                self.hand_ang_vel,
                self.dof_vel,
                self.dof_pos,
                self.dof_lower_limits,
                self.dof_upper_limits,
                self.prev_dof_vel,
                self.prev_hand_vel,
                self.prev_hand_ang_vel,
                self.prev_contacts
            )
            
            # Compute task-specific reward terms
            task_rewards = self.task.compute_task_reward_terms(self.obs_dict)
            
            # Compute total reward
            self.rewards, self.reward_components = self.reward_calculator.compute_total_reward(
                common_rewards,
                task_rewards,
                success_failure_rewards
            )
        except Exception as e:
            print(f"Error in reward/success evaluation: {e}")
            import traceback
            traceback.print_exc()
            
            # Set default values if there's an error
            self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.extras = {}
            self.rewards = torch.zeros(self.num_envs, device=self.device)
        
        # Update fingertip colors
        try:
            if hasattr(self, 'fingertip_visualizer') and self.fingertip_visualizer is not None:
                self.fingertip_visualizer.update_colors(self.obs_dict["contact_forces"])
        except Exception as e:
            print(f"Error updating fingertip colors: {e}")
        
        # Process camera control
        try:
            if hasattr(self, 'camera_controller') and self.camera_controller is not None:
                self.camera_controller.check_keyboard_events(lambda env_ids: self.reset_idx(env_ids))
                self.camera_controller.update_camera_position(self.hand_pos)
        except Exception as e:
            print(f"Error in camera control: {e}")
        
        # Update state tracking for next step
        self.prev_actions = self.actions.clone()
        self.prev_dof_vel = self.dof_vel.clone()
        self.prev_hand_vel = self.hand_vel.clone()
        self.prev_hand_ang_vel = self.hand_ang_vel.clone()
        
        # Update contact tracking
        try:
            contact_force_norm = torch.norm(self.obs_dict["contact_forces"], dim=2)
            self.prev_contacts = contact_force_norm > 0.1
        except Exception as e:
            print(f"Error updating contact tracking: {e}")
            # Initialize with default values
            self.prev_contacts = torch.zeros((self.num_envs, 5), device=self.device, dtype=torch.bool)
        
        # Increment progress buffer
        self.progress_buf += 1
        
        # Reset environments if needed
        if torch.any(self.reset_buf):
            self.reset_idx(torch.nonzero(self.reset_buf, as_tuple=False).flatten())
    
    def reset_idx(self, env_ids):
        """Reset environments at specified indices."""
        if len(env_ids) == 0:
            return
        
        # Verify reset_buf exists and has correct shape
        if not hasattr(self, 'reset_buf') or self.reset_buf is None:
            self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            
        # Reset buffers
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        
        # Check and initialize prev_active_targets if needed
        if not hasattr(self, 'prev_active_targets') or self.prev_active_targets is None:
            self.prev_active_targets = torch.zeros(
                (self.num_envs, self.NUM_BASE_DOFS + self.NUM_ACTIVE_FINGER_DOFS), 
                device=self.device, dtype=torch.float
            )
        elif self.prev_active_targets.shape[1] == 0:
            self.prev_active_targets = torch.zeros(
                (self.num_envs, self.NUM_BASE_DOFS + self.NUM_ACTIVE_FINGER_DOFS), 
                device=self.device, dtype=torch.float
            )
            
        self.prev_active_targets[env_ids] = 0
        
        try:
            # Reset hand state
            self._reset_hand_state(env_ids)
        except Exception as e:
            print(f"Error in _reset_hand_state: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            # Reset task-specific state
            self.task.reset_task_state(env_ids)
        except Exception as e:
            print(f"Error in task.reset_task_state: {e}")
            import traceback
            traceback.print_exc()
        
        # Reset tracking variables
        self.actions[env_ids] = 0
        self.prev_actions[env_ids] = 0
        self.prev_hand_vel[env_ids] = 0
        self.prev_hand_ang_vel[env_ids] = 0
        self.prev_dof_vel[env_ids] = 0
        self.prev_contacts[env_ids] = False
        
        try:
            # Reset success/failure tracking
            if hasattr(self, 'success_failure_tracker') and self.success_failure_tracker is not None:
                self.success_failure_tracker.reset(env_ids)
        except Exception as e:
            print(f"Error in success_failure_tracker.reset: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            # Apply tensor states
            self._apply_tensor_states(env_ids)
        except Exception as e:
            print(f"Error in _apply_tensor_states: {e}")
            import traceback
            traceback.print_exc()
    
    def pre_physics_step(self, actions):
        """Process actions before physics simulation step."""
        self.actions = actions.clone()
        
        # Track physics step count
        self.physics_step_count += 1
        
        # Determine whether this is a control step or just a physics step
        is_control_step = self.physics_step_count % self.physics_steps_per_control_step == 0
        
        if is_control_step:
            # Store the step count for verification
            self.last_control_step_count = self.physics_step_count
            
            # Process actions for the hand
            self._process_actions(actions)
    
    def _process_actions(self, actions):
        """
        Process actions from policy to joint targets.
        
        Args:
            actions: Actions from policy
        """
        try:
            # Ensure actions are properly initialized
            if not hasattr(self, 'actions') or self.actions is None:
                self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
                
            # Store actions for observation
            if actions is not None:
                try:
                    self.actions = actions.clone()
                except Exception as e:
                    print(f"Error cloning actions: {e}")
                    self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
            
            # Verify we have the required tensor attributes
            required_tensors = ['dof_pos', 'dof_vel', 'prev_active_targets']
            for tensor_name in required_tensors:
                if not hasattr(self, tensor_name) or getattr(self, tensor_name) is None:
                    print(f"Warning: {tensor_name} not initialized, initializing now")
                    if tensor_name == 'prev_active_targets':
                        self.prev_active_targets = torch.zeros(
                            (self.num_envs, self.NUM_BASE_DOFS + self.NUM_ACTIVE_FINGER_DOFS), 
                            device=self.device, dtype=torch.float
                        )

            # Check if prev_active_targets has the correct shape
            if hasattr(self, 'prev_active_targets') and self.prev_active_targets.shape[1] == 0:
                print("Warning: prev_active_targets has zero dimension, reinitializing")
                self.prev_active_targets = torch.zeros(
                    (self.num_envs, self.NUM_BASE_DOFS + self.NUM_ACTIVE_FINGER_DOFS), 
                    device=self.device, dtype=torch.float
                )

            # Ensure we have all the necessary tensor views
            if not hasattr(self, 'dof_pos') or self.dof_pos is None:
                print("DOF position tensor not initialized, skipping action processing")
                return
                
            # Fail fast on invalid tensor shapes
            if self.dof_pos.shape[1] == 0:
                raise ValueError(f"DOF position tensor has invalid shape {self.dof_pos.shape}. The hand model likely failed to load properly.")
                
            # Initialize targets with current positions
            targets = self.dof_pos.clone()
            
            # Print debug info
            print(f"DEBUG: actions shape: {self.actions.shape}")
            print(f"DEBUG: prev_active_targets shape: {self.prev_active_targets.shape}")
            print(f"DEBUG: dof_pos shape: {self.dof_pos.shape}")
            print(f"DEBUG: targets shape: {targets.shape}")
            
            # Split actions into base and finger components
            action_idx = 0
            
            # Apply base actions
            if self.control_hand_base:
                # Extract base actions
                if self.actions.shape[1] > 0 and self.NUM_BASE_DOFS > 0:
                    base_actions = self.actions[:, :min(self.NUM_BASE_DOFS, self.actions.shape[1])]
                    
                    # Base targets from actions (position control)
                    base_pos_targets = None
                    
                    if self.action_control_mode == "position":
                        # Direct position targets
                        base_pos_targets = base_actions
                    elif self.action_control_mode == "position_delta":
                        # Position targets as delta from current or previous targets
                        # Ensure dimensions match
                        if base_actions.shape[1] == self.prev_active_targets[:, :self.NUM_BASE_DOFS].shape[1]:
                            # When action is 0, maintain previous target (no movement)
                            # This is crucial for ARTz (vertical position) to avoid downward drift
                            base_pos_targets = self.prev_active_targets[:, :self.NUM_BASE_DOFS] + base_actions
                            
                            # Explicitly print the targets for debugging
                            print(f"Base targets before: {self.prev_active_targets[:, :self.NUM_BASE_DOFS][0]}")
                            print(f"Base actions: {base_actions[0]}")
                            print(f"Base targets after: {base_pos_targets[0]}")
                        else:
                            print(f"Warning: shape mismatch for base_actions {base_actions.shape} and prev_targets {self.prev_active_targets[:, :self.NUM_BASE_DOFS].shape}")
                            # Use direct position as fallback
                            base_pos_targets = base_actions
                    else:
                        print(f"Unknown control mode: {self.action_control_mode}")
                    
                    # Apply targets to the base DOFs (with safety limits)
                    if base_pos_targets is not None:
                        # Verify tensor shapes before assignment
                        if targets.shape[1] >= self.NUM_BASE_DOFS and base_pos_targets.shape[1] == self.NUM_BASE_DOFS:
                            # DEBUGGING: Print the values right before assignment
                            print(f"Before assignment - targets[:, :self.NUM_BASE_DOFS] = {targets[0, :self.NUM_BASE_DOFS]}")
                            print(f"Assigning base_pos_targets = {base_pos_targets[0]}")
                            
                            # Assign to targets tensor - THIS IS THE CRITICAL PART
                            targets[:, :self.NUM_BASE_DOFS] = base_pos_targets
                            
                            # DEBUGGING: Print the values right after assignment
                            print(f"After assignment - targets[:, :self.NUM_BASE_DOFS] = {targets[0, :self.NUM_BASE_DOFS]}")
                            
                            # Update previous targets
                            self.prev_active_targets[:, :self.NUM_BASE_DOFS] = base_pos_targets
                        else:
                            print(f"Error: Cannot assign base_pos_targets {base_pos_targets.shape} to targets[:, :self.NUM_BASE_DOFS] with shape {targets[:, :self.NUM_BASE_DOFS].shape}")
                    
                    # Update action index
                    action_idx += self.NUM_BASE_DOFS
                else:
                    # Fallback to default targets if action dimensions are wrong
                    print(f"Warning: action dimensions are invalid, using default targets")
                    # Verify tensor shapes before assignment
                    if targets.shape[1] >= self.NUM_BASE_DOFS and self.default_base_targets.numel() >= self.NUM_BASE_DOFS:
                        targets[:, :self.NUM_BASE_DOFS] = self.default_base_targets
                    else:
                        print(f"Error: Cannot assign default_base_targets {self.default_base_targets.shape} to targets[:, :self.NUM_BASE_DOFS] with shape {targets[:, :self.NUM_BASE_DOFS].shape if targets.shape[1] > 0 else 'empty'}")
            else:
                # Use default or task-provided base targets
                if hasattr(self.task, 'get_task_dof_targets'):
                    task_targets = self.task.get_task_dof_targets()
                    if task_targets is not None and 'base_targets' in task_targets:
                        # Verify tensor shapes before assignment
                        if targets.shape[1] >= self.NUM_BASE_DOFS and task_targets['base_targets'].shape[1] == self.NUM_BASE_DOFS:
                            targets[:, :self.NUM_BASE_DOFS] = task_targets['base_targets']
                        else:
                            print(f"Error: Cannot assign task base_targets {task_targets['base_targets'].shape} to targets[:, :self.NUM_BASE_DOFS]")
                            if targets.shape[1] >= self.NUM_BASE_DOFS and self.default_base_targets.numel() >= self.NUM_BASE_DOFS:
                                targets[:, :self.NUM_BASE_DOFS] = self.default_base_targets
                    else:
                        if targets.shape[1] >= self.NUM_BASE_DOFS and self.default_base_targets.numel() >= self.NUM_BASE_DOFS:
                            targets[:, :self.NUM_BASE_DOFS] = self.default_base_targets
                        else:
                            print(f"Error: Cannot assign default_base_targets to targets[:, :self.NUM_BASE_DOFS]")
                else:
                    if targets.shape[1] >= self.NUM_BASE_DOFS and self.default_base_targets.numel() >= self.NUM_BASE_DOFS:
                        targets[:, :self.NUM_BASE_DOFS] = self.default_base_targets
                    else:
                        print(f"Error: Cannot assign default_base_targets to targets[:, :self.NUM_BASE_DOFS]")
            
            # Apply finger actions
            if self.control_fingers:
                # Print debug info for finger actions
                print(f"DEBUG: action_idx: {action_idx}")
                print(f"DEBUG: num finger DOFs: {self.NUM_ACTIVE_FINGER_DOFS}")
                
                # Extract finger actions
                if self.actions.shape[1] > action_idx and self.NUM_ACTIVE_FINGER_DOFS > 0:
                    finger_end_idx = min(action_idx + self.NUM_ACTIVE_FINGER_DOFS, self.actions.shape[1])
                    finger_actions = self.actions[:, action_idx:finger_end_idx]
                    
                    # Finger targets from actions
                    finger_pos_targets = None
                    
                    if self.action_control_mode == "position":
                        # Direct position targets
                        finger_pos_targets = finger_actions
                    elif self.action_control_mode == "position_delta":
                        # Position targets as delta from current or previous targets
                        # Ensure dimensions match
                        if finger_actions.shape[1] == self.prev_active_targets[:, self.NUM_BASE_DOFS:].shape[1]:
                            finger_pos_targets = self.prev_active_targets[:, self.NUM_BASE_DOFS:] + finger_actions
                        else:
                            print(f"Warning: shape mismatch for finger_actions {finger_actions.shape} and prev_targets {self.prev_active_targets[:, self.NUM_BASE_DOFS:].shape}")
                            # Use direct position as fallback
                            finger_pos_targets = finger_actions
                    else:
                        print(f"Unknown control mode: {self.action_control_mode}")
                    
                    # Apply targets to the finger DOFs (with safety checks)
                    if finger_pos_targets is not None and finger_pos_targets.shape[1] > 0:
                        # Map active finger DOFs to full finger DOF space
                        # This is a simplified implementation - in a real controller, you'd have
                        # a more complex mapping to handle the underactuated fingers
                        for i, name in enumerate(self.finger_joint_names):
                            if name in self.joint_to_control:
                                control_name = self.joint_to_control[name]
                                try:
                                    control_idx = self.active_joint_names.index(control_name)
                                    if control_idx < finger_pos_targets.shape[1]:
                                        finger_dof_idx = i + self.NUM_BASE_DOFS
                                        targets[:, finger_dof_idx] = finger_pos_targets[:, control_idx]
                                except ValueError:
                                    print(f"Warning: control name {control_name} not found in active_joint_names")
                        
                        # Update previous targets
                        self.prev_active_targets[:, self.NUM_BASE_DOFS:] = finger_pos_targets
                else:
                    # Fallback to default targets if action dimensions are wrong
                    print(f"Warning: finger action dimensions are invalid, using default targets")
                    for i, name in enumerate(self.finger_joint_names):
                        if name in self.joint_to_control:
                            control_name = self.joint_to_control[name]
                            try:
                                control_idx = self.active_joint_names.index(control_name)
                                finger_dof_idx = i + self.NUM_BASE_DOFS
                                targets[:, finger_dof_idx] = self.default_finger_targets[control_idx]
                            except (ValueError, IndexError) as e:
                                print(f"Warning: Error setting default target for {name}: {e}")
            else:
                # Use default or task-provided finger targets
                if hasattr(self.task, 'get_task_dof_targets'):
                    task_targets = self.task.get_task_dof_targets()
                    if task_targets is not None and 'finger_targets' in task_targets:
                        # Map to full finger DOF space
                        for i, name in enumerate(self.finger_joint_names):
                            if name in self.joint_to_control:
                                control_name = self.joint_to_control[name]
                                try:
                                    control_idx = self.active_joint_names.index(control_name)
                                    
                                    finger_dof_idx = i + self.NUM_BASE_DOFS
                                    targets[:, finger_dof_idx] = task_targets['finger_targets'][:, control_idx]
                                except (ValueError, IndexError) as e:
                                    print(f"Warning: Error mapping task target for {name}: {e}")
                    else:
                        # Map default targets to full finger DOF space
                        for i, name in enumerate(self.finger_joint_names):
                            if name in self.joint_to_control:
                                control_name = self.joint_to_control[name]
                                try:
                                    control_idx = self.active_joint_names.index(control_name)
                                    
                                    finger_dof_idx = i + self.NUM_BASE_DOFS
                                    targets[:, finger_dof_idx] = self.default_finger_targets[control_idx]
                                except (ValueError, IndexError) as e:
                                    print(f"Warning: Error setting default target for {name}: {e}")
                else:
                    # Use default targets
                    for i, name in enumerate(self.finger_joint_names):
                        if name in self.joint_to_control:
                            control_name = self.joint_to_control[name]
                            try:
                                control_idx = self.active_joint_names.index(control_name)
                                
                                finger_dof_idx = i + self.NUM_BASE_DOFS
                                if control_idx < len(self.default_finger_targets):
                                    targets[:, finger_dof_idx] = self.default_finger_targets[control_idx]
                            except (ValueError, IndexError) as e:
                                print(f"Warning: Error setting default target for {name}: {e}")
            
            # Apply target positions with PD control
            try:
                # Check if DOF limits are properly initialized
                if not hasattr(self, 'dof_lower_limits') or self.dof_lower_limits is None:
                    print("Creating default DOF limits")
                    self.dof_lower_limits = torch.full((self.num_dof,), -1.0, device=self.device)
                    self.dof_upper_limits = torch.full((self.num_dof,), 1.0, device=self.device)
                
                # Make sure DOF limits have the right shape
                if self.dof_lower_limits.shape[0] != self.num_dof:
                    print(f"Resizing DOF limits from {self.dof_lower_limits.shape[0]} to {self.num_dof}")
                    # Resize limits if necessary
                    old_lower = self.dof_lower_limits
                    old_upper = self.dof_upper_limits
                    self.dof_lower_limits = torch.full((self.num_dof,), -1.0, device=self.device)
                    self.dof_upper_limits = torch.full((self.num_dof,), 1.0, device=self.device)
                    
                    # Copy as much of the old data as will fit
                    min_size = min(old_lower.shape[0], self.num_dof)
                    self.dof_lower_limits[:min_size] = old_lower[:min_size]
                    self.dof_upper_limits[:min_size] = old_upper[:min_size]
                
                # Print debug info
                print(f"DEBUG: targets shape: {targets.shape}")
                print(f"DEBUG: lower limits shape: {self.dof_lower_limits.shape}")
                print(f"DEBUG: upper limits shape: {self.dof_upper_limits.shape}")
                
                # CRITICAL FIX: Create direct targets that bypass any transformations
                # Create a completely new tensor from scratch to avoid any issues
                direct_targets = torch.zeros((self.num_envs, self.num_dof), device=self.device)
                
                # CRITICAL: Directly set the base DOF targets from prev_active_targets
                # This skips all the target tensor transformations that might zero things out
                if self.control_hand_base:
                    # Set all base DOFs directly from prev_active_targets
                    for env_idx in range(self.num_envs):
                        for dof_idx in range(self.NUM_BASE_DOFS):
                            if dof_idx < self.prev_active_targets.shape[1]:
                                # Just copy directly from prev_active_targets which is known good
                                direct_targets[env_idx, dof_idx] = self.prev_active_targets[env_idx, dof_idx]
                
                # DEBUG: Print the direct targets we just set
                print(f"DIRECT targets for ARTz: {direct_targets[0, 2]:.4f}")
                print(f"from prev_active_targets: {self.prev_active_targets[0, 2]:.4f}")
                
                # Copy finger targets from original targets tensor for completeness
                # But with direct DOF indexing to avoid any issues
                if self.control_fingers:
                    finger_start_idx = self.NUM_BASE_DOFS
                    for env_idx in range(self.num_envs):
                        for i, name in enumerate(self.finger_joint_names):
                            dof_idx = i + self.NUM_BASE_DOFS
                            if dof_idx < direct_targets.shape[1]:
                                # Start with 0.0 for all finger DOFs
                                direct_targets[env_idx, dof_idx] = 0.0
                                
                                # If we have a control mapping, use it
                                if name in self.joint_to_control:
                                    control_name = self.joint_to_control[name]
                                    try:
                                        control_idx = self.active_joint_names.index(control_name)
                                        finger_pos_idx = self.NUM_BASE_DOFS + control_idx
                                        if finger_pos_idx < self.prev_active_targets.shape[1]:
                                            direct_targets[env_idx, dof_idx] = self.prev_active_targets[env_idx, finger_pos_idx]
                                    except (ValueError, IndexError) as e:
                                        pass  # Safely continue if mapping fails
                
                # Make sure all targets are within limits
                direct_targets = tensor_clamp(
                    direct_targets, 
                    self.dof_lower_limits.unsqueeze(0), 
                    self.dof_upper_limits.unsqueeze(0)
                )
                
                # Store the direct_targets as a class variable to prevent garbage collection
                if not hasattr(self, 'current_targets') or self.current_targets is None or self.current_targets.shape != direct_targets.shape:
                    self.current_targets = direct_targets.clone()
                else:
                    self.current_targets.copy_(direct_targets)
                
                # DOUBLE CHECK: Verify targets right before sending to simulator
                print(f"FINAL ARTz target before sending: {self.current_targets[0, 2]:.4f}")
                
                # Set PD control targets with the robust tensor
                self.gym.set_dof_position_target_tensor(
                    self.sim, 
                    gymtorch.unwrap_tensor(self.current_targets)
                )
                print("Successfully set DOF targets")
                
                # DEBUGGING: Verify the targets were actually sent
                try:
                    for env_idx in range(min(1, self.num_envs)):  # Just check first env
                        print(f"Verifying target values from simulator for env {env_idx}:")
                        for dof_idx in range(min(6, self.num_dof)):  # Just check base DOFs
                            target_pos = self.gym.get_dof_target_position(self.envs[env_idx], dof_idx)
                            print(f"  DOF {dof_idx}: {target_pos:.4f} (expected {self.current_targets[env_idx, dof_idx]:.4f})")
                except Exception as e:
                    print(f"Error verifying target positions: {e}")
                
                # DEBUGGING: Verify targets were actually set by reading them back
                try:
                    # Create a tensor to store the target positions
                    target_positions = torch.zeros_like(targets)
                    
                    # Get the target positions
                    for env_idx in range(self.num_envs):
                        for dof_idx in range(min(6, targets.shape[1])):  # Just check base DOFs
                            # Get the target position for this DOF
                            target_pos = self.gym.get_dof_target_position(self.envs[env_idx], dof_idx)
                            target_positions[env_idx, dof_idx] = target_pos
                    
                    # Print the retrieved targets for verification
                    print(f"Targets retrieved from simulator:")
                    print(f"  Base DOFs: {target_positions[0, :self.NUM_BASE_DOFS]}")
                except Exception as e:
                    print(f"Error verifying target positions: {e}")
            except Exception as e:
                print(f"Error setting DOF targets: {e}")
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            print(f"Error in _process_actions: {e}")
            import traceback
            traceback.print_exc()