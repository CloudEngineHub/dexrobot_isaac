"""
Base class for DexHand tasks.

This module provides the DexHandBase class that uses composition
to delegate functionality to specialized components.
"""

import os
import sys
import time
import math
import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Set

# Import Gym
import gym
from gym import spaces

# Import IsaacGym first
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import to_torch, tensor_clamp, quat_mul, quat_conjugate

# Then import PyTorch
import torch

# Import components
from dex_hand_env.components.camera_controller import CameraController
from dex_hand_env.components.fingertip_visualizer import FingertipVisualizer
from dex_hand_env.components.success_failure_tracker import SuccessFailureTracker
from dex_hand_env.components.reward_calculator import RewardCalculator
from dex_hand_env.components.physics_manager import PhysicsManager
from dex_hand_env.components.hand_initializer import HandInitializer
from dex_hand_env.components.action_processor import ActionProcessor
from dex_hand_env.components.observation_encoder import ObservationEncoder
from dex_hand_env.components.reset_manager import ResetManager
from dex_hand_env.components.tensor_manager import TensorManager

# Import utilities
from dex_hand_env.utils.coordinate_transforms import point_in_hand_frame

# Import task interface
from dex_hand_env.tasks.task_interface import DexTask
# Import base task class
from dex_hand_env.tasks.base.vec_task import VecTask


class DexHandBase(VecTask):
    """
    Base class for DexHand tasks that implements common functionality for all dexterous hand tasks.

    This class provides a unified framework for dexterous manipulation tasks using the DexHand
    robot model. It uses composition to delegate functionality to specialized components:

    - PhysicsManager: Handles physics simulation and stepping
    - HandInitializer: Manages hand model loading and creation
    - ActionProcessor: Processes policy actions for PD control
    - ObservationEncoder: Builds observation space from states
    - ResetManager: Handles environment resets and randomization
    - TensorManager: Manages simulation tensors and handles
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

        # Store GPU pipeline flag for use in step() and other methods
        self.use_gpu_pipeline = cfg["sim"].get("use_gpu_pipeline", False)
        
        # Ensure we're using a GPU device if GPU pipeline is enabled
        if self.use_gpu_pipeline and sim_device == 'cpu':
            print("WARNING: GPU Pipeline is enabled but using CPU device. This may cause errors. Disabling GPU pipeline.")
            cfg["sim"]["use_gpu_pipeline"] = False
            self.use_gpu_pipeline = False

        print(f"GPU Pipeline: {'enabled' if self.use_gpu_pipeline else 'disabled'}")
        print("=============================================\n")
        
        # Core initialization variables
        self.cfg = cfg
        self.task = task
        self.asset_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets")
        
        # Set device early for tensor creation
        self.device = rl_device
        
        # Task specific parameters
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        
        # Physics parameters
        self.physics_dt = self.cfg["sim"].get("dt", 0.01)  # Physics simulation timestep
        self.physics_steps_per_control_step = 1  # Will be auto-detected by PhysicsManager
        
        # Define model constants for the hand
        self.NUM_BASE_DOFS = 6  # 3 translation, 3 rotation
        self.NUM_ACTIVE_FINGER_DOFS = 12  # 12 finger controls mapping to 19 DOFs with coupling
        
        # Define joint names
        self.base_joint_names = [
            "ARTx", "ARTy", "ARTz", "ARRx", "ARRy", "ARRz"
        ]
        
        self.finger_joint_names = [
            "r_f_joint1_1", "r_f_joint1_2", "r_f_joint1_3", "r_f_joint1_4",
            "r_f_joint2_1", "r_f_joint2_2", "r_f_joint2_3", "r_f_joint2_4",
            "r_f_joint3_2", "r_f_joint3_3", "r_f_joint3_4",  # r_f_joint3_1 is fixed
            "r_f_joint4_1", "r_f_joint4_2", "r_f_joint4_3", "r_f_joint4_4",
            "r_f_joint5_1", "r_f_joint5_2", "r_f_joint5_3", "r_f_joint5_4"
        ]
        
        # Define fingertip body names
        self.fingertip_body_names = [
            "r_f_link1_tip", "r_f_link2_tip", "r_f_link3_tip",
            "r_f_link4_tip", "r_f_link5_tip"
        ]
        
        self.fingerpad_body_names = [
            "r_f_link1_pad", "r_f_link2_pad", "r_f_link3_pad",
            "r_f_link4_pad", "r_f_link5_pad"
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
        
        # Default hand asset file
        self.hand_asset_file = "dexrobot_mujoco/dexrobot_mujoco/models/dexhand021_right_simplified_floating.xml"
        
        # Flag to track initialization state
        self._tensors_initialized = False
        
        # Initialize rule-based controller functions
        self.rule_based_base_controller = None
        self.rule_based_finger_controller = None
        
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
        
        # Initialize components
        self._init_components()
        
        # Verify tensors were properly initialized
        if not hasattr(self, 'tensor_manager'):
            raise RuntimeError("TensorManager component was not created. The simulation initialization has failed.")
        
        if not self.tensor_manager.tensors_initialized:
            raise RuntimeError("Tensors were not properly initialized. The simulation initialization has failed.")
        
        # Initialize observation dict
        self.obs_dict = {}
        
        # Additional setup (should only happen after components are created)
        self._setup_additional_tensors()
        
        print("DexHandBase initialization complete.")

    def _init_components(self):
        """
        Initialize all components for the environment.
        """
        print("Creating components...")
        
        # Check if simulation exists (should have been created by parent class)
        if not hasattr(self, 'sim') or self.sim is None:
            print("Simulation not created by parent class, creating now...")
            self.sim = self.create_sim()
        
        # Create hand initializer
        self.hand_initializer = HandInitializer(
            gym=self.gym,
            sim=self.sim,
            num_envs=self.num_envs,
            device=self.device,
            asset_root=self.asset_root
        )
        
        # Joint properties (stiffness/damping) now loaded from MJCF model
        
        # Set initial pose from config
        if "initialHandPos" in self.cfg["env"]:
            self.hand_initializer.set_initial_pose(
                pos=self.cfg["env"].get("initialHandPos", [0.0, 0.0, 0.5]),
                rot=self.cfg["env"].get("initialHandRot", [0.0, 0.0, 0.0, 1.0])
            )
        
        # Load hand asset
        self.hand_asset = self.hand_initializer.load_hand_asset(self.hand_asset_file)
        
        # Create the environments
        self._create_envs()
        
        # Create hands in the environments
        handles = self.hand_initializer.create_hands(self.envs, self.hand_asset)
        self.hand_handles = handles["hand_handles"]
        self.fingertip_body_handles = handles["fingertip_body_handles"]
        self.hand_indices = handles["hand_indices"]
        self.fingertip_indices = handles["fingertip_indices"]
        self.dof_properties_from_asset = handles.get("dof_properties", None)
        
        # Set up viewer
        # CRITICAL: Create viewer even in headless mode for proper DOF control
        print("Creating viewer...")
        self.set_viewer()
        print("Viewer created")
        
        # Create tensor manager after environment setup
        self.tensor_manager = TensorManager(
            gym=self.gym,
            sim=self.sim,
            num_envs=self.num_envs,
            device=self.device
        )
        
        # Acquire tensor handles BEFORE prepare_sim (critical for GPU pipeline)
        self.tensor_manager.acquire_tensor_handles()
        
        # Pass DOF properties from asset to tensor manager if available
        if self.dof_properties_from_asset is not None:
            self.tensor_manager.set_dof_properties(self.dof_properties_from_asset)
        
        # CRITICAL: Call prepare_sim after all actors are created and tensors acquired
        # This is needed for both GPU pipeline and proper DOF control in headless mode
        print("Preparing simulation...")
        self.gym.prepare_sim(self.sim)
        print("Simulation prepared successfully")
            
        # Set up tensors
        tensors = self.tensor_manager.setup_tensors(self.fingertip_indices)
        self.dof_state = tensors["dof_state"]
        self.dof_pos = tensors["dof_pos"]
        self.dof_vel = tensors["dof_vel"]
        self.root_state_tensor = tensors["root_state_tensor"]
        self.num_dof = tensors["num_dof"]
        self.dof_props = tensors["dof_props"]
        self.rigid_body_states = tensors["rigid_body_states"]
        self.contact_forces = tensors["contact_forces"]
        
        # Create physics manager
        self.physics_manager = PhysicsManager(
            gym=self.gym,
            sim=self.sim,
            device=self.device,
            use_gpu_pipeline=self.use_gpu_pipeline
        )
        
        # Set physics timestep
        self.physics_manager.set_dt(self.physics_dt)
        
        # Create action processor
        self.action_processor = ActionProcessor(
            gym=self.gym,
            sim=self.sim,
            num_envs=self.num_envs,
            device=self.device,
            dof_props=self.dof_props,
            hand_asset=self.hand_asset
        )
        
        # Configure action processor
        self.action_processor.setup(self.num_dof, self.dof_props)
        
        # Set control mode
        if "controlMode" in self.cfg["env"]:
            self.action_processor.set_control_mode(self.cfg["env"]["controlMode"])
        elif "controlType" in self.cfg["env"]:
            # Map old controlType to new controlMode
            control_type = self.cfg["env"]["controlType"]
            if control_type == "default":
                self.action_processor.set_control_mode("position")
            elif control_type == "relative":
                self.action_processor.set_control_mode("position_delta")
        
        # Set control options
        self.action_processor.set_control_options(
            control_hand_base=self.cfg["env"].get("controlHandBase", True),
            control_fingers=self.cfg["env"].get("controlFingers", True)
        )
        
        # Set default targets
        if "defaultBaseTargets" in self.cfg["env"]:
            self.action_processor.set_default_targets(
                base_targets=self.cfg["env"]["defaultBaseTargets"]
            )
        if "defaultFingerTargets" in self.cfg["env"]:
            self.action_processor.set_default_targets(
                finger_targets=self.cfg["env"]["defaultFingerTargets"]
            )
        
        # Set velocity limits
        self.action_processor.set_velocity_limits(
            finger_vel_limit=self.cfg["env"].get("maxFingerVelocity", 2.0),
            base_lin_vel_limit=self.cfg["env"].get("maxBaseLinearVelocity", 1.0),
            base_ang_vel_limit=self.cfg["env"].get("maxBaseAngularVelocity", 1.5)
        )
        
        # Create observation encoder
        self.observation_encoder = ObservationEncoder(
            gym=self.gym,
            sim=self.sim,
            num_envs=self.num_envs,
            device=self.device,
            hand_asset=self.hand_asset
        )
        
        # Configure observation encoder
        self.observation_encoder.configure(
            include_dof_pos=True,
            include_dof_vel=True,
            include_hand_pose=True,
            include_contact_forces=True,
            include_actions=self.cfg["env"].get("includeActionsInObs", False)
        )
        
        # Initialize observation buffers
        self.observation_encoder.initialize_buffers(self.num_dof)
        
        # Create reset manager
        self.reset_manager = ResetManager(
            gym=self.gym,
            sim=self.sim,
            num_envs=self.num_envs,
            device=self.device,
            max_episode_length=self.max_episode_length
        )
        
        # Configure randomization
        if "randomize" in self.cfg["env"] and self.cfg["env"]["randomize"]:
            self.reset_manager.set_randomization(
                randomize_positions=self.cfg["env"].get("randomizePositions", False),
                randomize_orientations=self.cfg["env"].get("randomizeOrientations", False),
                randomize_dofs=self.cfg["env"].get("randomizeDofs", False),
                position_range=self.cfg["env"].get("positionRandomizationRange", [0.05, 0.05, 0.05]),
                orientation_range=self.cfg["env"].get("orientationRandomizationRange", 0.1),
                dof_range=self.cfg["env"].get("dofRandomizationRange", 0.05)
            )
        
        # Create camera controller if viewer exists
        if hasattr(self, 'viewer') and self.viewer is not None:
            self.camera_controller = CameraController(
                gym=self.gym,
                viewer=self.viewer,
                envs=self.envs,
                num_envs=self.num_envs,
                device=self.device
            )
        else:
            self.camera_controller = None
        
        # Create fingertip visualizer
        self.fingertip_visualizer = FingertipVisualizer(
            gym=self.gym,
            envs=self.envs,
            hand_indices=self.hand_indices,
            fingerpad_handles=self.fingertip_body_handles,
            device=self.device
        )
        
        # Create success/failure tracker
        self.success_tracker = SuccessFailureTracker(
            num_envs=self.num_envs,
            device=self.device,
            cfg=self.cfg
        )
        
        # Create reward calculator
        self.reward_calculator = RewardCalculator(
            num_envs=self.num_envs,
            device=self.device,
            cfg=self.cfg
        )
        
        # Mark tensors as initialized
        self._tensors_initialized = True
        
        # Add control mode and other properties as direct attributes for easy access
        self.action_control_mode = self.action_processor.action_control_mode
        self.control_hand_base = self.action_processor.control_hand_base
        self.control_fingers = self.action_processor.control_fingers
        
        # Set observation space dimensions needed by VecTask
        self.num_observations = self.observation_encoder.num_observations
    
    def _create_envs(self):
        """
        Create environments in the simulation.
        """
        print("Creating environments...")
        
        # Define environment spacing
        env_lower = gymapi.Vec3(-1.0, -1.0, 0.0)
        env_upper = gymapi.Vec3(1.0, 1.0, 1.0)
        
        # Set up environment grid
        num_per_row = int(math.sqrt(self.num_envs))
        env_spacing = 2.0  # Increased spacing to avoid collisions
        
        self.envs = []
        
        # Create ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up
        plane_params.distance = 0
        plane_params.static_friction = 0.5
        plane_params.dynamic_friction = 0.5
        plane_params.restitution = 0.0
        
        self.gym.add_ground(self.sim, plane_params)
        
        # Create environments
        for i in range(self.num_envs):
            # Create environment
            env = self.gym.create_env(
                self.sim, env_lower, env_upper, num_per_row
            )
            
            self.envs.append(env)
            
            # Let the task add any task-specific objects
            if hasattr(self.task, 'create_task_objects'):
                self.task.create_task_objects(self.gym, self.sim, env, i)
        
        print(f"Created {self.num_envs} environments.")

    def _setup_additional_tensors(self):
        """
        Set up additional tensors needed after component initialization.
        """
        # Create observation buffer
        self.obs_buf = torch.zeros(
            (self.num_envs, self.observation_encoder.num_observations), device=self.device
        )
        
        # Create states buffer
        self.states_buf = torch.zeros(
            (self.num_envs, self.observation_encoder.num_observations), device=self.device
        )
        
        # Create reward buffer
        self.rew_buf = torch.zeros(
            (self.num_envs,), device=self.device
        )
        
        # Create reset buffer
        self.reset_buf = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.bool
        )
        
        # Create progress buffer
        self.progress_buf = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.long
        )
        
        # Share buffers with reset manager
        if hasattr(self, 'reset_manager'):
            self.reset_manager.set_buffers(self.reset_buf, self.progress_buf)
        
        # Set up action space
        if hasattr(self, 'action_processor'):
            # Calculate action space size
            if self.action_processor.control_hand_base:
                self.num_actions = self.action_processor.NUM_BASE_DOFS
            else:
                self.num_actions = 0
                
            if self.action_processor.control_fingers:
                self.num_actions += self.action_processor.NUM_ACTIVE_FINGER_DOFS
        else:
            # Fallback
            self.num_actions = 12  # 12 finger controls only (no base control)
        
        # Create the action space
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device
        )
        
        # Create extras dictionary for additional info
        self.extras = {}

    def reset_idx(self, env_ids):
        """Reset environments at specified indices."""
        try:
            if len(env_ids) == 0:
                return
            
            # Reset progress buffer for the reset environments
            self.progress_buf[env_ids] = 0
            
            # Create default DOF positions/velocities
            if not hasattr(self, 'default_dof_pos'):
                self.default_dof_pos = torch.zeros(self.num_dof, device=self.device)
                # All DOFs start at 0.0 (no offset from initial placement)
                # The hand actor itself is placed at Z=0.5m in world coordinates
                # but the ARTz DOF represents delta/offset from that initial position
            
            # We need to handle both [num_envs*num_dofs, 2] and [num_envs, num_dofs, 2] shapes
            if len(self.dof_state.shape) == 2:
                # Flat array format [num_envs*num_dofs, 2]
                # We need to compute the start/end indices for each env
                for env_id in env_ids:
                    start_idx = env_id * self.num_dof
                    end_idx = start_idx + self.num_dof
                    
                    # Set positions and velocities
                    self.dof_state[start_idx:end_idx, 0] = self.default_dof_pos  # Positions
                    self.dof_state[start_idx:end_idx, 1] = 0.0  # Velocities
            else:
                # Already in [num_envs, num_dofs, 2] format
                for env_id in env_ids:
                    # Set DOF positions and velocities
                    self.dof_state[env_id, :, 0] = self.default_dof_pos  # Positions
                    self.dof_state[env_id, :, 1] = 0.0  # Velocities
            
            # Reset hand pose in root state tensor
            for env_id in env_ids:
                if env_id < len(self.hand_indices):
                    hand_idx = self.hand_indices[env_id]
                    
                    # Set position (at default height)
                    self.root_state_tensor[env_id, hand_idx, 0:3] = torch.tensor([0.0, 0.0, 0.5], device=self.device)
                    
                    # Set rotation (identity quaternion)
                    self.root_state_tensor[env_id, hand_idx, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
                    
                    # Zero velocities
                    self.root_state_tensor[env_id, hand_idx, 7:13] = torch.zeros(6, device=self.device)
            
            # Call task-specific reset if available
            if hasattr(self.task, 'reset_task'):
                self.task.reset_task(env_ids)
            
            # Apply the updated tensor states to the simulation
            self.physics_manager.apply_tensor_states(
                gym=self.gym,
                sim=self.sim,
                env_ids=env_ids,
                dof_state=self.dof_state,
                root_state_tensor=self.root_state_tensor,
                hand_indices=self.hand_indices
            )
        except Exception as e:
            print(f"CRITICAL ERROR in reset_idx: {e}")
            import traceback
            traceback.print_exc()
            raise

    def reset(self):
        """Reset all environments and return initial observations."""
        # Reset all environments
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.reset_idx(env_ids)
        
        # Call post_physics_step to compute initial observations
        obs, rew, done, info = self.post_physics_step()
        return obs

    def pre_physics_step(self, actions):
        """Process actions before physics simulation step."""
        # Check for keyboard events if camera controller exists
        if self.camera_controller is not None:
            try:
                self.camera_controller.check_keyboard_events(
                    reset_callback=lambda env_ids: self.reset_idx(env_ids)
                )
            except Exception as e:
                print(f"ERROR in check_keyboard_events: {e}")
                import traceback
                traceback.print_exc()
                raise
        
        # Store actions
        self.actions = actions.clone()
        
        # Process actions using action processor
        try:
            task_targets = None
            if hasattr(self.task, 'get_task_dof_targets'):
                task_targets = self.task.get_task_dof_targets(
                    num_envs=self.num_envs,
                    device=self.device,
                    base_controlled=self.control_hand_base,
                    fingers_controlled=self.control_fingers
                )
            
            self.action_processor.process_actions(
                actions=self.actions,
                dof_pos=self.dof_pos,
                joint_to_control=self.hand_initializer.joint_to_control,
                active_joint_names=self.hand_initializer.active_joint_names,
                task_targets=task_targets
            )
        except Exception as e:
            print(f"ERROR in action_processor.process_actions: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Apply rule-based control for uncontrolled DOFs
        try:
            self._apply_rule_based_control()
        except Exception as e:
            print(f"ERROR in _apply_rule_based_control: {e}")
            import traceback
            traceback.print_exc()
            # Don't re-raise to avoid breaking the simulation
        
        # Update action in observation encoder for next frame
        try:
            self.observation_encoder.update_prev_actions(self.actions)
        except Exception as e:
            print(f"ERROR in update_prev_actions: {e}")
            import traceback
            traceback.print_exc()
            raise


    def post_physics_step(self):
        """Process state after physics simulation step."""
        try:
            # Refresh tensors from simulation
            self.tensor_manager.refresh_tensors(self.fingertip_indices)
            
            # Update cached tensors in observation encoder
            self.observation_encoder.update_cached_tensors(
                dof_pos=self.dof_pos,
                dof_vel=self.dof_vel,
                root_state_tensor=self.root_state_tensor
            )
            
            # Update contact forces
            self.observation_encoder.update_contact_forces(self.contact_forces)
            
            # Compute observations
            self.obs_buf, self.obs_dict = self.observation_encoder.compute_observations(
                hand_indices=self.hand_indices,
                fingertip_indices=self.fingertip_indices,
                joint_to_control=self.hand_initializer.joint_to_control,
                active_joint_names=self.hand_initializer.active_joint_names
            )
            
            # Add task-specific observations
            if hasattr(self.task, 'get_task_observations'):
                task_obs = self.task.get_task_observations(self.obs_dict)
                if task_obs:
                    self.observation_encoder.add_task_observations(task_obs)
            
            # Get task rewards
            if hasattr(self.task, 'compute_task_rewards'):
                self.rew_buf[:], task_rewards = self.task.compute_task_rewards(self.obs_dict)
                
                # Track successes
                if 'success' in task_rewards:
                    self.success_tracker.update(task_rewards['success'])
            else:
                self.rew_buf[:] = 0
            
            # Update episode progress
            self.reset_manager.increment_progress()
            
            # Check for episode termination
            if hasattr(self.task, 'check_task_reset'):
                task_reset = self.task.check_task_reset()
                self.reset_buf = self.reset_manager.check_termination(task_reset)
            else:
                self.reset_buf = self.reset_manager.check_termination()
            
            # Update fingertip visualization
            if self.fingertip_visualizer is not None:
                self.fingertip_visualizer.update_fingertip_visualization(self.contact_forces)
            
            # Update camera position if following robot
            if self.camera_controller is not None:
                # Get hand positions for camera following
                hand_positions = None
                if self.root_state_tensor is not None and self.hand_indices:
                    hand_positions = torch.zeros((self.num_envs, 3), device=self.device)
                    for i, hand_idx in enumerate(self.hand_indices):
                        if i >= self.num_envs:
                            break
                        # Get position using the correct tensor indexing (root_state_tensor shape is [num_envs, num_bodies, 13])
                        hand_positions[i] = self.root_state_tensor[i, hand_idx, :3]
                
                self.camera_controller.update_camera_position(hand_positions)
            
            # Reset environments that completed episodes
            if torch.any(self.reset_buf):
                self.reset_idx(torch.nonzero(self.reset_buf).flatten())
            
            # Physics step count tracking for auto-detecting steps per control
            self.physics_manager.mark_control_step()
            
            # Update extras
            self.extras = {
                "consecutive_successes": self.success_tracker.consecutive_successes if hasattr(self, 'success_tracker') else 0
            }
            
            return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
            
        except Exception as e:
            print(f"CRITICAL ERROR in post_physics_step: {e}")
            import traceback
            traceback.print_exc()
            raise

    def step(self, actions):
        """
        Apply actions, simulate physics, and return observations, rewards, resets, and info.
        """
        # Pre-physics: process actions
        try:
            self.pre_physics_step(actions)
        except Exception as e:
            print(f"ERROR in pre_physics_step: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Step physics simulation
        try:
            # Step physics and ensure tensors are refreshed
            # Pass refresh_tensors=True to ensure tensor data is updated
            self.physics_manager.step_physics(refresh_tensors=True)
            
            # When using GPU pipeline, we need an additional explicit refresh
            # to ensure all tensors are up-to-date
            if self.use_gpu_pipeline:
                self.physics_manager.refresh_tensors()
        except Exception as e:
            print(f"ERROR in physics step: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Post-physics: compute observations and rewards
        try:
            obs, rew, done, info = self.post_physics_step()
        except Exception as e:
            print(f"ERROR in post_physics_step: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        return obs, rew, done, info

    def set_rule_based_controllers(self, base_controller=None, finger_controller=None):
        """
        Set rule-based control functions for hand parts not controlled by the policy.
        
        Control functions should have the signature:
            def controller(env) -> torch.Tensor
        
        Where:
            - env: The environment instance (self), providing access to all properties
            - Returns: torch.Tensor of appropriate shape with target values in physical units
        
        Args:
            base_controller: Callable that returns (num_envs, 6) tensor with base DOF targets
                           in physical units (meters for translation, radians for rotation).
                           Only used if control_hand_base is False.
            finger_controller: Callable that returns (num_envs, 12) tensor with finger targets
                             in physical units (radians).
                             Only used if control_fingers is False.
        
        Example:
            def my_base_controller(env):
                t = env.progress_buf[0] * env.dt  # Get simulation time
                targets = torch.zeros((env.num_envs, 6), device=env.device)
                targets[:, 0] = 0.1 * torch.sin(t)  # Oscillate in X
                return targets
            
            env.set_rule_based_controllers(base_controller=my_base_controller)
        """
        self.rule_based_base_controller = base_controller
        self.rule_based_finger_controller = finger_controller
        
        # Validate controllers
        if base_controller is not None and self.control_hand_base:
            print("Warning: Base controller provided but control_hand_base=True. Controller will be ignored.")
        if finger_controller is not None and self.control_fingers:
            print("Warning: Finger controller provided but control_fingers=True. Controller will be ignored.")
    
    def _apply_rule_based_control(self):
        """
        Internal method to apply rule-based control using registered controller functions.
        Called automatically during pre_physics_step.
        """
        if not hasattr(self, 'action_processor') or not hasattr(self, 'dof_pos'):
            return
            
        # Apply base controller if available and base is not policy-controlled
        if not self.control_hand_base and hasattr(self, 'rule_based_base_controller') and self.rule_based_base_controller is not None:
            try:
                base_targets = self.rule_based_base_controller(self)
                if base_targets.shape == (self.num_envs, self.action_processor.NUM_BASE_DOFS):
                    # Directly set base DOF targets (raw physical values)
                    self.action_processor.current_targets[:, 0:self.action_processor.NUM_BASE_DOFS] = base_targets
                else:
                    print(f"Error: Base controller returned shape {base_targets.shape}, expected ({self.num_envs}, {self.action_processor.NUM_BASE_DOFS})")
            except Exception as e:
                print(f"Error in base controller: {e}")
                import traceback
                traceback.print_exc()
                
        # Apply finger controller if available and fingers are not policy-controlled
        if not self.control_fingers and hasattr(self, 'rule_based_finger_controller') and self.rule_based_finger_controller is not None:
            try:
                finger_targets = self.rule_based_finger_controller(self)
                if finger_targets.shape == (self.num_envs, self.action_processor.NUM_ACTIVE_FINGER_DOFS):
                    # Process finger targets through action processor to handle coupling
                    self.action_processor._apply_raw_finger_targets(
                        finger_targets=finger_targets,
                        dof_pos=self.dof_pos
                    )
                else:
                    print(f"Error: Finger controller returned shape {finger_targets.shape}, expected ({self.num_envs}, {self.action_processor.NUM_ACTIVE_FINGER_DOFS})")
            except Exception as e:
                print(f"Error in finger controller: {e}")
                import traceback
                traceback.print_exc()
                
        # Only apply targets if we actually modified them via rule-based control
        # If only policy controls both base and fingers, don't call set_dof_position_target_tensor
        # as it was already called by process_actions()
        if (not self.control_hand_base and hasattr(self, 'rule_based_base_controller') and self.rule_based_base_controller is not None) or \
           (not self.control_fingers and hasattr(self, 'rule_based_finger_controller') and self.rule_based_finger_controller is not None):
            try:
                self.gym.set_dof_position_target_tensor(
                    self.sim,
                    gymtorch.unwrap_tensor(self.action_processor.current_targets)
                )
            except Exception as e:
                print(f"Error setting DOF targets: {e}")
                import traceback
                traceback.print_exc()

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
            
    def compute_point_in_hand_frame(self, pos_world, hand_pos, hand_rot):
        """Convert a point from world frame to hand frame."""
        return point_in_hand_frame(pos_world, hand_pos, hand_rot)