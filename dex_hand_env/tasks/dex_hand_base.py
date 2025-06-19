"""
Base class for DexHand tasks.

This module provides the DexHandBase class that uses composition
to delegate functionality to specialized components.
"""

import os
import sys
import math
import time
import numpy as np
from loguru import logger

# Import Gym

# Import IsaacGym first
from isaacgym import gymapi, gymtorch

# Then import PyTorch
import torch

# Import components
from dex_hand_env.components.viewer_controller import ViewerController
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

# Import constants
from dex_hand_env.constants import (
    NUM_BASE_DOFS,
    NUM_ACTIVE_FINGER_DOFS,
    BASE_JOINT_NAMES,
    FINGER_JOINT_NAMES,
    FINGERTIP_BODY_NAMES,
    FINGERPAD_BODY_NAMES,
)


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
        logger.info("\n===== DEXHAND ENVIRONMENT INITIALIZATION =====")
        logger.info(f"RL Device: {rl_device}")
        logger.info(f"Sim Device: {sim_device}")
        logger.info(f"Graphics Device ID: {graphics_device_id}")
        logger.info(f"Headless Mode: {headless}")
        logger.info(f"Force Render: {force_render}")

        # Store GPU pipeline flag for use in step() and other methods
        self.use_gpu_pipeline = cfg["sim"].get("use_gpu_pipeline", False)

        # Ensure we're using a GPU device if GPU pipeline is enabled
        if self.use_gpu_pipeline and sim_device == "cpu":
            logger.warning(
                "GPU Pipeline is enabled but using CPU device. This may cause errors. Disabling GPU pipeline."
            )
            cfg["sim"]["use_gpu_pipeline"] = False
            self.use_gpu_pipeline = False

        logger.info(
            f"GPU Pipeline: {'enabled' if self.use_gpu_pipeline else 'disabled'}"
        )
        logger.info("=============================================\n")

        # Core initialization variables
        self.cfg = cfg

        # Configure logging based on user preferences
        self._configure_logging()
        self.task = task
        self.asset_root = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "assets",
        )

        # Set device early for tensor creation
        # Note: This will be updated after parent class initialization to match actual tensor device
        self.device = rl_device

        # Update task device to match environment device
        self.task.device = self.device

        # Task specific parameters
        self.max_episode_length = self.cfg["env"]["episodeLength"]

        # Physics parameters
        self.physics_dt = self.cfg["sim"].get("dt", 0.01)  # Physics simulation timestep
        self.physics_steps_per_control_step = (
            1  # Will be auto-detected by PhysicsManager
        )

        # Define model constants for the hand (imported from constants.py)
        self.NUM_BASE_DOFS = NUM_BASE_DOFS
        self.NUM_ACTIVE_FINGER_DOFS = NUM_ACTIVE_FINGER_DOFS

        # Define joint names (imported from constants.py)
        self.base_joint_names = BASE_JOINT_NAMES
        # Note: FINGER_JOINT_NAMES in constants.py includes all 20 joints (including fixed joint3_1)
        # but dex_hand_base.py excludes joint3_1, so we need to filter it out
        self.finger_joint_names = [
            name for name in FINGER_JOINT_NAMES if name != "r_f_joint3_1"
        ]

        # Define fingertip body names (imported from constants.py)
        self.fingertip_body_names = FINGERTIP_BODY_NAMES
        self.fingerpad_body_names = FINGERPAD_BODY_NAMES

        # Default hand asset file
        self.hand_asset_file = "dexrobot_mujoco/dexrobot_mujoco/models/dexhand021_right_simplified_floating.xml"

        # Flag to track initialization state
        self._tensors_initialized = False

        # Initialize rule-based controller functions
        self.rule_based_base_controller = None
        self.rule_based_finger_controller = None

        # Initialize the base class
        logger.info("Initializing VecTask parent class...")
        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )
        logger.info("VecTask parent class initialized, creating simulation...")

        # Initialize components
        self._init_components()

        # Verify tensors were properly initialized
        # tensor_manager must exist after _init_components()
        # If it doesn't, that indicates a critical initialization failure

        if not self.tensor_manager.tensors_initialized:
            raise RuntimeError(
                "Tensors were not properly initialized. The simulation initialization has failed."
            )

        # Initialize observation dict
        self.obs_dict = {}

        # Create index mappings for convenient access (must be before _setup_additional_tensors)
        self._create_index_mappings()

        # Additional setup (should only happen after components are created)
        self._setup_additional_tensors()

        # Perform control cycle measurement after all setup is complete
        logger.info("Performing control cycle measurement...")
        self._perform_control_cycle_measurement()

        logger.info("DexHandBase initialization complete.")

    def _perform_control_cycle_measurement(self):
        """
        Perform active measurement of physics steps per control cycle.

        This is done during initialization to ensure control_dt is available
        before any external calls to reset() or step().
        """
        # Start measurement
        if not self.physics_manager.start_control_cycle_measurement():
            # Already measured, nothing to do
            return

        # Perform a dummy action to process_actions (will use zero targets)
        dummy_actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device
        )
        self.action_processor.process_actions(dummy_actions)

        # Step physics
        self.physics_manager.step_physics(refresh_tensors=True)

        # Force a reset on all environments to measure full cycle
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self.reset_idx(all_env_ids)

        # Finish measurement and set control_dt
        self.physics_manager.finish_control_cycle_measurement()

        # Now that control_dt is available, finalize action processor
        self.action_processor.finalize_setup()

        logger.info(
            f"Control cycle measurement complete: control_dt = {self.physics_manager.control_dt}"
        )

    def _init_components(self):
        """
        Initialize all components for the environment.
        """
        logger.debug("Creating components...")

        # Create simulation (parent class only defines the method, doesn't call it)
        self.sim = self.create_sim()

        # Update task with sim and gym instances early
        self.task.sim = self.sim
        self.task.gym = self.gym

        # Load task assets before creating any actors
        logger.debug("Loading task-specific assets...")
        self.task.load_task_assets()
        logger.debug("Task assets loaded successfully")

        # Create hand initializer
        self.hand_initializer = HandInitializer(
            parent=self,
            asset_root=self.asset_root,
        )

        # Joint properties (stiffness/damping) now loaded from MJCF model

        # Set initial pose from config
        if "initialHandPos" in self.cfg["env"]:
            self.hand_initializer.set_initial_pose(
                pos=self.cfg["env"].get("initialHandPos", [0.0, 0.0, 0.5]),
                rot=self.cfg["env"].get("initialHandRot", [0.0, 0.0, 0.0, 1.0]),
            )

        # Set contact force bodies from config
        if "contactForceBodies" in self.cfg["env"]:
            self.hand_initializer.set_contact_force_bodies(
                self.cfg["env"]["contactForceBodies"]
            )

        # Load hand asset
        self.hand_asset = self.hand_initializer.load_hand_asset(self.hand_asset_file)

        # Create the environments
        self._create_envs()

        # Create hands in the environments FIRST (before task objects)
        handles = self.hand_initializer.create_hands(self.envs, self.hand_asset)
        self.hand_handles = handles["hand_handles"]
        self.fingertip_body_handles = handles["fingertip_body_handles"]
        self.fingerpad_body_handles = handles["fingerpad_body_handles"]
        self.dof_properties_from_asset = handles.get("dof_properties", None)
        self.hand_actor_indices = handles["hand_actor_indices"]

        # Now create task-specific objects AFTER hands are created
        # This ensures hands are always actor index 0
        for i in range(self.num_envs):
            if hasattr(self.task, "create_task_objects"):
                self.task.create_task_objects(self.gym, self.sim, self.envs[i], i)

        # Set up viewer
        # CRITICAL: Create viewer even in headless mode for proper DOF control
        logger.debug("Creating viewer...")
        self.set_viewer()
        logger.debug("Viewer created")

        # Initialize rigid body indices after all actors have been created
        # This must happen after task objects are created but before tensor manager
        rigid_body_indices = self.hand_initializer.initialize_rigid_body_indices(
            self.envs
        )

        # IMPORTANT: Isaac Gym uses different index types for different operations:
        # 1. Actor indices: Used for DOF operations (set_dof_state, set_dof_position_target)
        #    - Stored in hand_actor_indices (list) and hand_actor_indices_tensor (tensor)
        #    - Points to the articulated actor in the environment
        # 2. Rigid body indices: Used for pose/state queries (rigid_body_states tensor)
        #    - When all environments have the same index: stored as single constant
        #    - When indices differ: stored as tensor for vectorized operations
        #    - Points to specific rigid bodies within the global rigid body array
        # These are NOT interchangeable - using the wrong index type will cause errors!

        # Store rigid body indices - all environments must have identical structure
        self.hand_rigid_body_index = rigid_body_indices["hand_rigid_body_index"]
        if self.hand_rigid_body_index is None:
            raise RuntimeError(
                "hand_rigid_body_index is None - HandInitializer should have validated identical indices"
            )

        self.fingertip_indices = rigid_body_indices["fingertip_indices"]
        self.fingerpad_indices = rigid_body_indices["fingerpad_indices"]
        self.contact_force_body_indices = rigid_body_indices[
            "contact_force_body_indices"
        ]

        # Create tensor manager after environment setup
        self.tensor_manager = TensorManager(parent=self)

        # Acquire tensor handles BEFORE prepare_sim (critical for GPU pipeline)
        self.tensor_manager.acquire_tensor_handles()

        # Pass DOF properties from asset to tensor manager if available
        if self.dof_properties_from_asset is not None:
            self.tensor_manager.set_dof_properties(self.dof_properties_from_asset)

        # CRITICAL: Call prepare_sim after all actors are created and tensors acquired
        # This is needed for both GPU pipeline and proper DOF control in headless mode
        logger.debug("Preparing simulation...")
        self.gym.prepare_sim(self.sim)
        logger.debug("Simulation prepared successfully")

        # Set up tensors
        tensors = self.tensor_manager.setup_tensors(self.contact_force_body_indices)
        self.dof_state = tensors["dof_state"]
        self.dof_pos = tensors["dof_pos"]
        self.dof_vel = tensors["dof_vel"]
        self.actor_root_state_tensor = tensors["actor_root_state_tensor"]
        self.num_dof = tensors["num_dof"]
        self.dof_props = tensors["dof_props"]
        self.rigid_body_states = tensors["rigid_body_states"]
        self.contact_forces = tensors["contact_forces"]

        # Provide tensor references to the task
        if hasattr(self.task, "set_tensor_references"):
            self.task.set_tensor_references(self.actor_root_state_tensor)

        # Create physics manager
        self.physics_manager = PhysicsManager(
            parent=self,
            physics_dt=self.physics_dt,
        )

        # Create action processor
        self.action_processor = ActionProcessor(
            parent=self,
        )

        # Initialize action processor with all configuration at once
        if "controlMode" not in self.cfg["env"]:
            raise RuntimeError(
                "controlMode not specified in config. Must be 'position' or 'position_delta'."
            )

        action_processor_config = {
            "control_mode": self.cfg["env"]["controlMode"],
            "num_dof": self.num_dof,
            "policy_controls_hand_base": self.cfg["env"]["policyControlsHandBase"],
            "policy_controls_fingers": self.cfg["env"]["policyControlsFingers"],
            "finger_vel_limit": self.cfg["env"]["maxFingerJointVelocity"],
            "base_lin_vel_limit": self.cfg["env"]["maxBaseLinearVelocity"],
            "base_ang_vel_limit": self.cfg["env"]["maxBaseAngularVelocity"],
        }

        # Add optional default targets if present
        if "defaultBaseTargets" in self.cfg["env"]:
            action_processor_config["default_base_targets"] = self.cfg["env"][
                "defaultBaseTargets"
            ]
        if "defaultFingerTargets" in self.cfg["env"]:
            action_processor_config["default_finger_targets"] = self.cfg["env"][
                "defaultFingerTargets"
            ]

        self.action_processor.initialize_from_config(action_processor_config)

        # Note: finalize_setup() will be called after control_dt is measured

        # Create observation encoder with tensor manager reference (will be initialized later)
        self.observation_encoder = ObservationEncoder(
            parent=self,
        )

        # Create reset manager with all dependencies
        self.reset_manager = ResetManager(
            parent=self,
            dof_state=self.dof_state,
            root_state_tensor=self.actor_root_state_tensor,
            hand_actor_indices=self.hand_actor_indices_tensor,
            hand_rigid_body_index=self.hand_rigid_body_index,
            task=self.task,
            max_episode_length=self.max_episode_length,
        )

        # Configure randomization
        if "randomize" in self.cfg["env"] and self.cfg["env"]["randomize"]:
            self.reset_manager.set_randomization(
                randomize_positions=self.cfg["env"].get("randomizePositions", False),
                randomize_orientations=self.cfg["env"].get(
                    "randomizeOrientations", False
                ),
                randomize_dofs=self.cfg["env"].get("randomizeDofs", False),
                position_range=self.cfg["env"].get(
                    "positionRandomizationRange", [0.05, 0.05, 0.05]
                ),
                orientation_range=self.cfg["env"].get(
                    "orientationRandomizationRange", 0.1
                ),
                dof_range=self.cfg["env"].get("dofRandomizationRange", 0.05),
            )

        # Create viewer controller
        self.viewer_controller = ViewerController(
            parent=self,
            gym=self.gym,
            sim=self.sim,
            headless=self.headless,
        )

        # Create fingertip visualizer
        self.fingertip_visualizer = FingertipVisualizer(
            parent=self,
            hand_rigid_body_index=self.hand_rigid_body_index,
            fingerpad_handles=self.fingertip_body_handles,
        )

        # Create success/failure tracker
        self.success_tracker = SuccessFailureTracker(parent=self, cfg=self.cfg)

        # Create reward calculator
        self.reward_calculator = RewardCalculator(parent=self, cfg=self.cfg)

        # Real-time viewer synchronization is handled by parent class VecTask
        # via gym.sync_frame_time() - no need for duplicate sync here

        # Performance profiling
        self.enable_performance_profiling = self.cfg["env"].get(
            "enablePerformanceProfiling", False
        )
        self.performance_log_interval = self.cfg["env"].get(
            "performanceLogInterval", 50
        )
        self.step_count = 0
        self.timing_history = {
            "pre_physics": [],
            "physics": [],
            "post_physics": [],
            "render": [],
            "total": [],
        }

        # Mark tensors as initialized
        self._tensors_initialized = True

        # Add control mode and other properties as direct attributes for easy access
        self.action_control_mode = self.action_processor.action_control_mode
        self.policy_controls_hand_base = self.action_processor.policy_controls_hand_base
        self.policy_controls_fingers = self.action_processor.policy_controls_fingers

    def _create_envs(self):
        """
        Create environments in the simulation.
        """
        logger.info("Creating environments...")

        # Define environment spacing
        env_lower = gymapi.Vec3(-1.0, -1.0, 0.0)
        env_upper = gymapi.Vec3(1.0, 1.0, 1.0)

        # Set up environment grid
        num_per_row = int(math.sqrt(self.num_envs))

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
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)

            self.envs.append(env)

        logger.info(f"Created {self.num_envs} environments.")

    def _setup_additional_tensors(self):
        """
        Set up additional tensors needed after component initialization.
        """
        # Create observation buffer
        self.obs_buf = torch.zeros(
            (self.num_envs, self.observation_encoder.num_observations),
            device=self.device,
        )

        # Create states buffer
        self.states_buf = torch.zeros(
            (self.num_envs, self.observation_encoder.num_observations),
            device=self.device,
        )

        # Create reward buffer
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device)

        # Create reset buffer
        self.reset_buf = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.bool
        )

        # Create episode step count buffer
        self.episode_step_count = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.long
        )

        # Share buffers with reset manager
        # reset_manager must exist after _init_components()
        self.reset_manager.set_buffers(self.reset_buf, self.episode_step_count)

        # Set default DOF positions for reset
        default_dof_pos = torch.zeros(self.num_dof, device=self.device)
        # All DOFs start at 0.0 (no offset from initial placement)
        # The hand actor itself is placed at Z=0.5m in world coordinates
        # but the ARTz DOF represents delta/offset from that initial position
        self.reset_manager.set_default_state(dof_pos=default_dof_pos)

        # Set up action space
        # action_processor must exist after _init_components()
        # Calculate action space size
        if self.action_processor.policy_controls_hand_base:
            self.num_actions = self.action_processor.NUM_BASE_DOFS
        else:
            self.num_actions = 0

        if self.action_processor.policy_controls_fingers:
            self.num_actions += self.action_processor.NUM_ACTIVE_FINGER_DOFS

        # Create the action space
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device
        )

        # Initialize observation encoder now that we know the action space size
        observation_keys = self.cfg["env"].get(
            "observationKeys",
            [
                "base_dof_pos",
                "base_dof_vel",
                "finger_dof_pos",
                "finger_dof_vel",
                "hand_pose",
                "contact_forces",
            ],
        )

        self.observation_encoder.initialize(
            observation_keys=observation_keys,
            joint_to_control=self.hand_initializer.joint_to_control,
            active_joint_names=self.hand_initializer.active_joint_names,
            num_actions=self.num_actions,
            action_processor=self.action_processor,
            index_mappings={
                "base_joint_to_index": self.base_joint_to_index,
                "control_name_to_index": self.control_name_to_index,
                "raw_dof_name_to_index": self.raw_dof_name_to_index,
                "finger_body_to_index": self.finger_body_to_index,
            },
        )

        # Set observation space dimensions needed by VecTask
        self.num_observations = self.observation_encoder.num_observations

        # ObservationEncoder now accesses control_dt directly from physics_manager via property decorator

        # Create extras dictionary for additional info
        self.extras = {}

    @property
    def hand_actor_indices_tensor(self):
        """Access hand actor indices as tensor for DOF operations."""
        import torch

        return torch.tensor(
            self.hand_actor_indices, dtype=torch.int32, device=self.device
        )

    @property
    def episode_time(self):
        """Get current episode time in seconds for all environments.

        Returns:
            Tensor of shape (num_envs,) with episode time in seconds
        """
        return self.episode_step_count.float() * self.physics_manager.control_dt

    @property
    def random_actions_enabled(self):
        """Check if random actions mode is enabled via viewer controller.

        Returns:
            bool: True if random actions are enabled, False otherwise
        """
        if self.viewer_controller and self.viewer_controller.viewer:
            return self.viewer_controller.random_actions_enabled
        return False

    def get_episode_time(self, env_id=0):
        """Get current episode time in seconds for a specific environment.

        Args:
            env_id: Environment index (default: 0)

        Returns:
            Float episode time in seconds
        """
        return self.episode_step_count[env_id].item() * self.physics_manager.control_dt

    # Note: fingertip_indices and fingerpad_indices are now stored directly on self

    def reset_idx(self, env_ids):
        """Reset environments at specified indices."""
        try:
            if len(env_ids) == 0:
                return

            # Delegate all reset logic to reset_manager
            self.reset_manager.reset_idx(env_ids)

        except Exception as e:
            logger.critical(f"CRITICAL ERROR in reset_idx: {e}")
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
        # Keyboard events are now handled in render() method to avoid duplicate processing

        # Store actions
        self.actions = actions.clone()

        # Check if random actions mode is enabled
        if self.random_actions_enabled:
            # Generate random actions in the range [-1, 1]
            random_actions = 2.0 * torch.rand_like(self.actions) - 1.0
            # Log occasionally to avoid spam
            if self.step_count % 100 == 0:
                logger.info(
                    "Random actions mode active - generating random actions in range [-1, 1]"
                )
            self.actions = random_actions

        # Process actions using action processor
        try:
            self.action_processor.process_actions(actions=self.actions)
        except Exception as e:
            logger.error(f"ERROR in action_processor.process_actions: {e}")
            import traceback

            traceback.print_exc()
            raise

        # Apply rule-based control for uncontrolled DOFs
        try:
            self._apply_rule_based_control()
        except Exception as e:
            logger.error(f"ERROR in _apply_rule_based_control: {e}")
            import traceback

            traceback.print_exc()
            # Don't re-raise to avoid breaking the simulation

        # Update action in observation encoder for next frame
        try:
            self.observation_encoder.update_prev_actions(self.actions)
        except Exception as e:
            logger.error(f"ERROR in update_prev_actions: {e}")
            import traceback

            traceback.print_exc()
            raise

    def post_physics_step(self):
        """Process state after physics simulation step."""
        try:
            # Refresh tensors from simulation
            self.tensor_manager.refresh_tensors(self.contact_force_body_indices)

            # Compute observations using the new simplified interface
            (
                self.obs_buf,
                self.obs_dict,
            ) = self.observation_encoder.compute_observations()

            # Task-specific observations are now handled internally by the observation encoder
            # via the _compute_task_observations method

            # Get task rewards
            if hasattr(self.task, "compute_task_rewards"):
                self.rew_buf[:], task_rewards = self.task.compute_task_rewards(
                    self.obs_dict
                )

                # Track successes
                if "success" in task_rewards:
                    self.success_tracker.update(task_rewards["success"])
            else:
                self.rew_buf[:] = 0

            # Update episode progress
            self.reset_manager.increment_progress()

            # Check for episode termination
            if hasattr(self.task, "check_task_reset"):
                task_reset = self.task.check_task_reset()
                self.reset_buf = self.reset_manager.check_termination(task_reset)
            else:
                self.reset_buf = self.reset_manager.check_termination()

            # Update fingertip visualization
            if self.fingertip_visualizer:
                self.fingertip_visualizer.update_fingertip_visualization(
                    self.contact_forces
                )

            # Update camera position if following robot
            if self.viewer_controller:
                # Get hand positions for camera following
                # rigid_body_states shape: [num_envs, num_bodies, 13]
                # Extract positions for all hands using the constant index
                hand_positions = self.rigid_body_states[
                    :, self.hand_rigid_body_index, :3
                ]
                self.viewer_controller.update_camera_position(hand_positions)

            # Reset environments that completed episodes
            if torch.any(self.reset_buf):
                self.reset_idx(torch.nonzero(self.reset_buf).flatten())

            # Physics step count tracking for auto-detecting steps per control
            self.physics_manager.mark_control_step()

            # Control_dt is now accessed directly from physics_manager via property decorators
            # No need to explicitly update components when auto-detection occurs

            # Update extras
            self.extras = {
                "consecutive_successes": self.success_tracker.consecutive_successes
                if hasattr(self, "success_tracker")
                else 0
            }

            return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

        except Exception as e:
            logger.critical(f"CRITICAL ERROR in post_physics_step: {e}")
            import traceback

            traceback.print_exc()
            raise

    def step(self, actions):
        """
        Apply actions, simulate physics, and return observations, rewards, resets, and info.
        """
        step_start_time = time.time()

        # Pre-physics: process actions
        pre_physics_start = time.time()
        try:
            self.pre_physics_step(actions)
        except Exception as e:
            logger.error(f"ERROR in pre_physics_step: {e}")
            import traceback

            traceback.print_exc()
            raise
        pre_physics_time = time.time() - pre_physics_start

        # Step physics simulation
        physics_start = time.time()
        try:
            # Step physics and ensure tensors are refreshed
            # Pass refresh_tensors=True to ensure tensor data is updated
            self.physics_manager.step_physics(refresh_tensors=True)

            # When using GPU pipeline, we need an additional explicit refresh
            # to ensure all tensors are up-to-date
            if self.use_gpu_pipeline:
                self.physics_manager.refresh_tensors()
        except Exception as e:
            logger.error(f"ERROR in physics step: {e}")
            import traceback

            traceback.print_exc()
            raise
        physics_time = time.time() - physics_start

        # Post-physics: compute observations and rewards
        post_physics_start = time.time()
        try:
            obs, rew, done, info = self.post_physics_step()
        except Exception as e:
            logger.error(f"ERROR in post_physics_step: {e}")
            import traceback

            traceback.print_exc()
            raise
        post_physics_time = time.time() - post_physics_start

        # Viewer sync is handled by parent class render() method via gym.sync_frame_time()

        # Render viewer if it exists
        render_start = time.time()
        if self.viewer_controller and self.viewer_controller.viewer:
            self.render()
        render_time = time.time() - render_start

        total_time = time.time() - step_start_time

        # Performance profiling
        if self.enable_performance_profiling:
            self._log_performance_metrics(
                pre_physics_time,
                physics_time,
                post_physics_time,
                render_time,
                total_time,
            )

        return obs, rew, done, info

    def _log_performance_metrics(
        self,
        pre_physics_time,
        physics_time,
        post_physics_time,
        render_time,
        total_time,
    ):
        """Log detailed performance metrics for profiling."""
        self.step_count += 1

        # Store timing data
        self.timing_history["pre_physics"].append(
            pre_physics_time * 1000
        )  # Convert to ms
        self.timing_history["physics"].append(physics_time * 1000)
        self.timing_history["post_physics"].append(post_physics_time * 1000)
        self.timing_history["render"].append(render_time * 1000)
        self.timing_history["total"].append(total_time * 1000)

        # Log detailed metrics every N steps
        if self.step_count % self.performance_log_interval == 0:
            # Calculate averages over the last interval
            def avg_last_n(times, n):
                return sum(times[-n:]) / min(len(times), n) if times else 0.0

            n = self.performance_log_interval
            avg_pre = avg_last_n(self.timing_history["pre_physics"], n)
            avg_physics = avg_last_n(self.timing_history["physics"], n)
            avg_post = avg_last_n(self.timing_history["post_physics"], n)
            avg_render = avg_last_n(self.timing_history["render"], n)
            avg_total = avg_last_n(self.timing_history["total"], n)

            # Calculate percentages
            pre_pct = (avg_pre / avg_total * 100) if avg_total > 0 else 0
            physics_pct = (avg_physics / avg_total * 100) if avg_total > 0 else 0
            post_pct = (avg_post / avg_total * 100) if avg_total > 0 else 0
            render_pct = (avg_render / avg_total * 100) if avg_total > 0 else 0

            # Calculate real-time performance
            target_dt_ms = (
                self.physics_manager.control_dt * 1000
            )  # Target step time in ms
            performance_factor = (target_dt_ms / avg_total) if avg_total > 0 else 0

            logger.info(
                f"=== PERFORMANCE PROFILE (Steps {self.step_count-n+1}-{self.step_count}) ==="
            )
            logger.info(
                f"Total step time: {avg_total:.2f}ms (target: {target_dt_ms:.2f}ms)"
            )
            logger.info(f"Real-time factor: {performance_factor:.1%}")
            logger.info("Component breakdown:")
            logger.info(f"  Pre-physics:  {avg_pre:6.2f}ms ({pre_pct:5.1f}%)")
            logger.info(f"  Physics:      {avg_physics:6.2f}ms ({physics_pct:5.1f}%)")
            logger.info(f"  Post-physics: {avg_post:6.2f}ms ({post_pct:5.1f}%)")
            logger.info(f"  Rendering:    {avg_render:6.2f}ms ({render_pct:5.1f}%)")
            logger.info("=" * 55)

            # Warn if running significantly slower than real-time
            if performance_factor < 0.5:  # Less than 50% real-time
                logger.warning(
                    f"Simulation running at {performance_factor:.1%} of real-time. "
                    "Consider reducing environment count or simplifying the task."
                )

    def get_observations_dict(self):
        """
        Get the current observation dictionary for external access.

        Returns:
            Dictionary containing all computed observations
        """
        # obs_dict is initialized in __init__ and should always exist
        return self.obs_dict.copy()

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
                t = env.episode_step_count[0] * env.dt  # Get simulation time
                targets = torch.zeros((env.num_envs, 6), device=env.device)
                targets[:, 0] = 0.1 * torch.sin(t)  # Oscillate in X
                return targets

            env.set_rule_based_controllers(base_controller=my_base_controller)
        """
        self.rule_based_base_controller = base_controller
        self.rule_based_finger_controller = finger_controller

        # Validate controllers
        if base_controller is not None and self.policy_controls_hand_base:
            logger.warning(
                "Base controller provided but policy_controls_hand_base=True. Controller will be ignored."
            )
        if finger_controller is not None and self.policy_controls_fingers:
            logger.warning(
                "Finger controller provided but policy_controls_fingers=True. Controller will be ignored."
            )

    def _apply_rule_based_control(self):
        """
        Internal method to apply rule-based control using registered controller functions.
        Called automatically during pre_physics_step.
        """
        # action_processor and dof_pos must be initialized by this point
        # If they're not, that indicates an initialization bug

        # Apply base controller if available and base is not policy-controlled
        if (
            not self.policy_controls_hand_base
            and hasattr(self, "rule_based_base_controller")
            and self.rule_based_base_controller is not None
        ):
            try:
                base_targets = self.rule_based_base_controller(self)
                if base_targets.shape == (
                    self.num_envs,
                    self.action_processor.NUM_BASE_DOFS,
                ):
                    # Directly set base DOF targets (raw physical values)
                    self.action_processor.current_targets[
                        :, 0 : self.action_processor.NUM_BASE_DOFS
                    ] = base_targets
                else:
                    logger.error(
                        f"Base controller returned shape {base_targets.shape}, expected ({self.num_envs}, {self.action_processor.NUM_BASE_DOFS})"
                    )
            except Exception as e:
                logger.error(f"Error in base controller: {e}")
                import traceback

                traceback.print_exc()

        # Apply finger controller if available and fingers are not policy-controlled
        if (
            not self.policy_controls_fingers
            and hasattr(self, "rule_based_finger_controller")
            and self.rule_based_finger_controller is not None
        ):
            try:
                finger_targets = self.rule_based_finger_controller(self)
                if finger_targets.shape == (
                    self.num_envs,
                    self.action_processor.NUM_ACTIVE_FINGER_DOFS,
                ):
                    # Apply finger coupling by creating full active targets
                    active_targets = torch.zeros(
                        (self.num_envs, 18), device=self.device  # 6 base + 12 fingers
                    )
                    # Copy current base targets
                    active_targets[:, :6] = self.action_processor.current_targets[:, :6]
                    # Set finger targets
                    active_targets[:, 6:] = finger_targets

                    # Apply coupling to get full DOF targets and overwrite current_targets
                    self.action_processor.current_targets = (
                        self.action_processor.apply_coupling(active_targets)
                    )
                else:
                    logger.error(
                        f"Finger controller returned shape {finger_targets.shape}, expected ({self.num_envs}, {self.action_processor.NUM_ACTIVE_FINGER_DOFS})"
                    )
            except Exception as e:
                logger.error(f"Error in finger controller: {e}")
                import traceback

                traceback.print_exc()

        # Only apply targets if we actually modified them via rule-based control
        # If only policy controls both base and fingers, don't call set_dof_position_target_tensor
        # as it was already called by process_actions()
        if (
            not self.policy_controls_hand_base
            and hasattr(self, "rule_based_base_controller")
            and self.rule_based_base_controller is not None
        ) or (
            not self.policy_controls_fingers
            and hasattr(self, "rule_based_finger_controller")
            and self.rule_based_finger_controller is not None
        ):
            try:
                self.gym.set_dof_position_target_tensor(
                    self.sim,
                    gymtorch.unwrap_tensor(self.action_processor.current_targets),
                )
            except Exception as e:
                logger.error(f"Error setting DOF targets: {e}")
                import traceback

                traceback.print_exc()

    def _create_index_mappings(self):
        """
        Create index mappings for convenient access to tensors by key names.
        """
        logger.debug("Creating index mappings...")

        # 1. Base joint name to index mapping (ARTx, ARTy, etc. -> 0-5)
        self.base_joint_to_index = {}
        for i, joint_name in enumerate(self.base_joint_names):
            self.base_joint_to_index[joint_name] = i

        # 2. Control name to active finger DOF index mapping (th_dip, etc. -> 0-11)
        self.control_name_to_index = {}
        # hand_initializer must exist and have active_joint_names after _init_components()
        for i, control_name in enumerate(self.hand_initializer.active_joint_names):
            self.control_name_to_index[control_name] = i

        # 3. Raw finger DOF name to raw DOF tensor index mapping (r_f_joint1_1, etc. -> 0-25)
        self.raw_dof_name_to_index = {}
        # observation_encoder must exist but might not have dof_names yet
        if self.observation_encoder.dof_names:
            for i, dof_name in enumerate(self.observation_encoder.dof_names):
                self.raw_dof_name_to_index[dof_name] = i

        # 4. Finger name + pad/tip to body tensor index mapping
        self.finger_body_to_index = {}

        # Map fingertip body names to indices
        for i, tip_name in enumerate(self.fingertip_body_names):
            finger_name = tip_name.replace(
                "_tip", ""
            )  # e.g., "r_f_link1_tip" -> "r_f_link1"
            self.finger_body_to_index[f"{finger_name}_tip"] = ("fingertip", i)

        # Map fingerpad body names to indices
        for i, pad_name in enumerate(self.fingerpad_body_names):
            finger_name = pad_name.replace(
                "_pad", ""
            )  # e.g., "r_f_link1_pad" -> "r_f_link1"
            self.finger_body_to_index[f"{finger_name}_pad"] = ("fingerpad", i)

        logger.debug("Created mappings:")
        logger.debug(f"  Base joints: {len(self.base_joint_to_index)} entries")
        logger.debug(f"  Control names: {len(self.control_name_to_index)} entries")
        logger.debug(f"  Raw DOF names: {len(self.raw_dof_name_to_index)} entries")
        logger.debug(f"  Finger bodies: {len(self.finger_body_to_index)} entries")

    def render(self, mode="rgb_array"):
        """Draw the frame to the viewer, and check for keyboard events."""
        if self.viewer_controller and self.viewer_controller.viewer:
            # Delegate rendering to viewer controller with reset callback
            result = self.viewer_controller.render(
                mode, reset_callback=lambda env_ids: self.reset_idx(env_ids)
            )

            # Handle virtual display if needed
            if self.virtual_display and mode == "rgb_array":
                img = self.virtual_display.grab()
                return np.array(img)

            return result

    def close(self):
        """Close the environment."""
        # ViewerController handles viewer destruction
        if self.viewer_controller and self.viewer_controller.viewer:
            self.gym.destroy_viewer(self.viewer_controller.viewer)

        # Destroy simulation
        if self.sim:
            self.gym.destroy_sim(self.sim)
            self.sim = None

    def _configure_logging(self):
        """Configure logging based on user preferences in config.

        Only configures logging if no handlers are already present (respects command-line setup).
        """
        # Check if loguru already has handlers (e.g., from command-line setup)
        existing_handlers = len(logger._core.handlers)

        # Get logging preferences from config
        log_level = self.cfg.get("env", {}).get("logLevel", "INFO").upper()
        enable_debug_logs = self.cfg.get("env", {}).get(
            "enableComponentDebugLogs", False
        )
        force_config_logging = self.cfg.get("env", {}).get("forceConfigLogging", False)

        # Only configure if no handlers exist or explicitly forced via config
        if existing_handlers == 0 or force_config_logging:
            # Map string levels to loguru levels
            level_mapping = {
                "DEBUG": "DEBUG",
                "INFO": "INFO",
                "WARNING": "WARNING",
                "ERROR": "ERROR",
                "CRITICAL": "CRITICAL",
            }

            if log_level in level_mapping:
                if force_config_logging and existing_handlers > 0:
                    logger.remove()  # Remove existing handlers only if forced
                    logger.debug("Existing logging configuration overridden by config")

                # Configure loguru to use the specified level
                logger.add(
                    sys.stderr,
                    level=level_mapping[log_level],
                    format="{time:HH:mm:ss} | {level:8} | {message}",
                    colorize=True,
                )

                if log_level == "DEBUG" or enable_debug_logs:
                    logger.debug(
                        f"Logging configured: level={log_level}, component_debug={enable_debug_logs}"
                    )
            else:
                logger.warning(f"Unknown log level '{log_level}', using INFO")
        else:
            logger.debug(
                f"Logging already configured ({existing_handlers} handlers), respecting existing setup"
            )

    def compute_point_in_hand_frame(self, pos_world, hand_pos, hand_rot):
        """Convert a point from world frame to hand frame."""
        return point_in_hand_frame(pos_world, hand_pos, hand_rot)
