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
from isaacgym import gymapi

# Then import PyTorch
import torch

# Import components
from dexhand_env.components.graphics.viewer_controller import ViewerController
from dexhand_env.components.termination.termination_manager import TerminationManager
from dexhand_env.components.reward.reward_calculator import RewardCalculator
from dexhand_env.components.physics.physics_manager import PhysicsManager
from dexhand_env.components.initialization.hand_initializer import HandInitializer
from dexhand_env.components.action.action_processor import ActionProcessor
from dexhand_env.components.observation.observation_encoder import ObservationEncoder
from dexhand_env.components.reset.reset_manager import ResetManager
from dexhand_env.components.physics.tensor_manager import TensorManager
from dexhand_env.components.graphics.video_manager import VideoManager
from dexhand_env.components.graphics.graphics_manager import GraphicsManager
from dexhand_env.components.initialization.initialization_manager import (
    InitializationManager,
)
from dexhand_env.components.step_processor import StepProcessor
from dexhand_env.components.action import DefaultActionRules, RuleBasedController

# Import utilities
from dexhand_env.utils.coordinate_transforms import point_in_hand_frame

# Import task interface
from dexhand_env.tasks.task_interface import DexTask

# Import base task class
from dexhand_env.tasks.base.vec_task import VecTask

# Import constants
from dexhand_env.constants import (
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
        video_config=None,
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
            video_config: Optional video recording configuration dictionary.
        """
        logger.info("\n===== DEXHAND ENVIRONMENT INITIALIZATION =====")
        logger.info(f"RL Device: {rl_device}")
        logger.info(f"Sim Device: {sim_device}")
        logger.info(f"Graphics Device ID: {graphics_device_id}")
        logger.info(f"Headless Mode: {headless}")
        logger.info(f"Force Render: {force_render}")
        logger.info("=============================================\n")

        # Core initialization variables
        self.cfg = cfg

        # Create section references to avoid repeated nested access
        self.env_cfg = self.cfg["env"]
        self.task_cfg = self.cfg["task"]
        self.sim_cfg = self.cfg["sim"]

        self.video_config = video_config

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
        # Set parent environment reference for reward computation
        self.task.parent_env = self

        # Task specific parameters - episodeLength is now in env section
        self.max_episode_length = self.env_cfg["episodeLength"]

        # Physics parameters
        self.physics_dt = self.sim_cfg["dt"]  # Physics simulation timestep
        self.physics_steps_per_control_step = (
            1  # Will be auto-detected by PhysicsManager
        )

        # Define model constants for the hand (imported from constants.py)
        self.NUM_BASE_DOFS = NUM_BASE_DOFS
        self.NUM_ACTIVE_FINGER_DOFS = NUM_ACTIVE_FINGER_DOFS

        # Define joint names (imported from constants.py)
        self.base_joint_names = BASE_JOINT_NAMES
        # Note: FINGER_JOINT_NAMES in constants.py includes all 20 joints (including fixed joint3_1)
        # but dexhand_base.py excludes joint3_1, so we need to filter it out
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

        # After parent initialization, use_gpu_pipeline is available
        self.use_gpu_pipeline = self.sim_cfg["use_gpu_pipeline"]
        logger.info(
            f"GPU Pipeline: {'enabled' if self.use_gpu_pipeline else 'disabled'}"
        )

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
        self.initialization_manager.create_index_mappings()

        # Additional setup (should only happen after components are created)
        self.initialization_manager.setup_additional_tensors()

        # Set up action rules (default + task-provided)
        self._setup_action_rules()

        # Add initialization flag to prevent video recording during init resets
        self._initialization_complete = False

        # Perform control cycle measurement after all setup is complete
        logger.info("Performing control cycle measurement...")
        self._perform_control_cycle_measurement()

        # Mark initialization as complete - video recording can now start
        self._initialization_complete = True

        # Initialize timing tracking for render method
        self._render_count = 0
        self._simulation_start_time = time.time()

        logger.info("DexHandBase initialization complete.")

    # ============================================================================
    # ACTION PROCESSING METHODS
    # ============================================================================

    def _setup_action_rules(self):
        """Set up action rules: default + task-provided overrides."""
        # First set up default action rule based on control mode
        self._setup_default_action_rule()

        # Check for task-provided action rules and register them
        # Pre-action rule
        if self.task.pre_action_rule is not None:
            self.action_processor.set_pre_action_rule(self.task.pre_action_rule)

        # Action rule
        if self.task.action_rule is not None:
            self.action_processor.set_action_rule(self.task.action_rule)

        # Register custom filters from task
        self.task.register_custom_filters(self.action_processor)

        # Add task-specific post-action filters to configuration
        task_filters = self.task.post_action_filters
        if task_filters:
            # Get current enabled filters and add task filters
            current_filters = self.action_processor._enabled_post_action_filters
            extended_filters = current_filters + task_filters
            self.action_processor._enabled_post_action_filters = extended_filters

    def _setup_default_action_rule(self):
        """Set up a default action rule based on control mode."""
        control_mode = self.task_cfg["controlMode"]
        DefaultActionRules.setup_default_action_rule(
            self.action_processor, control_mode
        )

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
        # For initialization, use zero rule targets
        num_active = (
            self.action_processor.NUM_BASE_DOFS
            + self.action_processor.NUM_ACTIVE_FINGER_DOFS
        )
        dummy_rule_targets = torch.zeros(
            (self.num_envs, num_active), device=self.device
        )
        self.action_processor.process_actions(dummy_actions, dummy_rule_targets)

        # Step physics
        self.physics_manager.step_physics(refresh_tensors=True)

        # Force a reset on all environments to measure full cycle
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self.reset_idx(all_env_ids)

        # Finish measurement and set control_dt
        self.physics_manager.finish_control_cycle_measurement()

        # Now that control_dt is available, finalize action processor and task
        self.action_processor.finalize_setup()
        self.task.finalize_setup()

        logger.info(
            f"Control cycle measurement complete: control_dt = {self.physics_manager.control_dt}"
        )

    # ============================================================================
    # COMPONENT INITIALIZATION METHODS
    # ============================================================================

    def _init_components(self):
        """
        Initialize all components for the environment.
        """
        logger.debug("Creating components...")

        # Create simulation (parent class only defines the method, doesn't call it)
        self.sim = self.create_sim()

        # Create graphics manager - centralizes all Isaac Gym graphics operations
        self.graphics_manager = GraphicsManager(parent=self)
        logger.debug("GraphicsManager created")

        # Create video manager with graphics manager dependency
        self.video_manager = VideoManager(parent=self)

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
        if "initialHandPos" in self.env_cfg:
            self.hand_initializer.set_initial_pose(
                pos=self.env_cfg["initialHandPos"],
                rot=self.env_cfg["initialHandRot"],
            )

        # Set contact force bodies from config
        if "contactForceBodies" in self.task_cfg:
            self.hand_initializer.set_contact_force_bodies(
                self.task_cfg["contactForceBodies"]
            )

        # Load hand asset
        self.hand_asset = self.hand_initializer.load_hand_asset(self.hand_asset_file)

        # Create environments and actors following Isaac Gym's required pattern:
        # ALL actors must be added to an env before creating the next env
        self._create_envs_and_actors()

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

        # IMPORTANT: We now use LOCAL indices within each environment:
        # 1. Local actor index: For DOF operations, typically 0 for single-actor environments
        # 2. Local rigid body indices: For accessing rigid body states within each environment
        # Isaac Gym APIs that need global indices will compute them from local indices

        # Store LOCAL indices - all environments have identical structure
        self.hand_local_rigid_body_index = rigid_body_indices[
            "hand_local_rigid_body_index"
        ]
        # hand_local_rigid_body_index is required after initialization - fail fast if missing
        if self.hand_local_rigid_body_index is None:
            raise RuntimeError(
                "hand_local_rigid_body_index is None - initialization failed"
            )

        self.fingertip_local_indices = rigid_body_indices["fingertip_local_indices"]
        self.fingerpad_local_indices = rigid_body_indices["fingerpad_local_indices"]
        self.contact_force_local_body_indices = rigid_body_indices[
            "contact_force_local_body_indices"
        ]

        # Store conversion info for debugging
        self._env0_first_body_global_idx = rigid_body_indices[
            "_env0_first_body_global_idx"
        ]
        self._num_bodies_per_env = rigid_body_indices["_num_bodies_per_env"]

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

        # Create camera for video recording if enabled (AFTER prepare_sim)
        self.video_manager.setup_video_camera(self.video_config, self.envs)

        # Set up tensors
        tensors = self.tensor_manager.setup_tensors(
            self.contact_force_local_body_indices
        )
        self.dof_state = tensors["dof_state"]
        self.dof_pos = tensors["dof_pos"]
        self.dof_vel = tensors["dof_vel"]
        self.actor_root_state_tensor = tensors["actor_root_state_tensor"]
        self.num_dof = tensors["num_dof"]
        self.dof_props = tensors["dof_props"]
        self.rigid_body_states = tensors["rigid_body_states"]
        self.contact_forces = tensors["contact_forces"]

        # Provide tensor references to the task
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
        if "controlMode" not in self.task_cfg:
            raise RuntimeError(
                "controlMode not specified in config. Must be 'position' or 'position_delta'."
            )

        action_processor_config = {
            "control_mode": self.task_cfg["controlMode"],
            "num_dof": self.num_dof,
            "policy_controls_hand_base": self.task_cfg["policyControlsHandBase"],
            "policy_controls_fingers": self.task_cfg["policyControlsFingers"],
            "finger_vel_limit": self.task_cfg["maxFingerJointVelocity"],
            "base_lin_vel_limit": self.task_cfg["maxBaseLinearVelocity"],
            "base_ang_vel_limit": self.task_cfg["maxBaseAngularVelocity"],
            "post_action_filters": [
                "velocity_clamp",
                "position_clamp",
            ],  # Standard filters
        }

        # Add optional default targets if present
        if "defaultBaseTargets" in self.task_cfg:
            action_processor_config["default_base_targets"] = self.task_cfg[
                "defaultBaseTargets"
            ]
        if "defaultFingerTargets" in self.task_cfg:
            action_processor_config["default_finger_targets"] = self.task_cfg[
                "defaultFingerTargets"
            ]

        self.action_processor.initialize_from_config(action_processor_config)

        # Note: finalize_setup() will be called after control_dt is measured

        # Create observation encoder with tensor manager reference (will be initialized later)
        self.observation_encoder = ObservationEncoder(
            parent=self,
        )

        # Create default DOF positions tensor
        default_dof_pos = torch.zeros(self.num_dof, device=self.device)
        # All DOFs start at 0.0 (no offset from initial placement)
        # The hand actor itself is placed at Z=0.5m in world coordinates
        # but the ARTz DOF represents delta/offset from that initial position

        # Create reset manager with all dependencies
        self.reset_manager = ResetManager(
            parent=self,
            dof_state=self.dof_state,
            root_state_tensor=self.actor_root_state_tensor,
            hand_local_actor_index=self.hand_local_actor_index,
            default_dof_pos=default_dof_pos,
            task=self.task,
        )

        # Create viewer controller with graphics manager dependency
        self.viewer_controller = ViewerController(parent=self, headless=self.headless)

        # Create termination manager
        self.termination_manager = TerminationManager(
            parent=self, task_cfg=self.task_cfg
        )

        # Create reward calculator
        self.reward_calculator = RewardCalculator(parent=self, task_cfg=self.task_cfg)

        # Create initialization manager
        self.initialization_manager = InitializationManager(parent=self)

        # Create step processor
        self.step_processor = StepProcessor(parent=self)

        # Create rule-based controller
        self.rule_based_controller = RuleBasedController(parent=self)

        # Initialize video recorder if video recording is enabled
        self.video_recorder = None
        self._video_episode_count = 0
        if (
            self.video_config
            and self.video_config.get("enabled", False)
            and self.video_config.get("output_dir")
        ):
            try:
                from dexhand_env.components.graphics.video.video_recorder import (
                    create_video_recorder_from_config,
                )

                self.video_recorder = create_video_recorder_from_config(
                    output_dir=self.video_config["output_dir"],
                    video_config=self.video_config,
                )
                logger.info(
                    f"Video recorder initialized: {self.video_config['output_dir']}"
                )
            except ImportError as e:
                logger.warning(f"Failed to import VideoRecorder (OpenCV required): {e}")
                self.video_recorder = None
            except Exception as e:
                logger.error(f"Failed to initialize VideoRecorder: {e}")
                self.video_recorder = None
        else:
            logger.info("Video recording disabled")

        # Initialize HTTP video streamer if streaming is enabled
        self.http_streamer = None
        if self.video_config and self.video_config.get("stream_enabled", False):
            try:
                from dexhand_env.components.graphics.video.http_video_streamer import (
                    create_http_video_streamer_from_config,
                )

                self.http_streamer = create_http_video_streamer_from_config(
                    self.video_config
                )
                success = self.http_streamer.start_server()
                if success:
                    logger.info(
                        f"HTTP video streamer started: http://{self.video_config['stream_host']}:{self.video_config['stream_port']}"
                    )
                else:
                    logger.error("Failed to start HTTP video streamer")
                    self.http_streamer = None
            except ImportError as e:
                logger.warning(
                    f"Failed to import HTTPVideoStreamer (Flask required): {e}"
                )
                self.http_streamer = None
            except Exception as e:
                logger.error(f"Failed to initialize HTTPVideoStreamer: {e}")
                self.http_streamer = None
        else:
            logger.info("HTTP video streaming disabled")

        # Real-time viewer synchronization is handled by parent class VecTask
        # via gym.sync_frame_time() - no need for duplicate sync here

        # Mark tensors as initialized
        self._tensors_initialized = True

        # Add control mode and other properties as direct attributes for easy access
        self.action_control_mode = self.action_processor.action_control_mode
        self.policy_controls_hand_base = self.action_processor.policy_controls_hand_base
        self.policy_controls_fingers = self.action_processor.policy_controls_fingers

    def _create_envs_and_actors(self):
        """
        Create environments and all actors following Isaac Gym's required pattern.
        All actors must be added to an environment before creating the next environment.
        """
        logger.info("Creating environments and actors...")

        # Define environment spacing (configurable)
        env_spacing = self.env_cfg["envSpacing"]
        half_spacing = env_spacing / 2.0
        env_lower = gymapi.Vec3(-half_spacing, -half_spacing, 0.0)
        env_upper = gymapi.Vec3(
            half_spacing, half_spacing, half_spacing * 2
        )  # Height = spacing

        # Set up environment grid
        num_per_row = int(math.sqrt(self.num_envs))

        self.envs = []
        self.hand_handles = []
        self.fingerpad_body_handles = []

        # Create ground plane first
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up
        plane_params.distance = 0
        plane_params.static_friction = 0.5
        plane_params.dynamic_friction = 0.5
        plane_params.restitution = 0.0
        self.gym.add_ground(self.sim, plane_params)

        # Create environments and actors in the correct order
        for i in range(self.num_envs):
            # Step 1: Create environment
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)

            # Step 2: Create hand actor for this environment
            hand_data = self.hand_initializer.create_hand_for_env(
                env, self.hand_asset, i
            )
            self.hand_handles.append(hand_data["hand_handle"])
            self.fingerpad_body_handles.append(hand_data["fingerpad_body_handles"])

            # Store DOF properties from first environment
            if i == 0:
                self.dof_properties_from_asset = hand_data["dof_properties"]
                self.hand_local_actor_index = (
                    0  # Hand is always actor 0 in single-actor envs
                )

                # Log DOF debug info for first environment
                self._log_dof_debug_info(
                    hand_data["dof_names"], hand_data["dof_properties"]
                )

            # Step 3: Create task-specific objects for this environment
            self.task.create_task_objects(self.gym, self.sim, env, i)

        # Update hand_initializer's internal lists so initialize_rigid_body_indices can work
        self.hand_initializer.hand_handles = self.hand_handles
        self.hand_initializer.fingerpad_body_handles = self.fingerpad_body_handles

        logger.info(f"Created {self.num_envs} environments with all actors.")

    def _log_dof_debug_info(self, dof_names, dof_properties):
        """Log DOF debug information for the first environment."""
        # Debug: Print DOF limits for first 6 joints
        logger.debug("===== BASE DOF LIMITS FROM ACTOR =====")
        for j in range(min(6, len(dof_names))):
            logger.debug(
                f"DOF {j} ({dof_names[j]}): lower={dof_properties['lower'][j]:.6f}, upper={dof_properties['upper'][j]:.6f}"
            )
        logger.debug("=====================================")

        # Log DOF names for verification
        logger.debug("===== DOF NAMES VERIFICATION =====")
        logger.debug(f"Total DOFs found: {len(dof_names)}")
        logger.debug("DOF Index -> Joint Name:")
        for j, name in enumerate(dof_names):
            # Determine joint type
            joint_type = "UNKNOWN"
            if any(base_name in name for base_name in BASE_JOINT_NAMES):
                joint_type = "BASE"
            elif any(finger_name in name for finger_name in FINGER_JOINT_NAMES):
                joint_type = "FINGER"
            logger.debug(f"  {j:2d}: {name:<20} ({joint_type})")
        logger.debug("=====================================")

    # ============================================================================
    # PROPERTIES
    # ============================================================================

    @property
    def episode_time(self):
        """Get current episode time in seconds for all environments.

        Returns:
            Tensor of shape (num_envs,) with episode time in seconds
        """
        # During initialization, control_dt might not be available yet
        if self.physics_manager.control_dt is None:
            return torch.zeros(self.num_envs, device=self.device)
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
        # During initialization, control_dt might not be available yet
        if self.physics_manager.control_dt is None:
            return 0.0
        return self.episode_step_count[env_id].item() * self.physics_manager.control_dt

    # ============================================================================
    # ENVIRONMENT LIFECYCLE METHODS
    # ============================================================================

    def reset_idx(self, env_ids):
        """Reset environments at specified indices."""
        try:
            if len(env_ids) == 0:
                return

            # Clear termination tracking FIRST to prevent stale states
            self.termination_manager.reset_tracking(env_ids)

            # Delegate all reset logic to reset_manager
            self.reset_manager.reset_idx(env_ids)

            # Reset observer internal state for the reset environments
            self.observation_encoder.reset_observer_state(env_ids)

            # Handle video recording for episode resets
            if self.video_recorder:
                # If recording is active and env 0 is being reset (episode end)
                if self.video_recorder.is_recording() and 0 in env_ids:
                    # Stop current recording
                    saved_file = self.video_recorder.stop_recording()
                    if saved_file:
                        # Only increment counter for videos that were actually saved (not empty)
                        self._video_episode_count = self._video_episode_count + 1
                        logger.info(f"Episode video saved: {saved_file}")

                # Start new recording for env 0 (new episode)
                if 0 in env_ids:
                    # Only start recording if camera system is ready AND initialization is complete
                    # This prevents recording during initialization measurement resets
                    # Check if video config is available (camera will be created lazily when needed)
                    video_system_ready = (
                        self.video_manager
                        and hasattr(self.video_manager, "_video_config")
                        and self.video_manager._video_config is not None
                    )
                    if video_system_ready and self._initialization_complete:
                        # Use NEXT episode ID for recording (current + 1)
                        episode_id = self._video_episode_count + 1
                        success = self.video_recorder.start_episode_recording(
                            episode_id
                        )
                        if success:
                            logger.info(f"Started recording episode {episode_id}")
                    else:
                        init_status = (
                            "complete"
                            if self._initialization_complete
                            else "in progress"
                        )
                        camera_status = "ready" if video_system_ready else "not ready"
                        logger.info(
                            f"Skipping recording start - initialization: {init_status}, video_system: {camera_status}"
                        )

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

        # Compute observations after reset including pre-action rule
        # This ensures obs_dict is populated with active_rule_targets before first step
        try:
            # Stage 1: Compute partial observations (exclude active_rule_targets)
            obs_dict = self.observation_encoder.compute_observations(
                exclude_components=["active_rule_targets"]
            )
        except Exception as e:
            logger.error(f"ERROR in compute_observations during reset: {e}")
            import traceback

            traceback.print_exc()
            raise

        # Stage 2: Apply pre-action rule with partial observations
        state = {"obs_dict": obs_dict, "env": self}
        active_rule_targets = self.action_processor.apply_pre_action_rule(
            self.action_processor.active_prev_targets, state
        )

        # Stage 3: Update observation with rule targets
        obs_dict["active_rule_targets"] = active_rule_targets
        self.obs_buf = self.observation_encoder.concatenate_observations(obs_dict)
        self.obs_dict = obs_dict

        # Call post_physics_step to compute initial rewards
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
            # Log occasionally to avoid spam (use episode step count instead)
            if self.episode_step_count[0] % 100 == 0:
                logger.info(
                    "Random actions mode active - generating random actions in range [-1, 1]"
                )
            self.actions = random_actions

        # Apply policy actions (action rule + post filters + coupling)
        # Note: Observations and pre-action rule are now computed in post_physics_step
        # to align with RL rollout patterns where observations for step N are computed in step N-1
        try:
            self.action_processor.process_actions(
                actions=self.actions,
                active_rule_targets=self.obs_dict["active_rule_targets"],
            )
        except Exception as e:
            logger.error(f"ERROR in action_processor.process_actions: {e}")
            import traceback

            traceback.print_exc()
            raise

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
        return self.step_processor.process_physics_step()

    def step(self, actions):
        """
        Apply actions, simulate physics, and return observations, rewards, resets, and info.
        """
        # Pre-physics: process actions
        try:
            self.pre_physics_step(actions)
        except Exception as e:
            logger.error(f"ERROR in pre_physics_step: {e}")
            import traceback

            traceback.print_exc()
            raise

        # Step physics simulation
        try:
            # Step physics and ensure tensors are refreshed
            # Pass refresh_tensors=True to ensure tensor data is updated
            self.physics_manager.step_physics(refresh_tensors=True)
        except Exception as e:
            logger.error(f"ERROR in physics step: {e}")
            import traceback

            traceback.print_exc()
            raise

        # Post-physics: compute observations and rewards
        try:
            obs, rew, done, info = self.post_physics_step()
        except Exception as e:
            logger.error(f"ERROR in post_physics_step: {e}")
            import traceback

            traceback.print_exc()
            raise

        # Render viewer if it exists OR if video recording/streaming is active
        should_render = (
            (self.viewer_controller and self.viewer_controller.viewer)
            or (self.video_recorder and self.video_recorder.is_recording())
            or (self.http_streamer and self.http_streamer.is_streaming())
        )

        # Call render if viewer is enabled OR if video recording/streaming is active
        needs_render = should_render

        if needs_render:
            self.render()  # Let it crash and show exact stack trace and array shape

        return obs, rew, done, info

    # ============================================================================
    # OBSERVATION AND CONTROL METHODS
    # ============================================================================

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
        self.rule_based_controller.set_controllers(base_controller, finger_controller)

    def _apply_rule_based_control(self):
        """
        Internal method to apply rule-based control using registered controller functions.
        Called automatically during pre_physics_step.
        """
        self.rule_based_controller.apply_rule_based_control()

    # ============================================================================
    # RENDERING AND VISUALIZATION METHODS
    # ============================================================================

    def render(self, mode="rgb_array"):
        """Draw the frame to the viewer, and check for keyboard events."""
        # Global clock synchronization
        self._render_count += 1
        current_time = time.time()

        # CRITICAL: Reset graphics state for new frame
        self.graphics_manager.reset_graphics_state()

        # CRITICAL: Step graphics pipeline ONCE before any camera operations
        # This is the centralized fix for the segfault issue
        self.graphics_manager.step_graphics()

        # Now all graphics consumers can safely operate
        result = None

        # Update camera position if following robot (moved from StepProcessor)
        if self.viewer_controller and self.viewer_controller.viewer:
            # Get hand positions for camera following
            # rigid_body_states shape: [num_envs, num_bodies, 13]
            # Extract positions for all hands using the constant index
            hand_positions = self.rigid_body_states[
                :, self.hand_local_rigid_body_index, :3
            ]
            self.viewer_controller.update_camera_position(hand_positions)

            # Delegate rendering to viewer controller with reset callback
            result = self.viewer_controller.render(
                mode,
                reset_callback=lambda env_ids: self.reset_idx(env_ids),
                obs_dict=self.obs_dict,
            )

        # Unified video capture using VideoManager (works with or without viewer)
        if (self.video_recorder or self.http_streamer) and mode == "rgb_array":
            # Use VideoManager for all video capture (unified path)
            frame = self.video_manager.capture_frame(self.envs)
            if frame is not None:
                # Add frame to video recorder if recording
                if self.video_recorder and self.video_recorder.is_recording():
                    self.video_recorder.add_frame(frame)
                # Add frame to HTTP streamer if streaming
                if self.http_streamer and self.http_streamer.is_streaming():
                    self.http_streamer.add_frame(frame)
                return frame

        # Real-time synchronization at proper abstraction level
        if (self.viewer_controller and self.viewer_controller.viewer) or (
            self.http_streamer and self.http_streamer.is_streaming()
        ):
            # Calculate expected simulation time based on control_dt
            control_dt = (
                self.physics_manager.control_dt if self.physics_manager else None
            )
            if control_dt:
                expected_sim_time = self._render_count * control_dt
                elapsed_real_time = current_time - self._simulation_start_time

                # Calculate how much time we should have taken
                if elapsed_real_time < expected_sim_time:
                    # We're ahead - sleep to maintain real-time
                    time_to_sleep = expected_sim_time - elapsed_real_time
                    time.sleep(time_to_sleep)
                else:
                    # We're behind - just call Isaac Gym sync
                    self.graphics_manager.sync_frame_time()

        # Return viewer result if available, otherwise None
        return result

    # ============================================================================
    # UTILITY METHODS
    # ============================================================================

    def close(self):
        """Close the environment."""
        # Stop video recording if active
        if self.video_recorder and self.video_recorder.is_recording():
            saved_file = self.video_recorder.stop_recording()
            if saved_file:
                logger.info(f"Final video saved on close: {saved_file}")

        # Clean up video recorder
        if self.video_recorder:
            self.video_recorder.cleanup()

        # Clean up HTTP streamer
        if self.http_streamer:
            self.http_streamer.cleanup()

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

        # Get logging preferences from config (now passed from logging.logLevel)
        log_level = self.env_cfg.get("logLevel", "INFO").upper()
        enable_debug_logs = self.env_cfg.get("enableComponentDebugLogs", False)
        force_config_logging = self.env_cfg.get("forceConfigLogging", False)

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

    @property
    def observation_space(self):
        """Return the observation space for RL libraries."""
        import gym

        return gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(self.num_observations,),
            dtype=np.float32,
        )

    @property
    def action_space(self):
        """Return the action space for RL libraries."""
        import gym

        return gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32
        )
