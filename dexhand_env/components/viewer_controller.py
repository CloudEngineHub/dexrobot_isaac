"""
Viewer controller component for DexHand environment.

This module provides viewer interaction functionality for the DexHand environment,
including keyboard shortcuts for camera control, robot selection, and environment resets.

Keyboard Shortcuts:
    R     - Reset all environments
    E     - Reset current Environment (the one being followed)
    G     - Toggle between Global view and single robot view
    UP    - Previous robot (in single robot mode)
    DOWN  - Next robot (in single robot mode)
    SPACE - Toggle random actions mode
    C     - Toggle contact force visualization
"""

# Import IsaacGym first
from isaacgym import gymapi

# Then import PyTorch
import torch

# Import loguru
from loguru import logger


class ViewerController:
    """
    Handles viewer interactions and keyboard shortcuts for the DexHand environment.

    This component provides functionality to:
    - Toggle between different camera views (free, rear, right, bottom)
    - Switch between single robot follow and global view modes
    - Navigate between different robots in the environment
    - Reset selected environments
    - Update camera position based on the current mode
    """

    def __init__(self, parent, gym, sim, env_handles, headless):
        """
        Initialize the viewer controller and create the viewer.

        Args:
            parent: Parent object (typically DexHandBase) that provides shared properties
            gym: IsaacGym instance
            sim: Simulation instance
            env_handles: List of environment handles for camera targeting
            headless: Whether running in headless mode
        """
        self.parent = parent
        self.gym = gym
        self.sim = sim
        self.env_handles = env_handles
        self.headless = headless

        # Create viewer if not in headless mode
        if not self.headless:
            self.viewer = self._create_viewer()
        else:
            self.viewer = None

        # Camera control state
        self.camera_view_mode = "rear"  # Options: "free", "rear", "right", "bottom"
        self.camera_follow_mode = (
            "single"  # Options: "single" (follow one robot), "global" (overview)
        )
        self.follow_robot_index = 0  # Which robot to follow in single mode

        # Random actions toggle state
        self.random_actions_enabled = False

        # Track whether keyboard events have been subscribed
        self._keyboard_subscribed = False

        # Contact force visualization settings - load from config if available
        contact_viz_config = (
            parent.cfg.get("env", {}).get("contactVisualization", {})
            if hasattr(parent, "cfg")
            else {}
        )
        self.enable_contact_visualization = contact_viz_config.get("enabled", True)
        self.contact_force_threshold = contact_viz_config.get(
            "forceThreshold", 1.0
        )  # Newtons
        self.contact_force_max_intensity = contact_viz_config.get(
            "forceMaxIntensity", 10.0
        )  # Newtons

        # Load default color from config or use default gray
        default_color = contact_viz_config.get("defaultColor", [0.7, 0.7, 0.7])
        self.default_body_color = gymapi.Vec3(
            default_color[0], default_color[1], default_color[2]
        )

        # Load base color from config or use default
        base_color = contact_viz_config.get("baseColor", [1.0, 0.2, 0.2])
        self.contact_base_color = gymapi.Vec3(
            base_color[0], base_color[1], base_color[2]
        )

        # Initialize keyboard events if viewer is available
        if self.viewer is not None:
            self.subscribe_keyboard_events()

    def _create_viewer(self):
        """Create the viewer and set initial camera position."""
        # Create viewer
        viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())

        # Subscribe to base keyboard shortcuts
        self.gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_ESCAPE, "QUIT")

        # Set initial camera position based on up axis
        sim_params = self.gym.get_sim_params(self.sim)
        if sim_params.up_axis == gymapi.UP_AXIS_Z:
            cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
            cam_target = gymapi.Vec3(10.0, 15.0, 0.0)
        else:
            cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
            cam_target = gymapi.Vec3(10.0, 0.0, 15.0)

        self.gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

        return viewer

    @property
    def envs(self):
        """Get environments from parent."""
        return self.parent.envs

    @property
    def num_envs(self):
        """Get number of environments from parent."""
        return self.parent.num_envs

    @property
    def device(self):
        """Get device from parent."""
        return self.parent.device

    def subscribe_keyboard_events(self):
        """
        Subscribe to keyboard events for interactive control in the viewer.

        Sets up keyboard shortcuts for:
        - Enter: Toggle camera view mode (free, rear, right, bottom)
        - G: Toggle between single robot follow and global view
        - Up/Down arrows: Navigate to previous/next robot to follow (in single mode)
        - E: Reset the currently selected environment
        - Space: Toggle random actions mode
        - C: Toggle contact force visualization
        """
        if self.viewer is None:
            return

        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_ENTER, "toggle view mode"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_G, "toggle follow mode"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_UP, "previous robot"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_DOWN, "next robot"
        )
        # Use KEY_E for "reset Environment" (single env) to avoid confusion with Pause
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_E, "reset environment"
        )
        # Spacebar to toggle random actions
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_SPACE, "toggle random actions"
        )
        # C to toggle contact force visualization
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_C, "toggle contact visualization"
        )

        self._keyboard_subscribed = True

    def check_keyboard_events(self, reset_callback=None):
        """
        Process keyboard events for interactive control.

        Args:
            reset_callback: Callback function to reset environments, takes env_ids as argument

        Returns:
            Boolean indicating whether any keyboard events were processed
        """
        if self.viewer is None:
            return False

        # Make sure keyboard events are subscribed
        if not self._keyboard_subscribed:
            self.subscribe_keyboard_events()

        try:
            # Process all queued events
            events_processed = False
            for evt in self.gym.query_viewer_action_events(self.viewer):
                events_processed = True

                if evt.action == "toggle view mode" and evt.value > 0:
                    # Cycle through view modes: free -> rear -> right -> bottom -> free
                    view_modes = ["free", "rear", "right", "bottom"]
                    current_idx = view_modes.index(self.camera_view_mode)
                    self.camera_view_mode = view_modes[
                        (current_idx + 1) % len(view_modes)
                    ]

                    # Display current camera state
                    view_names = {
                        "free": "Free Camera",
                        "rear": "Rear View",
                        "right": "Right View",
                        "bottom": "Bottom View",
                    }
                    follow_text = (
                        f"following robot {self.follow_robot_index}"
                        if self.camera_follow_mode == "single"
                        else "global view"
                    )
                    logger.info(
                        f"Camera: {view_names[self.camera_view_mode]} ({follow_text})"
                    )
                elif evt.action == "toggle follow mode" and evt.value > 0:
                    # Toggle between single robot follow and global view
                    self.camera_follow_mode = (
                        "global" if self.camera_follow_mode == "single" else "single"
                    )
                    follow_text = (
                        f"following robot {self.follow_robot_index}"
                        if self.camera_follow_mode == "single"
                        else "global view"
                    )
                    view_names = {
                        "free": "Free Camera",
                        "rear": "Rear View",
                        "right": "Right View",
                        "bottom": "Bottom View",
                    }
                    logger.info(
                        f"Camera: {view_names[self.camera_view_mode]} ({follow_text})"
                    )
                elif evt.action == "previous robot" and evt.value > 0:
                    # Move to previous robot (only in single mode)
                    if self.camera_follow_mode == "single":
                        self.follow_robot_index = (
                            self.follow_robot_index - 1
                        ) % self.num_envs
                        logger.info(f"Following robot {self.follow_robot_index}")
                    else:
                        logger.warning(
                            "Cannot change robot in global view mode. Press G to switch to single robot mode."
                        )
                elif evt.action == "next robot" and evt.value > 0:
                    # Move to next robot (only in single mode)
                    if self.camera_follow_mode == "single":
                        self.follow_robot_index = (
                            self.follow_robot_index + 1
                        ) % self.num_envs
                        logger.info(f"Following robot {self.follow_robot_index}")
                    else:
                        logger.warning(
                            "Cannot change robot in global view mode. Press G to switch to single robot mode."
                        )
                elif (
                    evt.action == "reset environment"
                    and evt.value > 0
                    and reset_callback is not None
                ):
                    # Reset only the selected environment
                    logger.info(f"Resetting robot {self.follow_robot_index}")
                    reset_callback(
                        torch.tensor([self.follow_robot_index], device=self.device)
                    )
                elif evt.action == "toggle random actions" and evt.value > 0:
                    # Toggle random actions mode
                    self.random_actions_enabled = not self.random_actions_enabled
                    state_text = (
                        "ENABLED" if self.random_actions_enabled else "DISABLED"
                    )
                    logger.info(f"Random actions {state_text} (press SPACE to toggle)")
                elif evt.action == "toggle contact visualization" and evt.value > 0:
                    # Toggle contact force visualization
                    self.enable_contact_visualization = (
                        not self.enable_contact_visualization
                    )
                    if self.enable_contact_visualization:
                        logger.info(
                            f"Contact force visualization ENABLED (press C to toggle) - "
                            f"Threshold: {self.contact_force_threshold}N, "
                            f"Max intensity: {self.contact_force_max_intensity}N"
                        )
                    else:
                        logger.info(
                            "Contact force visualization DISABLED (press C to toggle)"
                        )

            return events_processed
        except Exception:
            # Handle exceptions silently - this can happen during initialization
            return False

    def update_camera_position(self, hand_positions):
        """
        Update camera position based on the following mode.

        Args:
            hand_positions: Tensor of hand positions for all environments

        Returns:
            Boolean indicating whether camera was updated
        """
        if self.viewer is None:
            return False

        # Only update camera if we're not in free mode
        if self.camera_view_mode == "free":
            return False

        # Safety check for valid input and index
        if hand_positions is None or hand_positions.shape[0] == 0:
            return False

        # Define different camera positions based on the viewing mode
        camera_offsets = {
            "rear": gymapi.Vec3(-1.0, 0.0, 0.6),  # Behind the hand
            "right": gymapi.Vec3(0.0, -1.0, 0.6),  # From the right side
            "bottom": gymapi.Vec3(0.0, 0.3, -1.0),  # Looking up from below
        }
        camera_offset = camera_offsets.get(
            self.camera_view_mode, gymapi.Vec3(-1.0, 0.0, 0.6)
        )

        # Determine camera target based on follow mode
        if self.camera_follow_mode == "single":
            # Follow a specific robot
            safe_index = min(self.follow_robot_index, hand_positions.shape[0] - 1)
            target_pos = hand_positions[safe_index].cpu().numpy()
        else:
            # Global view - focus on center of all robots
            target_pos = hand_positions.mean(dim=0).cpu().numpy()
            # For global view, increase camera distance
            camera_offset = gymapi.Vec3(
                camera_offset.x * 2.0,
                camera_offset.y * 2.0,
                camera_offset.z + 1.0
                if self.camera_view_mode != "bottom"
                else camera_offset.z - 1.0,
            )

        try:
            # Set camera position and target
            cam_pos = gymapi.Vec3(
                target_pos[0] + camera_offset.x,
                target_pos[1] + camera_offset.y,
                target_pos[2] + camera_offset.z,
            )
            cam_target = gymapi.Vec3(target_pos[0], target_pos[1], target_pos[2])

            # Determine which environment to use for camera coordinate system
            if self.camera_follow_mode == "single" and self.env_handles:
                # Use the specific environment we're following
                safe_index = min(self.follow_robot_index, len(self.env_handles) - 1)
                env_handle = self.env_handles[safe_index]

                # Debug: Log camera positioning for environment switching
                from loguru import logger

                if (
                    hasattr(self, "_last_follow_index")
                    and self._last_follow_index != safe_index
                ):
                    logger.debug(
                        f"Camera switching from env {self._last_follow_index} to env {safe_index}"
                    )
                    logger.debug(
                        f"  Target position: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]"
                    )
                self._last_follow_index = safe_index
            else:
                # For global view or if no env handles available, use None (world coordinates)
                env_handle = None

            # Update camera with correct environment coordinate system
            self.gym.viewer_camera_look_at(self.viewer, env_handle, cam_pos, cam_target)
            return True
        except Exception:
            # Handle exceptions silently - this can happen during initialization
            return False

    def update_contact_force_colors(self, contact_forces):
        """
        Update body colors based on contact forces.

        Colors bodies red with intensity proportional to contact force magnitude.
        Bodies with forces below threshold return to default color.

        Args:
            contact_forces: Tensor of contact forces [num_envs, num_bodies, 3]

        Returns:
            Boolean indicating whether colors were updated
        """
        if self.viewer is None or not self.enable_contact_visualization:
            return False

        # Get LOCAL contact force body indices from parent
        contact_body_local_indices = self.parent.contact_force_local_body_indices
        if not contact_body_local_indices:
            return False

        # Get number of bodies per environment for global index computation
        num_bodies_per_env = self.parent.rigid_body_states.shape[1]

        # Calculate contact force magnitudes
        force_magnitudes = torch.norm(contact_forces, dim=2)  # [num_envs, num_bodies]
        has_contact = force_magnitudes > self.contact_force_threshold

        # Update colors for each environment
        for env_idx in range(self.num_envs):
            # For each contact body (using local indices)
            for body_idx, local_body_idx in enumerate(contact_body_local_indices):
                # Compute global body index for Isaac Gym API
                global_body_idx = env_idx * num_bodies_per_env + local_body_idx

                if has_contact[env_idx, body_idx]:
                    # Calculate color based on force magnitude
                    force_mag = force_magnitudes[env_idx, body_idx].item()

                    # Normalize force between threshold and max
                    normalized_force = (force_mag - self.contact_force_threshold) / (
                        self.contact_force_max_intensity - self.contact_force_threshold
                    )
                    normalized_force = max(
                        0.0, min(1.0, normalized_force)
                    )  # Clamp to [0, 1]

                    # Interpolate between default color and base color
                    color = gymapi.Vec3(
                        self.default_body_color.x
                        + (self.contact_base_color.x - self.default_body_color.x)
                        * normalized_force,
                        self.default_body_color.y
                        + (self.contact_base_color.y - self.default_body_color.y)
                        * normalized_force,
                        self.default_body_color.z
                        + (self.contact_base_color.z - self.default_body_color.z)
                        * normalized_force,
                    )
                else:
                    # No contact - use default color
                    color = self.default_body_color

                # Set the rigid body color
                # Note: Using hand_handles from parent for actor handle
                self.gym.set_rigid_body_color(
                    self.envs[env_idx],
                    self.parent.hand_handles[env_idx],
                    global_body_idx,
                    gymapi.MESH_VISUAL,
                    color,
                )

        return True

    def render(self, mode="rgb_array", reset_callback=None):
        """
        Handle viewer rendering and keyboard events.

        Args:
            mode: Rendering mode (currently only supports "rgb_array")
            reset_callback: Callback function to reset environments, takes env_ids as argument

        Returns:
            None or image array if in rgb_array mode
        """
        if self.viewer is None:
            return None

        # Check for window closed
        if self.gym.query_viewer_has_closed(self.viewer):
            import sys

            sys.exit()

        # Process all keyboard events using check_keyboard_events
        # This handles E, G, UP, DOWN keys properly
        self.check_keyboard_events(reset_callback=reset_callback)

        # Fetch results if using GPU
        if hasattr(self.parent, "device") and self.parent.device != "cpu":
            self.gym.fetch_results(self.sim, True)

        # Update contact force visualization if enabled
        if self.enable_contact_visualization and hasattr(self.parent, "contact_forces"):
            self.update_contact_force_colors(self.parent.contact_forces)

        # Step graphics and render
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)

        # Synchronize to real-time
        self.gym.sync_frame_time(self.sim)

        return None
