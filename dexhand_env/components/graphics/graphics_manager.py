"""
Graphics Manager component for DexHand environment.

This module provides a unified facade for Isaac Gym graphics operations,
centralizing the required step_graphics() call and camera management.
"""

from loguru import logger
from isaacgym import gymapi
import numpy as np


class GraphicsManager:
    """
    Facade for Isaac Gym graphics operations.

    This component centralizes all Isaac Gym graphics calls to ensure proper
    ordering (step_graphics before camera operations) and provides a clean
    interface for ViewerController and VideoManager.

    Key responsibilities:
    - Execute step_graphics() at the right time
    - Manage camera creation and lifecycle
    - Provide safe frame capture operations
    - Handle viewer rendering operations
    """

    def __init__(self, parent):
        """Initialize GraphicsManager with parent environment reference."""
        self.parent = parent
        self.gym = parent.gym
        self.sim = parent.sim

        # Camera registry
        self._cameras = {}  # name -> camera_handle mapping
        self._camera_envs = {}  # name -> env_handle mapping

        # Graphics state tracking
        self._graphics_stepped = False

    @property
    def device(self):
        """Access device from parent (single source of truth)."""
        return self.parent.device

    def step_graphics(self):
        """
        Execute Isaac Gym graphics step.

        CRITICAL: This must be called before any camera operations in each frame.
        This is the centralized location for this requirement.
        """
        self.gym.step_graphics(self.sim)
        self._graphics_stepped = True

    def create_camera(
        self, name: str, env_handle, camera_props: gymapi.CameraProperties = None
    ):
        """
        Create and register a camera.

        Args:
            name: Unique identifier for the camera
            env_handle: Environment handle for camera placement
            camera_props: Camera properties (if None, uses defaults)

        Returns:
            Camera name for future operations
        """
        if name in self._cameras:
            logger.warning(f"Camera '{name}' already exists, skipping creation")
            return name

        # Use provided properties or create defaults
        if camera_props is None:
            camera_props = gymapi.CameraProperties()
            camera_props.horizontal_fov = 75.0
            camera_props.width = 1920
            camera_props.height = 1080
            camera_props.enable_tensors = False  # Avoid GPU pipeline issues

        # Create camera sensor
        camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)

        if camera_handle is None or camera_handle == -1:
            logger.error(f"Failed to create camera '{name}' - handle is None or -1")
            return None

        # Register camera
        self._cameras[name] = camera_handle
        self._camera_envs[name] = env_handle

        logger.info(f"Created camera '{name}' with handle {camera_handle}")
        return name

    def set_camera_location(
        self, camera_name: str, position: gymapi.Vec3, target: gymapi.Vec3
    ):
        """
        Set camera position and orientation.

        Args:
            camera_name: Name of the camera to position
            position: Camera position in world coordinates
            target: Point camera should look at
        """
        if camera_name not in self._cameras:
            logger.error(f"Camera '{camera_name}' not found")
            return False

        camera_handle = self._cameras[camera_name]
        env_handle = self._camera_envs[camera_name]

        try:
            self.gym.set_camera_location(camera_handle, env_handle, position, target)
            return True
        except Exception as e:
            logger.error(f"Failed to set camera '{camera_name}' location: {e}")
            return False

    def render_all_cameras(self):
        """
        Render all camera sensors.

        REQUIRES: step_graphics() must have been called first.
        """
        if not self._graphics_stepped:
            logger.error(
                "render_all_cameras() called before step_graphics() - this will cause issues"
            )
            return False

        try:
            self.gym.render_all_camera_sensors(self.sim)
            return True
        except Exception as e:
            logger.error(f"Failed to render camera sensors: {e}")
            return False

    def capture_frame(self, camera_name: str) -> np.ndarray:
        """
        Capture frame from specified camera.

        REQUIRES: step_graphics() and render_all_cameras() must have been called first.

        Args:
            camera_name: Name of the camera to capture from

        Returns:
            Frame as numpy array, or None if capture failed
        """
        if camera_name not in self._cameras:
            logger.error(f"Camera '{camera_name}' not found")
            return None

        if not self._graphics_stepped:
            logger.error(
                f"capture_frame('{camera_name}') called before step_graphics() - this will cause segfault"
            )
            return None

        camera_handle = self._cameras[camera_name]
        env_handle = self._camera_envs[camera_name]

        # Get the image from the camera
        frame = self.gym.get_camera_image(
            self.sim, env_handle, camera_handle, gymapi.IMAGE_COLOR
        )

        if frame is None:
            raise RuntimeError(f"Camera '{camera_name}' returned None frame")

        # Convert RGBA to RGB - Isaac Gym returns 2D flattened format [width, height*4]
        # Need to reshape to [width, height, 4] then extract RGB channels
        width, height_times_4 = frame.shape
        height = height_times_4 // 4
        frame = frame.reshape(width, height, 4)[:, :, :3]

        return frame

    def update_viewer(self, viewer):
        """
        Update viewer display.

        REQUIRES: step_graphics() must have been called first.

        Args:
            viewer: Isaac Gym viewer handle
        """
        if not self._graphics_stepped:
            logger.error(
                "update_viewer() called before step_graphics() - this will cause issues"
            )
            return False

        try:
            self.gym.draw_viewer(viewer, self.sim, True)
            return True
        except Exception as e:
            logger.error(f"Failed to update viewer: {e}")
            return False

    def sync_frame_time(self):
        """Synchronize to real-time using Isaac Gym's sync mechanism."""
        try:
            self.gym.sync_frame_time(self.sim)
        except Exception as e:
            logger.error(f"Failed to sync frame time: {e}")

    def reset_graphics_state(self):
        """Reset graphics state for new frame (call at start of render cycle)."""
        self._graphics_stepped = False

    def get_camera_info(self, camera_name: str) -> dict:
        """
        Get information about a camera.

        Args:
            camera_name: Name of the camera

        Returns:
            Dictionary with camera information or None if not found
        """
        if camera_name not in self._cameras:
            return None

        return {
            "name": camera_name,
            "handle": self._cameras[camera_name],
            "env_handle": self._camera_envs[camera_name],
        }

    def list_cameras(self) -> list:
        """Get list of all registered camera names."""
        return list(self._cameras.keys())

    def cleanup(self):
        """Clean up graphics resources."""
        # Isaac Gym handles camera cleanup automatically
        self._cameras.clear()
        self._camera_envs.clear()
        logger.debug("GraphicsManager cleaned up")
