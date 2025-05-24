"""
Camera controller component for DexHand environment.

This module provides camera control functionality for the DexHand environment,
including keyboard shortcuts for changing camera views and following specific robots.
"""

# Import IsaacGym first
from isaacgym import gymapi

# Then import PyTorch
import torch


class CameraController:
    """
    Handles camera control and keyboard shortcuts for the DexHand environment.
    
    This component provides functionality to:
    - Toggle between different camera views (free, side view, top-down)
    - Navigate between different robots in the environment
    - Reset selected environments
    - Update camera position based on the current mode
    """
    
    def __init__(self, gym, viewer, envs, num_envs, device):
        """
        Initialize the camera controller.
        
        Args:
            gym: The isaacgym gym instance
            viewer: The isaacgym viewer
            envs: List of environment instances
            num_envs: Number of environments
            device: PyTorch device
        """
        self.gym = gym
        self.viewer = viewer
        self.envs = envs
        self.num_envs = num_envs
        self.device = device
        
        # Camera control state
        self.lock_viewer_to_robot = 0  # 0 = free camera, 1 = side view, 2 = top-down view
        self.follow_robot_index = 0  # Which robot to follow
        
        # Track whether keyboard events have been subscribed
        self._keyboard_subscribed = False
        
        # Initialize keyboard events if viewer is available
        if self.viewer is not None:
            self.subscribe_keyboard_events()
    
    def subscribe_keyboard_events(self):
        """
        Subscribe to keyboard events for interactive control in the viewer.
        
        Sets up keyboard shortcuts for:
        - Enter: Toggle camera following mode (free, side view, top-down view)
        - Up/Down arrows: Navigate to previous/next robot to follow
        - P: Reset the currently selected environment
        """
        if self.viewer is None:
            return
            
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_ENTER, "toggle camera mode"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_UP, "previous robot"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_DOWN, "next robot"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_P, "reset environment"
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
                
                if evt.action == "toggle camera mode" and evt.value > 0:
                    # Cycle through camera modes: free (0) -> side view (1) -> top-down (2) -> free (0)
                    self.lock_viewer_to_robot = (self.lock_viewer_to_robot + 1) % 3
                    print(f"Camera mode changed to: {['Free', 'Side View', 'Top-Down'][self.lock_viewer_to_robot]}")
                elif evt.action == "previous robot" and evt.value > 0:
                    # Move to previous robot
                    self.follow_robot_index = (self.follow_robot_index - 1) % self.num_envs
                    print(f"Following robot {self.follow_robot_index}")
                elif evt.action == "next robot" and evt.value > 0:
                    # Move to next robot
                    self.follow_robot_index = (self.follow_robot_index + 1) % self.num_envs
                    print(f"Following robot {self.follow_robot_index}")
                elif evt.action == "reset environment" and evt.value > 0 and reset_callback is not None:
                    # Reset only the selected environment
                    print(f"Resetting robot {self.follow_robot_index}")
                    reset_callback(torch.tensor([self.follow_robot_index], device=self.device))
            
            return events_processed
        except Exception as e:
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
        if self.viewer is None or self.lock_viewer_to_robot == 0:
            return False
            
        # Safety check for valid input and index
        if hand_positions is None or hand_positions.shape[0] == 0:
            return False
            
        # Make sure follow_robot_index is within bounds
        safe_index = min(self.follow_robot_index, hand_positions.shape[0] - 1)
        
        # Define different camera positions based on the viewing mode
        if self.lock_viewer_to_robot == 1:
            # Side view
            camera_offset = gymapi.Vec3(-1.0, 0.0, 0.6)
        else:  # mode 2
            # Top-down view
            camera_offset = gymapi.Vec3(0.0, -1.0, 1.0)
            
        try:
            # Get position of the hand we're following
            hand_pos = hand_positions[safe_index].cpu().numpy()
            
            # Set camera position and target
            cam_pos = gymapi.Vec3(
                hand_pos[0] + camera_offset.x,
                hand_pos[1] + camera_offset.y,
                hand_pos[2] + camera_offset.z
            )
            cam_target = gymapi.Vec3(hand_pos[0], hand_pos[1], hand_pos[2])
            
            # Update camera
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
            return True
        except Exception as e:
            # Handle exceptions silently - this can happen during initialization
            return False