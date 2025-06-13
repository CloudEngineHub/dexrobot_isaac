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
        self.camera_view_mode = "rear"  # Options: "free", "rear", "right", "bottom"
        self.camera_follow_mode = "single"  # Options: "single" (follow one robot), "global" (overview)
        self.follow_robot_index = 0  # Which robot to follow in single mode
        
        # Track whether keyboard events have been subscribed
        self._keyboard_subscribed = False
        
        # Initialize keyboard events if viewer is available
        if self.viewer is not None:
            self.subscribe_keyboard_events()
    
    def subscribe_keyboard_events(self):
        """
        Subscribe to keyboard events for interactive control in the viewer.
        
        Sets up keyboard shortcuts for:
        - Enter: Toggle camera view mode (free, rear, right, bottom)
        - G: Toggle between single robot follow and global view
        - Up/Down arrows: Navigate to previous/next robot to follow (in single mode)
        - P: Reset the currently selected environment
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
                
                if evt.action == "toggle view mode" and evt.value > 0:
                    # Cycle through view modes: free -> rear -> right -> bottom -> free
                    view_modes = ["free", "rear", "right", "bottom"]
                    current_idx = view_modes.index(self.camera_view_mode)
                    self.camera_view_mode = view_modes[(current_idx + 1) % len(view_modes)]
                    
                    # Display current camera state
                    view_names = {"free": "Free Camera", "rear": "Rear View", "right": "Right View", "bottom": "Bottom View"}
                    follow_text = f"following robot {self.follow_robot_index}" if self.camera_follow_mode == "single" else "global view"
                    print(f"Camera: {view_names[self.camera_view_mode]} ({follow_text})")
                elif evt.action == "toggle follow mode" and evt.value > 0:
                    # Toggle between single robot follow and global view
                    self.camera_follow_mode = "global" if self.camera_follow_mode == "single" else "single"
                    follow_text = f"following robot {self.follow_robot_index}" if self.camera_follow_mode == "single" else "global view"
                    view_names = {"free": "Free Camera", "rear": "Rear View", "right": "Right View", "bottom": "Bottom View"}
                    print(f"Camera: {view_names[self.camera_view_mode]} ({follow_text})")
                elif evt.action == "previous robot" and evt.value > 0:
                    # Move to previous robot (only in single mode)
                    if self.camera_follow_mode == "single":
                        self.follow_robot_index = (self.follow_robot_index - 1) % self.num_envs
                        print(f"Following robot {self.follow_robot_index}")
                    else:
                        print("Cannot change robot in global view mode. Press Tab to switch to single robot mode.")
                elif evt.action == "next robot" and evt.value > 0:
                    # Move to next robot (only in single mode)
                    if self.camera_follow_mode == "single":
                        self.follow_robot_index = (self.follow_robot_index + 1) % self.num_envs
                        print(f"Following robot {self.follow_robot_index}")
                    else:
                        print("Cannot change robot in global view mode. Press Tab to switch to single robot mode.")
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
            "rear": gymapi.Vec3(-1.0, 0.0, 0.6),     # Behind the hand
            "right": gymapi.Vec3(0.0, -1.0, 0.6),    # From the right side
            "bottom": gymapi.Vec3(0.0, 0.3, -1.0),   # Looking up from below
        }
        camera_offset = camera_offsets.get(self.camera_view_mode, gymapi.Vec3(-1.0, 0.0, 0.6))
        
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
                camera_offset.z + 1.0 if self.camera_view_mode != "bottom" else camera_offset.z - 1.0
            )
            
        try:
            # Set camera position and target
            cam_pos = gymapi.Vec3(
                target_pos[0] + camera_offset.x,
                target_pos[1] + camera_offset.y,
                target_pos[2] + camera_offset.z
            )
            cam_target = gymapi.Vec3(target_pos[0], target_pos[1], target_pos[2])
            
            # Update camera
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
            return True
        except Exception as e:
            # Handle exceptions silently - this can happen during initialization
            return False