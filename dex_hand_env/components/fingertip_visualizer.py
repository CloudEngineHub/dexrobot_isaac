"""
Fingertip visualizer component for DexHand environment.

This module provides functionality to visualize fingertip contacts by changing
their color based on contact forces.
"""

# Import IsaacGym first
from isaacgym import gymapi

# Then import PyTorch
import torch


class FingertipVisualizer:
    """
    Visualizes fingertip contacts by changing their color based on contact forces.
    
    This component provides functionality to:
    - Update fingertip colors based on contact forces
    - Provide visual feedback during rendering
    """
    
    def __init__(self, gym, envs, hand_indices, fingerpad_handles, device):
        """
        Initialize the fingertip visualizer.
        
        Args:
            gym: The isaacgym gym instance
            envs: List of environment instances
            hand_indices: Indices of hand actors in each environment
            fingerpad_handles: Handles of fingerpad rigid bodies
            device: PyTorch device
        """
        self.gym = gym
        self.envs = envs
        self.hand_indices = hand_indices
        self.fingerpad_handles = fingerpad_handles
        self.device = device
        
        # Default colors
        self.default_color = gymapi.Vec3(0.7, 0.7, 0.7)  # Light gray
        
        # Check if handles are valid
        self.handles_valid = len(self.fingerpad_handles) > 0 and len(self.envs) > 0
        
    def update_colors(self, contact_forces):
        """
        Update fingertip colors based on contact forces.
        
        Changes the color of fingertips with non-zero contact forces to provide visual feedback
        during rendering. Fingertips with contacts are colored red (intensity based on force),
        while those without contacts remain their default color.
        
        Args:
            contact_forces: Tensor of contact forces for each fingerpad [num_envs, num_fingerpads, 3]
            
        Returns:
            Boolean indicating whether colors were updated
        """
        # Skip if no rendering or invalid handles
        if not hasattr(self, 'handles_valid') or not self.handles_valid:
            return False
            
        # Calculate contact force magnitude for each fingertip
        contact_force_norm = torch.norm(contact_forces, dim=2)
        has_contact = contact_force_norm > 0.1
        
        # Update color for each environment
        for i in range(contact_forces.shape[0]):  # For each environment
            # Update each fingertip
            for ft_idx in range(min(contact_forces.shape[1], len(self.fingerpad_handles))):  # For each fingertip
                # Get the rigid body handle for this fingertip
                handle = self.fingerpad_handles[ft_idx]
                
                # Check if this fingertip has contact
                if has_contact[i, ft_idx]:
                    # Normalize force to get color intensity (clamped between 0.3 and 1.0)
                    force_mag = contact_force_norm[i, ft_idx]
                    intensity = min(1.0, 0.3 + force_mag * 0.7)
                    
                    # Red color with intensity based on contact force
                    color = gymapi.Vec3(intensity, 0.2, 0.2)
                    
                    # Set the color for this fingertip
                    try:
                        self.gym.set_rigid_body_color(
                            self.envs[i], 
                            self.hand_indices[i], 
                            handle, 
                            gymapi.MESH_VISUAL,
                            color
                        )
                    except Exception as e:
                        # Handle exception silently - this can happen during initialization
                        pass
                else:
                    # Reset to default color when no contact
                    try:
                        self.gym.set_rigid_body_color(
                            self.envs[i], 
                            self.hand_indices[i], 
                            handle, 
                            gymapi.MESH_VISUAL,
                            self.default_color
                        )
                    except Exception as e:
                        # Handle exception silently - this can happen during initialization
                        pass
        
        return True