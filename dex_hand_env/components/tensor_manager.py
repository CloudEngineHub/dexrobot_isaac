"""
Tensor manager component for DexHand environment.

This module provides tensor management functionality for the DexHand environment,
including tensor acquisition, setup, and synchronization.
"""

# Import standard libraries
import torch
import numpy as np

# Import IsaacGym
from isaacgym import gymapi, gymtorch


class TensorManager:
    """
    Manages simulation tensors for the DexHand environment.
    
    This component provides functionality to:
    - Acquire tensor handles from simulation
    - Set up and initialize tensors
    - Handle tensor synchronization between CPU and GPU
    - Manage tensor memory and buffers
    """
    
    def __init__(self, gym, sim, num_envs, device):
        """
        Initialize the tensor manager.
        
        Args:
            gym: The isaacgym gym instance
            sim: The isaacgym simulation instance
            num_envs: Number of environments
            device: PyTorch device
        """
        self.gym = gym
        self.sim = sim
        self.num_envs = num_envs
        self.device = device
        
        # Flag to track initialization state
        self.tensors_initialized = False
        
        # Tensor handles
        self._dof_state_tensor_handle = None
        self._rigid_body_state_tensor_handle = None
        self._contact_force_tensor_handle = None
        
        # Simulation tensors
        self.dof_state = None
        self.dof_pos = None
        self.dof_vel = None
        self.root_state_tensor = None
        self.num_dof = None
        self.dof_props = None
        self.rigid_body_states = None
        self.contact_forces = None
    
    def acquire_tensor_handles(self):
        """
        Acquire handles to simulation tensors.
        
        Returns:
            Dictionary of tensor handles
        """
        # Reset flags
        self.tensors_initialized = False
        
        # Get handle for DOF states
        self._dof_state_tensor_handle = self.gym.acquire_dof_state_tensor(self.sim)
        
        # Get handle for rigid body states
        self._rigid_body_state_tensor_handle = self.gym.acquire_rigid_body_state_tensor(self.sim)
        
        # Get handle for contact forces
        self._contact_force_tensor_handle = self.gym.acquire_net_contact_force_tensor(self.sim)
        
        return {
            "dof_state": self._dof_state_tensor_handle,
            "rigid_body_state": self._rigid_body_state_tensor_handle,
            "contact_force": self._contact_force_tensor_handle
        }
    
    def setup_tensors(self, fingertip_indices=None):
        """
        Set up tensors from handles.
        
        Args:
            fingertip_indices: Indices of fingertips (for contact forces)
            
        Returns:
            Dictionary of tensors
        """
        # Wrap tensor handles as PyTorch tensors
        if self._dof_state_tensor_handle is not None:
            self.dof_state = gymtorch.wrap_tensor(self._dof_state_tensor_handle)
            
            # Get number of DOFs
            self.num_dof = self.dof_state.shape[1]
            
            # Extract position and velocity components
            self.dof_pos = self.dof_state[..., 0]
            self.dof_vel = self.dof_state[..., 1]
            
            # Get DOF properties
            dof_props_tensor = self.gym.get_actor_dof_properties(self.sim)
            if dof_props_tensor is not None:
                self.dof_props = gymtorch.wrap_tensor(dof_props_tensor)
        
        if self._rigid_body_state_tensor_handle is not None:
            self.rigid_body_states = gymtorch.wrap_tensor(self._rigid_body_state_tensor_handle)
            
            # Extract root state tensor (one root state per environment)
            self.root_state_tensor = self.rigid_body_states.view(self.num_envs, -1, 13)
        
        if self._contact_force_tensor_handle is not None:
            contact_forces_flat = gymtorch.wrap_tensor(self._contact_force_tensor_handle)
            
            # Reshape contact forces for fingertips
            if fingertip_indices is not None and len(fingertip_indices) > 0:
                # Number of fingers
                num_fingers = len(fingertip_indices[0])
                
                # Reshape contact forces to (num_envs, num_fingers, 3)
                self.contact_forces = torch.zeros(
                    (self.num_envs, num_fingers, 3), device=self.device
                )
                
                # Copy contact forces for each fingertip
                for i in range(self.num_envs):
                    for j in range(num_fingers):
                        if i < len(fingertip_indices) and j < len(fingertip_indices[i]):
                            # Get the rigid body index for this fingertip
                            finger_idx = fingertip_indices[i][j]
                            
                            if finger_idx < contact_forces_flat.shape[0]:
                                # Copy contact force (x, y, z)
                                self.contact_forces[i, j, :] = contact_forces_flat[finger_idx]
            else:
                # Just reshape flat tensor to (num_envs, -1, 3)
                self.contact_forces = contact_forces_flat.view(self.num_envs, -1, 3)
        
        # Mark tensors as initialized
        self.tensors_initialized = True
        
        return {
            "dof_state": self.dof_state,
            "dof_pos": self.dof_pos,
            "dof_vel": self.dof_vel,
            "root_state_tensor": self.root_state_tensor,
            "num_dof": self.num_dof,
            "dof_props": self.dof_props,
            "rigid_body_states": self.rigid_body_states,
            "contact_forces": self.contact_forces
        }
    
    def refresh_tensors(self, fingertip_indices=None):
        """
        Refresh tensor data from simulation.
        
        Args:
            fingertip_indices: Indices of fingertips (for contact forces)
            
        Returns:
            Dictionary of refreshed tensors
        """
        # Only refresh if tensors are initialized
        if not self.tensors_initialized:
            print("Warning: Tensors not initialized. Cannot refresh.")
            return None
            
        # Refresh DOF state
        if self.dof_state is not None:
            self.dof_pos = self.dof_state[..., 0]
            self.dof_vel = self.dof_state[..., 1]
        
        # Refresh contact forces
        if self._contact_force_tensor_handle is not None and fingertip_indices is not None:
            contact_forces_flat = gymtorch.wrap_tensor(self._contact_force_tensor_handle)
            
            # Number of fingers
            num_fingers = len(fingertip_indices[0])
            
            # Copy contact forces for each fingertip
            for i in range(self.num_envs):
                for j in range(num_fingers):
                    if i < len(fingertip_indices) and j < len(fingertip_indices[i]):
                        # Get the rigid body index for this fingertip
                        finger_idx = fingertip_indices[i][j]
                        
                        if finger_idx < contact_forces_flat.shape[0]:
                            # Copy contact force (x, y, z)
                            self.contact_forces[i, j, :] = contact_forces_flat[finger_idx]
        
        return {
            "dof_pos": self.dof_pos,
            "dof_vel": self.dof_vel,
            "root_state_tensor": self.root_state_tensor,
            "contact_forces": self.contact_forces
        }
    
    def tensor_clamp(self, tensor, min_val, max_val):
        """
        Clamp tensor values between min and max.
        
        Args:
            tensor: Input tensor
            min_val: Minimum value(s)
            max_val: Maximum value(s)
            
        Returns:
            Clamped tensor
        """
        return torch.max(torch.min(tensor, max_val), min_val)
    
    def to_device(self, tensor, device=None):
        """
        Move tensor to specified device.
        
        Args:
            tensor: Input tensor
            device: Target device (if None, use self.device)
            
        Returns:
            Tensor on target device
        """
        if device is None:
            device = self.device
            
        if tensor.device != device:
            return tensor.to(device)
        return tensor