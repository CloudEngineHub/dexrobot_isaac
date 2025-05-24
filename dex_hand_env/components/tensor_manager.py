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
        
        Raises:
            RuntimeError: If any required tensor handle acquisition fails
        """
        # Reset flags
        self.tensors_initialized = False
        
        print("Acquiring tensor handles from simulation...")
        
        # Step the simulation to ensure it's ready for tensor acquisition
        print("Stepping simulation to prepare for tensor acquisition...")
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        
        # Get handle for DOF states
        print("Acquiring DOF state tensor handle...")
        self._dof_state_tensor_handle = self.gym.acquire_dof_state_tensor(self.sim)
        if self._dof_state_tensor_handle is None:
            raise RuntimeError("Failed to acquire DOF state tensor handle. Cannot continue.")
        print("Successfully acquired DOF state tensor handle")
        
        # Get handle for rigid body states
        print("Acquiring rigid body state tensor handle...")
        self._rigid_body_state_tensor_handle = self.gym.acquire_rigid_body_state_tensor(self.sim)
        if self._rigid_body_state_tensor_handle is None:
            raise RuntimeError("Failed to acquire rigid body state tensor handle. Cannot continue.")
        print("Successfully acquired rigid body state tensor handle")
        
        # Get handle for contact forces
        print("Acquiring contact force tensor handle...")
        self._contact_force_tensor_handle = self.gym.acquire_net_contact_force_tensor(self.sim)
        if self._contact_force_tensor_handle is None:
            raise RuntimeError("Failed to acquire contact force tensor handle. Cannot continue.")
        print("Successfully acquired contact force tensor handle")
        
        # Step the simulation again to ensure tensor handles are valid
        print("Stepping simulation to ensure tensor handles are valid...")
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        
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
            
        Raises:
            RuntimeError: If any tensor setup fails
        """
        print("Setting up tensors from handles...")
        
        # Verify tensor handles exist
        if self._dof_state_tensor_handle is None:
            raise RuntimeError("DOF state tensor handle is None. Cannot set up tensors.")
            
        if self._rigid_body_state_tensor_handle is None:
            raise RuntimeError("Rigid body state tensor handle is None. Cannot set up tensors.")
            
        if self._contact_force_tensor_handle is None:
            raise RuntimeError("Contact force tensor handle is None. Cannot set up tensors.")
        
        # Wrap DOF state tensor
        print("Wrapping DOF state tensor...")
        try:
            self.dof_state = gymtorch.wrap_tensor(self._dof_state_tensor_handle)
            print(f"DOF state tensor handle type: {type(self._dof_state_tensor_handle)}")
            
            if self.dof_state is None:
                raise RuntimeError("DOF state tensor is None after wrapping")
            
            # This line is causing the error if the tensor exists but is empty
            # Let's check what's actually in the tensor before assuming it's empty
            print(f"DOF state tensor type: {type(self.dof_state)}")
            print(f"DOF state tensor device: {self.dof_state.device}")
            print(f"DOF state tensor shape exists: {'shape' in dir(self.dof_state)}")
            
            # Check if the tensor has elements before accessing shape
            if hasattr(self.dof_state, 'numel') and self.dof_state.numel() == 0:
                raise RuntimeError("DOF state tensor is empty (zero elements) after wrapping")
                
        except Exception as e:
            print(f"Error while wrapping DOF state tensor: {e}")
            print(f"DOF state tensor handle: {self._dof_state_tensor_handle}")
            raise RuntimeError(f"Failed to wrap DOF state tensor: {e}")
        
        print(f"DOF state tensor shape: {self.dof_state.shape}")
        
        # Get number of DOFs per environment
        # DOF state is shaped (num_envs * num_dofs_per_env, 2)
        # where 2 is (position, velocity)
        # We need to reshape to (num_envs, num_dofs_per_env, 2)
        if self.dof_state.shape[0] % self.num_envs != 0:
            raise RuntimeError(f"DOF state tensor shape {self.dof_state.shape[0]} is not divisible by num_envs {self.num_envs}")
        
        dofs_per_env = self.dof_state.shape[0] // self.num_envs
        self.num_dof = dofs_per_env
        
        print(f"Total DOFs: {self.dof_state.shape[0]}, Environments: {self.num_envs}, DOFs per env: {self.num_dof}")
        
        # Reshape DOF state to (num_envs, num_dofs_per_env, 2)
        reshaped_dof_state = self.dof_state.view(self.num_envs, self.num_dof, 2)
        
        # Extract position and velocity components
        self.dof_pos = reshaped_dof_state[..., 0]
        self.dof_vel = reshaped_dof_state[..., 1]
        
        print(f"DOF position tensor shape: {self.dof_pos.shape}")
        print(f"DOF velocity tensor shape: {self.dof_vel.shape}")
        
        # Get DOF properties
        print("Getting DOF properties...")
        
        # We cannot use get_envs since it doesn't exist in the API
        # Instead, we'll use the fact that environments are stored in an array in DexHandBase
        # and passed to this component during initialization
        
        # Use simpler approach to get actor DOF properties
        try:
            # Verify we can get DOF properties directly using acquire_dof_attribute_tensor
            dof_props_tensor_handle = self.gym.acquire_dof_attribute_tensor(self.sim, gymapi.DOMAIN_ENV, gymapi.ATTRIB_DOF_PROPERTIES)
            if dof_props_tensor_handle is None:
                raise RuntimeError("Failed to acquire DOF properties tensor handle")
                
            # Wrap the tensor handle
            self.dof_props = gymtorch.wrap_tensor(dof_props_tensor_handle)
            if self.dof_props is None or self.dof_props.numel() == 0:
                raise RuntimeError("DOF properties tensor is empty or None after wrapping")
                
            print(f"DOF properties acquired with shape: {self.dof_props.shape}")
        except Exception as e:
            print(f"Error getting DOF properties: {e}")
            # Fallback to a simpler approach
            print("Using default DOF properties")
            # Create properties tensor based on the dof_state shape
            # DOF properties have 6 values: stiffness, damping, friction, armature, min, max
            self.dof_props = torch.zeros((self.num_dof, 6), device=self.device)
            # Set reasonable defaults
            self.dof_props[:, 0] = 100.0  # stiffness
            self.dof_props[:, 1] = 5.0    # damping
            self.dof_props[:, 4] = -1.0   # min
            self.dof_props[:, 5] = 1.0    # max
            print(f"Created default DOF properties tensor with shape: {self.dof_props.shape}")
        
        # Wrap rigid body state tensor
        print("Wrapping rigid body state tensor...")
        self.rigid_body_states = gymtorch.wrap_tensor(self._rigid_body_state_tensor_handle)
        
        if self.rigid_body_states is None or self.rigid_body_states.numel() == 0:
            raise RuntimeError("Rigid body state tensor is empty or None after wrapping")
        
        print(f"Rigid body states tensor shape: {self.rigid_body_states.shape}")
        
        # Reshape rigid body states
        if self.rigid_body_states.shape[0] % self.num_envs != 0:
            raise RuntimeError(f"Rigid body tensor shape {self.rigid_body_states.shape[0]} is not divisible by num_envs {self.num_envs}")
        
        # Extract root state tensor (one root state per environment)
        num_bodies_per_env = self.rigid_body_states.shape[0] // self.num_envs
        self.root_state_tensor = self.rigid_body_states.view(self.num_envs, num_bodies_per_env, 13)
        
        print(f"Root state tensor shape: {self.root_state_tensor.shape}")
        
        # Wrap contact force tensor
        print("Wrapping contact force tensor...")
        contact_forces_flat = gymtorch.wrap_tensor(self._contact_force_tensor_handle)
        
        if contact_forces_flat is None or contact_forces_flat.numel() == 0:
            raise RuntimeError("Contact force tensor is empty or None after wrapping")
        
        print(f"Contact forces flat tensor shape: {contact_forces_flat.shape}")
        
        # Reshape contact forces for fingertips
        if fingertip_indices is None or len(fingertip_indices) == 0:
            raise RuntimeError("No fingertip indices provided. Cannot set up contact forces.")
        
        # Number of fingers
        num_fingers = len(fingertip_indices[0])
        print(f"Number of fingers: {num_fingers}")
        
        # Create tensor for contact forces
        self.contact_forces = torch.zeros((self.num_envs, num_fingers, 3), device=self.device)
        print(f"Contact forces tensor shape: {self.contact_forces.shape}")
        
        # Copy contact forces for each fingertip
        for i in range(self.num_envs):
            for j in range(num_fingers):
                if i < len(fingertip_indices) and j < len(fingertip_indices[i]):
                    # Get the rigid body index for this fingertip
                    finger_idx = fingertip_indices[i][j]
                    
                    if finger_idx < contact_forces_flat.shape[0]:
                        # Copy contact force (x, y, z)
                        self.contact_forces[i, j, :] = contact_forces_flat[finger_idx]
        
        # Mark tensors as initialized
        self.tensors_initialized = True
        print("Tensors initialized successfully")
        
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
            
        Raises:
            RuntimeError: If tensors are not initialized or refresh fails
        """
        try:
            # Verify tensors are initialized
            if not self.tensors_initialized:
                raise RuntimeError("Tensors not initialized. Cannot refresh.")
            
            # Refresh DOF state - verify shape first
            if self.dof_state is None:
                raise RuntimeError("DOF state tensor is None. Cannot refresh.")
                
            expected_size = self.num_envs * self.num_dof * 2
            if self.dof_state.numel() != expected_size:
                raise RuntimeError(f"DOF state tensor size mismatch. Expected {expected_size} elements, got {self.dof_state.numel()}")
                
            # Reshape to [num_envs, num_dof, 2] and extract position and velocity
            self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
            self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
            
            # Refresh contact forces
            if fingertip_indices is None or len(fingertip_indices) == 0:
                raise RuntimeError("No fingertip indices provided. Cannot refresh contact forces.")
            
            contact_forces_flat = gymtorch.wrap_tensor(self._contact_force_tensor_handle)
            if contact_forces_flat is None or contact_forces_flat.numel() == 0:
                raise RuntimeError("Contact force tensor is empty or None during refresh")
            
            # Number of fingers
            num_fingers = len(fingertip_indices[0])
            
            # Copy contact forces for each fingertip
            for i in range(self.num_envs):
                if i >= len(fingertip_indices):
                    raise RuntimeError(f"Fingertip indices missing for environment {i}")
                    
                for j in range(num_fingers):
                    if j >= len(fingertip_indices[i]):
                        raise RuntimeError(f"Fingertip index {j} missing for environment {i}")
                        
                    # Get the rigid body index for this fingertip
                    finger_idx = fingertip_indices[i][j]
                    
                    if finger_idx >= contact_forces_flat.shape[0]:
                        raise RuntimeError(f"Fingertip index {finger_idx} exceeds contact forces size {contact_forces_flat.shape[0]}")
                    
                    # Copy contact force (x, y, z)
                    self.contact_forces[i, j, :] = contact_forces_flat[finger_idx]
            
            return {
                "dof_pos": self.dof_pos,
                "dof_vel": self.dof_vel,
                "root_state_tensor": self.root_state_tensor,
                "contact_forces": self.contact_forces
            }
        except Exception as e:
            print(f"CRITICAL ERROR in refresh_tensors: {e}")
            import traceback
            traceback.print_exc()
            raise
    
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