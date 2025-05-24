"""
Reset manager component for DexHand environment.

This module provides reset functionality for the DexHand environment,
including environment resets, randomization, and initialization.
"""

# Import standard libraries
import torch
import numpy as np

# Import IsaacGym
from isaacgym import gymapi, gymtorch


class ResetManager:
    """
    Manages environment resets for the DexHand environment.
    
    This component provides functionality to:
    - Handle environment resets based on episode termination
    - Manage task-specific reset conditions
    - Track episode progress
    - Apply randomization during resets
    """
    
    def __init__(self, gym, sim, num_envs, device, max_episode_length=1000):
        """
        Initialize the reset manager.
        
        Args:
            gym: The isaacgym gym instance
            sim: The isaacgym simulation instance
            num_envs: Number of environments
            device: PyTorch device
            max_episode_length: Maximum episode length
        """
        self.gym = gym
        self.sim = sim
        self.num_envs = num_envs
        self.device = device
        self.max_episode_length = max_episode_length
        
        # Reset and progress buffers
        self.reset_buf = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.progress_buf = torch.zeros(num_envs, device=device, dtype=torch.long)
        
        # Random state for reproducibility
        self.rng_state = None
        
        # Initialize randomization settings
        self.randomize_initial_positions = False
        self.randomize_initial_orientations = False
        self.randomize_dof_positions = False
        self.position_randomization_range = [0.0, 0.0, 0.0]
        self.orientation_randomization_range = 0.0
        self.dof_position_randomization_range = 0.0
        
        # Default initial values
        self.default_dof_pos = None
        self.default_hand_pos = torch.tensor([0.0, 0.0, 0.5], device=device)
        self.default_hand_rot = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
    
    def set_episode_length(self, max_episode_length):
        """
        Set the maximum episode length.
        
        Args:
            max_episode_length: Maximum episode length
        """
        self.max_episode_length = max_episode_length
    
    def set_randomization(self, randomize_positions=False, randomize_orientations=False, 
                        randomize_dofs=False, position_range=None, orientation_range=None, 
                        dof_range=None):
        """
        Configure randomization settings for resets.
        
        Args:
            randomize_positions: Whether to randomize hand positions
            randomize_orientations: Whether to randomize hand orientations
            randomize_dofs: Whether to randomize DOF positions
            position_range: Position randomization range [x, y, z]
            orientation_range: Orientation randomization range in radians
            dof_range: DOF position randomization range in radians
        """
        self.randomize_initial_positions = randomize_positions
        self.randomize_initial_orientations = randomize_orientations
        self.randomize_dof_positions = randomize_dofs
        
        if position_range is not None:
            self.position_randomization_range = position_range
        if orientation_range is not None:
            self.orientation_randomization_range = orientation_range
        if dof_range is not None:
            self.dof_position_randomization_range = dof_range
    
    def set_default_state(self, dof_pos=None, hand_pos=None, hand_rot=None):
        """
        Set default state for resets.
        
        Args:
            dof_pos: Default DOF positions
            hand_pos: Default hand position
            hand_rot: Default hand rotation (quaternion)
        """
        if dof_pos is not None:
            self.default_dof_pos = dof_pos
        if hand_pos is not None:
            self.default_hand_pos = torch.tensor(hand_pos, device=self.device)
        if hand_rot is not None:
            self.default_hand_rot = torch.tensor(hand_rot, device=self.device)
    
    def seed(self, seed=None):
        """
        Set random seed for reproducibility.
        
        Args:
            seed: Random seed
        """
        if seed is not None:
            # Save current PyTorch RNG state
            self.rng_state = torch.get_rng_state()
            # Set new seed
            torch.manual_seed(seed)
    
    def restore_rng_state(self):
        """
        Restore previous RNG state.
        """
        if self.rng_state is not None:
            torch.set_rng_state(self.rng_state)
            self.rng_state = None
    
    def check_termination(self, task_reset=None):
        """
        Check for episode termination.
        
        Args:
            task_reset: Optional task-specific reset conditions
            
        Returns:
            Updated reset buffer
        """
        # Reset environments that have reached max episode length
        self.reset_buf = torch.where(
            self.progress_buf >= self.max_episode_length - 1,
            torch.ones_like(self.reset_buf),
            self.reset_buf
        )
        
        # Apply task-specific reset conditions if provided
        if task_reset is not None:
            self.reset_buf = torch.logical_or(self.reset_buf, task_reset)
        
        # Convert to boolean
        self.reset_buf = self.reset_buf.bool()
        
        return self.reset_buf
    
    def increment_progress(self):
        """
        Increment progress buffers.
        
        Returns:
            Updated progress buffer
        """
        self.progress_buf += 1
        return self.progress_buf
    
    def reset_idx(self, env_ids, physics_manager=None, dof_state=None, root_state_tensor=None,
                hand_indices=None, task_reset_func=None):
        """
        Reset specified environments.
        
        Args:
            env_ids: Tensor of environment IDs to reset
            physics_manager: Physics manager component
            dof_state: DOF state tensor
            root_state_tensor: Root state tensor
            hand_indices: Indices of hand actors
            task_reset_func: Optional task-specific reset function
            
        Returns:
            Boolean indicating success
        """
        if len(env_ids) == 0:
            return True
            
        # Validate inputs
        if dof_state is None or root_state_tensor is None:
            print("Error: dof_state and root_state_tensor must be provided")
            return False
            
        # Reset progress buffer for reset environments
        self.progress_buf[env_ids] = 0
        
        # Reset DOF states
        if self.default_dof_pos is not None:
            # Use default DOF positions
            dof_pos = self.default_dof_pos.clone()
            
            # Apply randomization if enabled
            if self.randomize_dof_positions and self.dof_position_randomization_range > 0:
                dof_pos = dof_pos + torch.rand(
                    dof_pos.shape, device=self.device
                ) * self.dof_position_randomization_range - self.dof_position_randomization_range/2
            
            # Set DOF positions for reset environments
            dof_state[env_ids, :, 0] = dof_pos
            
            # Zero DOF velocities
            dof_state[env_ids, :, 1] = 0
        
        # Reset hand pose in root state tensor
        if hand_indices is not None and len(hand_indices) > 0:
            for i, env_id in enumerate(env_ids):
                if i >= len(hand_indices):
                    break
                    
                # Get actor index in root state tensor
                hand_root_idx = hand_indices[env_id] * 13
                
                # Set position
                pos = self.default_hand_pos.clone()
                
                # Apply position randomization if enabled
                if self.randomize_initial_positions:
                    pos = pos + torch.tensor([
                        (torch.rand(1, device=self.device).item() * 2 - 1) * self.position_randomization_range[0],
                        (torch.rand(1, device=self.device).item() * 2 - 1) * self.position_randomization_range[1],
                        (torch.rand(1, device=self.device).item() * 2 - 1) * self.position_randomization_range[2]
                    ], device=self.device)
                
                root_state_tensor[env_id, hand_root_idx:hand_root_idx + 3] = pos
                
                # Set rotation (quaternion)
                rot = self.default_hand_rot.clone()
                
                # Apply orientation randomization if enabled
                if self.randomize_initial_orientations and self.orientation_randomization_range > 0:
                    # Generate random rotation around z-axis
                    rand_angle = (torch.rand(1, device=self.device).item() * 2 - 1) * self.orientation_randomization_range
                    cos_angle = torch.cos(torch.tensor(rand_angle/2, device=self.device))
                    sin_angle = torch.sin(torch.tensor(rand_angle/2, device=self.device))
                    
                    # Create quaternion for z-axis rotation [x, y, z, w]
                    rand_quat = torch.tensor([0, 0, sin_angle, cos_angle], device=self.device)
                    
                    # Apply random rotation using quaternion multiplication
                    # For simplicity, we're only applying a rotation around z
                    # A full implementation would use quaternion multiplication
                    rot = rand_quat
                
                root_state_tensor[env_id, hand_root_idx + 3:hand_root_idx + 7] = rot
                
                # Zero velocities
                root_state_tensor[env_id, hand_root_idx + 7:hand_root_idx + 13] = 0
        
        # Call task-specific reset function if provided
        if task_reset_func is not None:
            task_reset_func(env_ids)
        
        # Apply tensor states to simulation
        if physics_manager is not None:
            physics_manager.apply_tensor_states(
                self.gym, self.sim, env_ids, dof_state, root_state_tensor
            )
            
        return True
    
    def reset_all(self, physics_manager=None, dof_state=None, root_state_tensor=None,
                hand_indices=None, task_reset_func=None):
        """
        Reset all environments.
        
        Args:
            physics_manager: Physics manager component
            dof_state: DOF state tensor
            root_state_tensor: Root state tensor
            hand_indices: Indices of hand actors
            task_reset_func: Optional task-specific reset function
            
        Returns:
            Boolean indicating success
        """
        env_ids = torch.arange(self.num_envs, device=self.device)
        return self.reset_idx(
            env_ids, physics_manager, dof_state, root_state_tensor,
            hand_indices, task_reset_func
        )