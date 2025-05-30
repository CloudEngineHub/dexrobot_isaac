"""
Observation encoder component for DexHand environment.

This module provides observation encoding functionality for the DexHand environment,
including proprioceptive states, sensor readings, and task-specific observations.
"""

# Import standard libraries
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

# Import gym for spaces
import gym

# Import IsaacGym
from isaacgym import gymapi, gymtorch


class ObservationEncoder:
    """
    Encodes observations for the DexHand environment.
    
    This component provides functionality to:
    - Generate proprioceptive observations (joint positions, velocities)
    - Process sensor readings (contact forces, tactile)
    - Create task-specific observations
    - Normalize and format observations for the policy
    
    The new design follows a cleaner architecture:
    1. Compute a dictionary of "default" observations
    2. Call a function to compute task-specific observations
    3. Merge the two dictionaries
    4. Concat tensors with selected keys into final observation buffer
    """
    
    def __init__(self, gym, sim, num_envs, device, tensor_manager, hand_asset=None):
        """
        Initialize the observation encoder.
        
        Args:
            gym: The isaacgym gym instance
            sim: The isaacgym simulation instance
            num_envs: Number of environments
            device: PyTorch device
            tensor_manager: Reference to tensor manager for accessing tensors
            hand_asset: Hand asset for getting DOF names (optional)
        """
        self.gym = gym
        self.sim = sim
        self.num_envs = num_envs
        self.device = device
        self.tensor_manager = tensor_manager
        self.hand_asset = hand_asset
        
        # Store DOF names if asset is provided
        self.dof_names = []
        if hand_asset is not None:
            self.dof_names = self.gym.get_asset_dof_names(hand_asset)
            print(f"ObservationEncoder initialized with {len(self.dof_names)} DOF names from asset")
        
        # Constants for observation dimensions
        self.NUM_BASE_DOFS = 6
        self.NUM_ACTIVE_FINGER_DOFS = 12  # 12 finger controls mapping to 19 DOFs with coupling
        self.NUM_FINGERS = 5
        
        # Configuration - will be set during initialize()
        self.observation_keys = []
        self.num_observations = 0
        
        # Initialize observation buffers
        self.obs_buf = None
        self.states_buf = None
        
        # Previous actions for observation (if enabled)
        self.prev_actions = None

    def initialize(self, observation_keys: List[str], hand_indices: List[int], 
                  fingertip_indices: List[int], joint_to_control: Dict[str, str], 
                  active_joint_names: List[str], num_actions: int = None, action_processor=None):
        """
        Initialize the observation encoder with configuration.
        
        Args:
            observation_keys: List of observation components to include
            hand_indices: Indices of hand actors
            fingertip_indices: Indices of fingertips  
            joint_to_control: Mapping from joint names to control names
            active_joint_names: List of active joint names
            num_actions: Actual number of actions in the action space
            action_processor: Reference to action processor for accessing DOF targets
        """
        self.observation_keys = observation_keys
        self.hand_indices = hand_indices
        self.fingertip_indices = fingertip_indices
        self.joint_to_control = joint_to_control
        self.active_joint_names = active_joint_names
        self.action_processor = action_processor
        
        # Pre-compute active finger DOF indices for efficient observation extraction
        self.active_finger_dof_indices = self._compute_active_finger_dof_indices()
        
        # Initialize previous actions tensor if needed
        if "prev_actions" in self.observation_keys:
            # Use actual action space size if provided, otherwise default to full size
            prev_action_size = num_actions if num_actions is not None else (self.NUM_BASE_DOFS + self.NUM_ACTIVE_FINGER_DOFS)
            self.prev_actions = torch.zeros(
                (self.num_envs, prev_action_size), 
                device=self.device
            )
        
        # Compute observation dimension dynamically by creating a test observation
        test_obs_dict = self._compute_default_observations()
        test_task_obs_dict = self._compute_task_observations(test_obs_dict)
        merged_obs_dict = {**test_obs_dict, **test_task_obs_dict}
        
        # Print dimensions of each observation component
        print("Observation component dimensions:")
        total_dim = 0
        for key in self.observation_keys:
            if key in merged_obs_dict:
                tensor = merged_obs_dict[key]
                if len(tensor.shape) > 2:
                    tensor = tensor.reshape(self.num_envs, -1)
                dim = tensor.shape[1]
                total_dim += dim
                print(f"  {key}: {dim}")
            else:
                print(f"  {key}: MISSING")
        
        test_obs_tensor = self._concat_selected_observations(merged_obs_dict)
        
        self.num_observations = test_obs_tensor.shape[1]
        print(f"Total observation dimension: {total_dim}")
        print(f"ObservationEncoder initialized with dynamic observation dimension: {self.num_observations}")
        
        # Initialize observation buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_observations), device=self.device)
        self.states_buf = torch.zeros((self.num_envs, self.num_observations), device=self.device)

    def update_prev_actions(self, actions: torch.Tensor):
        """
        Update previous actions for observation.
        
        Args:
            actions: Current action tensor
        """
        if actions is not None and hasattr(actions, 'shape') and self.prev_actions is not None:
            self.prev_actions = actions.clone()

    def compute_observations(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the observation vector and dictionary.
        
        Returns:
            Tuple of (observation tensor, observation dictionary)
        """
        # Step 1: Compute default observations
        default_obs_dict = self._compute_default_observations()
        
        # Step 2: Compute task-specific observations
        task_obs_dict = self._compute_task_observations(default_obs_dict)
        
        # Step 3: Merge dictionaries
        merged_obs_dict = {**default_obs_dict, **task_obs_dict}
        
        # Step 4: Concat selected observations into final observation buffer
        self.obs_buf = self._concat_selected_observations(merged_obs_dict)
        
        # Copy to states buffer (for asymmetric actor-critic)
        self.states_buf = self.obs_buf.clone()
        
        return self.obs_buf, merged_obs_dict

    def _compute_active_finger_dof_indices(self) -> torch.Tensor:
        """
        Pre-compute indices mapping from full DOF tensor to active finger DOFs.
        
        Returns:
            torch.Tensor of shape (num_active_finger_dofs,) with DOF indices
        """
        if not self.dof_names:
            raise RuntimeError("DOF names not available from asset. Cannot compute active finger DOF indices.")
        
        if not self.joint_to_control:
            raise RuntimeError("joint_to_control mapping not provided. Cannot compute active finger DOF indices.")
        
        if not self.active_joint_names:
            raise RuntimeError("active_joint_names not provided. Cannot compute active finger DOF indices.")
        
        # Create mapping from control name to its index in the active joint list
        control_to_idx = {name: idx for idx, name in enumerate(self.active_joint_names)}
        
        # Create array to store DOF indices for each active control
        active_indices = torch.full((len(self.active_joint_names),), -1, dtype=torch.long, device=self.device)
        
        # Map each finger DOF to its corresponding active control index
        for i, dof_name in enumerate(self.dof_names[self.NUM_BASE_DOFS:]):
            if dof_name in self.joint_to_control:
                control_name = self.joint_to_control[dof_name]
                if control_name in control_to_idx:
                    control_idx = control_to_idx[control_name]
                    active_indices[control_idx] = i + self.NUM_BASE_DOFS
        
        # Check that all active controls have been mapped
        valid_mask = active_indices >= 0
        if not valid_mask.all():
            missing_indices = torch.nonzero(~valid_mask).flatten()
            missing_controls = [self.active_joint_names[idx] for idx in missing_indices]
            raise RuntimeError(f"Failed to map active finger controls to DOF indices: {missing_controls}")
        
        return active_indices

    def _compute_default_observations(self) -> Dict[str, torch.Tensor]:
        """
        Compute default observations dictionary.
        
        Returns:
            Dictionary of default observations
        """
        obs_dict = {}
        
        # Access tensors directly from tensor manager (no caching needed)
        dof_pos = self.tensor_manager.dof_pos
        dof_vel = self.tensor_manager.dof_vel
        root_state_tensor = self.tensor_manager.root_state_tensor
        contact_forces = self.tensor_manager.contact_forces
        
        # Safety check
        if dof_pos is None or dof_vel is None or root_state_tensor is None:
            print("Warning: Tensor handles not initialized. Cannot compute observations.")
            return obs_dict
        
        # Base DOF positions (6 DOFs: x, y, z, rx, ry, rz)
        if "base_dof_pos" in self.observation_keys:
            obs_dict["base_dof_pos"] = dof_pos[:, :self.NUM_BASE_DOFS]
        
        # Base DOF velocities (6 DOFs: x, y, z, rx, ry, rz)
        if "base_dof_vel" in self.observation_keys:
            obs_dict["base_dof_vel"] = dof_vel[:, :self.NUM_BASE_DOFS]
        
        # Active finger DOF positions (12 active finger controls)
        if "finger_dof_pos" in self.observation_keys:
            obs_dict["finger_dof_pos"] = dof_pos[:, self.active_finger_dof_indices]
        
        # Active finger DOF velocities (12 active finger controls)
        if "finger_dof_vel" in self.observation_keys:
            obs_dict["finger_dof_vel"] = dof_vel[:, self.active_finger_dof_indices]
        
        # Hand pose (position and orientation)
        if "hand_pose" in self.observation_keys and self.hand_indices is not None:
            if len(self.hand_indices) > 0:
                hand_poses = torch.zeros((self.num_envs, 7), device=self.device)
                
                # Extract root state for hand actors
                for i, hand_idx in enumerate(self.hand_indices):
                    if i >= self.num_envs:
                        break
                        
                    if i >= root_state_tensor.shape[0] or hand_idx >= root_state_tensor.shape[1]:
                        continue
                        
                    # Position (3) and orientation (4)
                    hand_poses[i, :3] = root_state_tensor[i, hand_idx, :3]  # Position
                    hand_poses[i, 3:7] = root_state_tensor[i, hand_idx, 3:7]  # Orientation
                
                obs_dict["hand_pose"] = hand_poses
            else:
                obs_dict["hand_pose"] = torch.zeros((self.num_envs, 7), device=self.device)
        
        # Contact forces (3D force for each finger)
        if "contact_forces" in self.observation_keys and contact_forces is not None:
            # Reshape contact forces to flat tensor
            flat_contacts = contact_forces.reshape(self.num_envs, -1)
            obs_dict["contact_forces"] = flat_contacts
        
        # Previous actions
        if "prev_actions" in self.observation_keys and self.prev_actions is not None:
            obs_dict["prev_actions"] = self.prev_actions
        
        # Base DOF targets (6 DOFs: x, y, z, rx, ry, rz)
        if "base_dof_target" in self.observation_keys and self.action_processor is not None:
            if hasattr(self.action_processor, 'current_targets') and self.action_processor.current_targets is not None:
                obs_dict["base_dof_target"] = self.action_processor.current_targets[:, :self.NUM_BASE_DOFS]
            else:
                obs_dict["base_dof_target"] = torch.zeros((self.num_envs, self.NUM_BASE_DOFS), device=self.device)
        
        # Active finger DOF targets (12 active finger controls)
        if "finger_dof_target" in self.observation_keys and self.action_processor is not None:
            if hasattr(self.action_processor, 'current_targets') and self.action_processor.current_targets is not None:
                obs_dict["finger_dof_target"] = self.action_processor.current_targets[:, self.active_finger_dof_indices]
            else:
                obs_dict["finger_dof_target"] = torch.zeros((self.num_envs, self.NUM_ACTIVE_FINGER_DOFS), device=self.device)
        
        # Contact force magnitude (magnitude for each finger)
        if "contact_force_magnitude" in self.observation_keys and contact_forces is not None:
            # Compute magnitude of contact forces for each finger (L2 norm of 3D force)
            contact_magnitudes = torch.norm(contact_forces, dim=2)  # Shape: (num_envs, num_fingers)
            obs_dict["contact_force_magnitude"] = contact_magnitudes
        
        return obs_dict

    def _compute_task_observations(self, default_obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute task-specific observations.
        
        This method can be overridden by specific tasks to add custom observations.
        The base implementation returns an empty dictionary.
        
        Args:
            default_obs_dict: Dictionary of default observations
            
        Returns:
            Dictionary of task-specific observations
        """
        # Base implementation returns empty dict
        # Specific tasks can override this method to add custom observations
        return {}

    def _concat_selected_observations(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Concatenate selected observations into final observation tensor.
        
        Args:
            obs_dict: Dictionary of all available observations
            
        Returns:
            Concatenated observation tensor
        """
        obs_tensors = []
        
        # Concat observations in the order specified by observation_keys
        for key in self.observation_keys:
            if key in obs_dict:
                tensor = obs_dict[key]
                # Ensure tensor is 2D (num_envs, obs_dim)
                if len(tensor.shape) > 2:
                    tensor = tensor.reshape(self.num_envs, -1)
                obs_tensors.append(tensor)
            else:
                print(f"Warning: Observation key '{key}' not found in observation dictionary")
        
        if obs_tensors:
            return torch.cat(obs_tensors, dim=1)
        else:
            return torch.zeros((self.num_envs, 0), device=self.device)

    def get_observation_space(self):
        """
        Get the observation space for the environment.
        
        Returns:
            gym.spaces.Box observation space
        """
        return gym.spaces.Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(self.num_observations,)
        )