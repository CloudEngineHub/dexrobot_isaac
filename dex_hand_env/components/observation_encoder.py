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
from isaacgym.torch_utils import quat_mul, quat_conjugate

# Import utilities
from dex_hand_env.utils.coordinate_transforms import point_in_hand_frame


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
                  fingertip_indices: List[int], fingerpad_indices: List[int],
                  joint_to_control: Dict[str, str], 
                  active_joint_names: List[str], num_actions: int = None, action_processor=None,
                  index_mappings: Dict = None):
        """
        Initialize the observation encoder with configuration.
        
        Args:
            observation_keys: List of observation components to include
            hand_indices: Indices of hand actors
            fingertip_indices: Indices of fingertips
            fingerpad_indices: Indices of fingerpads  
            joint_to_control: Mapping from joint names to control names
            active_joint_names: List of active joint names
            num_actions: Actual number of actions in the action space
            action_processor: Reference to action processor for accessing DOF targets
        """
        self.observation_keys = observation_keys
        self.hand_indices = hand_indices
        self.fingertip_indices = fingertip_indices
        self.fingerpad_indices = fingerpad_indices
        self.joint_to_control = joint_to_control
        self.active_joint_names = active_joint_names
        self.action_processor = action_processor
        
        # Pre-compute active finger DOF indices for efficient observation extraction
        self.active_finger_dof_indices = self._compute_active_finger_dof_indices()
        
        # Store index mappings from DexHandBase
        if index_mappings:
            self.base_joint_to_index = index_mappings.get('base_joint_to_index', {})
            self.control_name_to_index = index_mappings.get('control_name_to_index', {})
            self.raw_dof_name_to_index = index_mappings.get('raw_dof_name_to_index', {})
            self.finger_body_to_index = index_mappings.get('finger_body_to_index', {})
        else:
            self.base_joint_to_index = {}
            self.control_name_to_index = {}
            self.raw_dof_name_to_index = {}
            self.finger_body_to_index = {}
        
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
        
        # Active finger DOF positions (12 active finger controls) - for RL observation tensor
        if "active_finger_dof_pos" in self.observation_keys:
            obs_dict["active_finger_dof_pos"] = dof_pos[:, self.active_finger_dof_indices]
        
        # Active finger DOF velocities (12 active finger controls) - for RL observation tensor
        if "active_finger_dof_vel" in self.observation_keys:
            obs_dict["active_finger_dof_vel"] = dof_vel[:, self.active_finger_dof_indices]
        
        # All finger DOF positions (20 finger DOFs: excluding base 6 DOFs, r_f_joint3_1 is fixed)
        # Always available in obs_dict for semantic access, but not included in RL tensor by default
        all_finger_dof_indices = torch.arange(self.NUM_BASE_DOFS, dof_pos.shape[1], device=self.device)
        obs_dict["all_finger_dof_pos"] = dof_pos[:, all_finger_dof_indices]
        obs_dict["all_finger_dof_vel"] = dof_vel[:, all_finger_dof_indices]
        
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
        
        # Active finger DOF targets (12 active finger controls) - for RL observation tensor
        if "active_finger_dof_target" in self.observation_keys and self.action_processor is not None:
            if hasattr(self.action_processor, 'current_targets') and self.action_processor.current_targets is not None:
                obs_dict["active_finger_dof_target"] = self.action_processor.current_targets[:, self.active_finger_dof_indices]
            else:
                obs_dict["active_finger_dof_target"] = torch.zeros((self.num_envs, self.NUM_ACTIVE_FINGER_DOFS), device=self.device)
        
        # All finger DOF targets (20 finger DOFs) - always available for semantic access
        if self.action_processor is not None:
            if hasattr(self.action_processor, 'current_targets') and self.action_processor.current_targets is not None:
                obs_dict["all_finger_dof_target"] = self.action_processor.current_targets[:, all_finger_dof_indices]
            else:
                obs_dict["all_finger_dof_target"] = torch.zeros((self.num_envs, all_finger_dof_indices.shape[0]), device=self.device)
        
        # Contact force magnitude (magnitude for each finger)
        if "contact_force_magnitude" in self.observation_keys and contact_forces is not None:
            # Compute magnitude of contact forces for each finger (L2 norm of 3D force)
            contact_magnitudes = torch.norm(contact_forces, dim=2)  # Shape: (num_envs, num_fingers)
            obs_dict["contact_force_magnitude"] = contact_magnitudes
        
        # Fingertip poses in world frame (5 fingers × 7 pose dimensions = 35)
        if "fingertip_poses_world" in self.observation_keys:
            fingertip_poses_world = self._extract_fingertip_poses_world()
            obs_dict["fingertip_poses_world"] = fingertip_poses_world
        
        # Fingertip poses in hand frame (5 fingers × 7 pose dimensions = 35)
        if "fingertip_poses_hand" in self.observation_keys:
            fingertip_poses_hand = self._extract_fingertip_poses_hand()
            obs_dict["fingertip_poses_hand"] = fingertip_poses_hand
        
        # Fingerpad poses in world frame (5 fingers × 7 pose dimensions = 35)
        if "fingerpad_poses_world" in self.observation_keys:
            fingerpad_poses_world = self._extract_fingerpad_poses_world()
            obs_dict["fingerpad_poses_world"] = fingerpad_poses_world
        
        # Fingerpad poses in hand frame (5 fingers × 7 pose dimensions = 35)
        if "fingerpad_poses_hand" in self.observation_keys:
            fingerpad_poses_hand = self._extract_fingerpad_poses_hand()
            obs_dict["fingerpad_poses_hand"] = fingerpad_poses_hand
        
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
        Records slice indices for each component during concatenation.
        
        Args:
            obs_dict: Dictionary of all available observations
            
        Returns:
            Concatenated observation tensor
        """
        obs_tensors = []
        
        # Initialize slice indices dictionary if not exists
        if not hasattr(self, 'component_slice_indices'):
            self.component_slice_indices = {}
        
        current_idx = 0
        
        # Concat observations in the order specified by observation_keys
        for key in self.observation_keys:
            if key in obs_dict:
                tensor = obs_dict[key]
                # Ensure tensor is 2D (num_envs, obs_dim)
                if len(tensor.shape) > 2:
                    tensor = tensor.reshape(self.num_envs, -1)
                
                # Record slice indices for this component
                component_size = tensor.shape[1]
                self.component_slice_indices[key] = (current_idx, current_idx + component_size)
                current_idx += component_size
                
                obs_tensors.append(tensor)
            else:
                print(f"Warning: Observation key '{key}' not found in observation dictionary")
        
        if obs_tensors:
            return torch.cat(obs_tensors, dim=1)
        else:
            return torch.zeros((self.num_envs, 0), device=self.device)

    def _extract_fingertip_poses_world(self) -> torch.Tensor:
        """
        Extract fingertip poses in world frame.
        
        Returns:
            torch.Tensor of shape (num_envs, 35) with 5 fingers × 7 pose dimensions
        """
        root_state_tensor = self.tensor_manager.root_state_tensor
        if root_state_tensor is None or self.fingertip_indices is None:
            return torch.zeros((self.num_envs, 35), device=self.device)
        
        poses = torch.zeros((self.num_envs, 35), device=self.device)
        
        for env_idx in range(self.num_envs):
            if env_idx < len(self.fingertip_indices):
                for finger_idx, tip_idx in enumerate(self.fingertip_indices[env_idx]):
                    if finger_idx < 5:  # 5 fingers
                        start_idx = finger_idx * 7
                        # Position (3) + quaternion (4) = 7 dimensions per fingertip
                        poses[env_idx, start_idx:start_idx+3] = root_state_tensor[env_idx, tip_idx, :3]  # Position
                        poses[env_idx, start_idx+3:start_idx+7] = root_state_tensor[env_idx, tip_idx, 3:7]  # Quaternion
        
        return poses
    
    def _extract_fingerpad_poses_world(self) -> torch.Tensor:
        """
        Extract fingerpad poses in world frame.
        
        Returns:
            torch.Tensor of shape (num_envs, 35) with 5 fingers × 7 pose dimensions
        """
        root_state_tensor = self.tensor_manager.root_state_tensor
        if root_state_tensor is None:
            return torch.zeros((self.num_envs, 35), device=self.device)
        
        poses = torch.zeros((self.num_envs, 35), device=self.device)
        
        # Get fingerpad body indices from hand initializer if available
        if hasattr(self, 'fingerpad_indices') and self.fingerpad_indices is not None:
            for env_idx in range(self.num_envs):
                if env_idx < len(self.fingerpad_indices):
                    for finger_idx, pad_idx in enumerate(self.fingerpad_indices[env_idx]):
                        if finger_idx < 5:  # 5 fingers
                            start_idx = finger_idx * 7
                            poses[env_idx, start_idx:start_idx+3] = root_state_tensor[env_idx, pad_idx, :3]  # Position
                            poses[env_idx, start_idx+3:start_idx+7] = root_state_tensor[env_idx, pad_idx, 3:7]  # Quaternion
        
        return poses
    
    def _extract_fingertip_poses_hand(self) -> torch.Tensor:
        """
        Extract fingertip poses in hand reference frame.
        
        Returns:
            torch.Tensor of shape (num_envs, 35) with 5 fingers × 7 pose dimensions
        """
        # Get fingertip poses in world frame
        fingertip_poses_world = self._extract_fingertip_poses_world()
        
        # Transform to hand frame
        fingertip_poses_hand = self._transform_poses_to_hand_frame(fingertip_poses_world)
        
        return fingertip_poses_hand
    
    def _extract_fingerpad_poses_hand(self) -> torch.Tensor:
        """
        Extract fingerpad poses in hand reference frame.
        
        Returns:
            torch.Tensor of shape (num_envs, 35) with 5 fingers × 7 pose dimensions
        """
        # Get fingerpad poses in world frame
        fingerpad_poses_world = self._extract_fingerpad_poses_world()
        
        # Transform to hand frame
        fingerpad_poses_hand = self._transform_poses_to_hand_frame(fingerpad_poses_world)
        
        return fingerpad_poses_hand
    
    def _transform_poses_to_hand_frame(self, poses_world: torch.Tensor) -> torch.Tensor:
        """
        Transform poses from world frame to hand reference frame.
        
        Args:
            poses_world: tensor of shape (num_envs, 35) with poses in world frame
            
        Returns:
            torch.Tensor of shape (num_envs, 35) with poses in hand frame
        """
        root_state_tensor = self.tensor_manager.root_state_tensor
        if root_state_tensor is None or self.hand_indices is None:
            return poses_world  # Return world poses if transformation not possible
        
        poses_hand = torch.zeros_like(poses_world)
        
        for env_idx in range(self.num_envs):
            if env_idx < len(self.hand_indices):
                hand_idx = self.hand_indices[env_idx]
                
                # Get hand pose in world frame
                hand_pos = root_state_tensor[env_idx, hand_idx, :3]  # Position
                hand_quat = root_state_tensor[env_idx, hand_idx, 3:7]  # Quaternion
                
                # Transform each finger pose to hand frame
                for finger_idx in range(5):
                    start_idx = finger_idx * 7
                    
                    # Extract finger pose in world frame
                    finger_pos_world = poses_world[env_idx, start_idx:start_idx+3]
                    finger_quat_world = poses_world[env_idx, start_idx+3:start_idx+7]
                    
                    # Transform position to hand frame using existing utility
                    finger_pos_hand = point_in_hand_frame(
                        finger_pos_world.unsqueeze(0), 
                        hand_pos.unsqueeze(0), 
                        hand_quat.unsqueeze(0)
                    ).squeeze(0)
                    
                    # Transform quaternion to hand frame using Isaac Gym utilities
                    hand_quat_conj = quat_conjugate(hand_quat.unsqueeze(0)).squeeze(0)
                    finger_quat_hand = quat_mul(
                        hand_quat_conj.unsqueeze(0), 
                        finger_quat_world.unsqueeze(0)
                    ).squeeze(0)
                    
                    # Store transformed pose
                    poses_hand[env_idx, start_idx:start_idx+3] = finger_pos_hand
                    poses_hand[env_idx, start_idx+3:start_idx+7] = finger_quat_hand
        
        return poses_hand

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

    def get_obs_index_for_base_joint(self, joint_name: str, obs_type: str = "pos"):
        """
        Get the index in the concatenated observation tensor for a base joint.
        
        Args:
            joint_name: Name of the base joint (e.g., "ARTx", "ARTy", etc.)
            obs_type: Type of observation ("pos", "vel", "target")
            
        Returns:
            Index in the concatenated observation tensor
        """
        if joint_name not in self.base_joint_to_index:
            raise ValueError(f"Unknown base joint name: {joint_name}. Available: {list(self.base_joint_to_index.keys())}")
        
        joint_idx = self.base_joint_to_index[joint_name]
        
        # Determine which observation component to use
        if obs_type == "pos":
            component_key = "base_dof_pos"
        elif obs_type == "vel":
            component_key = "base_dof_vel"
        elif obs_type == "target":
            component_key = "base_dof_target"
        else:
            raise ValueError(f"Unknown obs_type: {obs_type}. Available: pos, vel, target")
        
        if component_key not in self.component_slice_indices:
            raise ValueError(f"Observation component {component_key} not found in observation keys")
        
        component_start, _ = self.component_slice_indices[component_key]
        return component_start + joint_idx

    def get_obs_index_for_finger_control(self, control_name: str, obs_type: str = "pos"):
        """
        Get the index in the concatenated observation tensor for a finger control.
        
        Args:
            control_name: Name of the finger control (e.g., "th_dip", "if_pip", etc.)
            obs_type: Type of observation ("pos", "vel", "target")
            
        Returns:
            Index in the concatenated observation tensor
        """
        if control_name not in self.control_name_to_index:
            raise ValueError(f"Unknown control name: {control_name}. Available: {list(self.control_name_to_index.keys())}")
        
        control_idx = self.control_name_to_index[control_name]
        
        # Determine which observation component to use
        if obs_type == "pos":
            component_key = "active_finger_dof_pos"
        elif obs_type == "vel":
            component_key = "active_finger_dof_vel"
        elif obs_type == "target":
            component_key = "active_finger_dof_target"
        else:
            raise ValueError(f"Unknown obs_type: {obs_type}. Available: pos, vel, target")
        
        if component_key not in self.component_slice_indices:
            raise ValueError(f"Observation component {component_key} not found in observation keys")
        
        component_start, _ = self.component_slice_indices[component_key]
        return component_start + control_idx

    def get_raw_finger_dof(self, dof_name: str, obs_type: str = "pos", 
                           obs_data = None, env_idx: int = None):
        """
        Get finger DOF value by raw DOF name (including inactive DOFs like r_f_joint3_1).
        
        Args:
            dof_name: Name of the raw finger DOF (e.g., "r_f_joint1_1", "r_f_joint3_1", etc.)
            obs_type: Type of observation ("pos", "vel", "target")
            obs_data: Either observation tensor OR observation dictionary (not both)
            env_idx: Environment index (if None, returns data for all environments)
            
        Returns:
            DOF value(s) - scalar if env_idx specified, tensor if env_idx is None
        """
        if dof_name not in self.raw_dof_name_to_index:
            raise ValueError(f"Unknown DOF name: {dof_name}. Available: {list(self.raw_dof_name_to_index.keys())}")
        
        if obs_data is None:
            raise ValueError("obs_data must be provided")
        
        # Determine which observation component to use
        if obs_type == "pos":
            component_key = "all_finger_dof_pos"
        elif obs_type == "vel":
            component_key = "all_finger_dof_vel"
        elif obs_type == "target":
            component_key = "all_finger_dof_target"
        else:
            raise ValueError(f"Unknown obs_type: {obs_type}. Available: pos, vel, target")
        
        # Check if obs_data is a dictionary
        if isinstance(obs_data, dict):
            if component_key not in obs_data:
                raise ValueError(f"Component '{component_key}' not found in obs_dict")
            
            raw_dof_idx = self.raw_dof_name_to_index[dof_name]
            finger_dof_idx = raw_dof_idx - self.NUM_BASE_DOFS
            
            if finger_dof_idx < 0:
                raise ValueError(f"DOF {dof_name} is not a finger DOF (index {raw_dof_idx} < {self.NUM_BASE_DOFS})")
            
            data = obs_data[component_key][:, finger_dof_idx]
            return data[env_idx].item() if env_idx is not None else data
        
        # obs_data is a tensor - raw finger DOFs not available in concatenated tensor
        else:
            raise ValueError(f"Raw finger DOF '{dof_name}' access requires obs_dict. "
                           f"obs_tensor only contains active finger DOFs, not all raw DOFs.")

    def get_obs_range_for_finger_pose(self, finger_body_name: str, frame: str = "world"):
        """
        Get the index range in the concatenated observation tensor for a finger pose.
        
        Args:
            finger_body_name: Name of the finger body (e.g., "r_f_link1_tip", "r_f_link2_pad", etc.)
            frame: Reference frame ("world" or "hand")
            
        Returns:
            Tuple of (start_index, end_index) for the 7D pose in concatenated tensor
        """
        if finger_body_name not in self.finger_body_to_index:
            raise ValueError(f"Unknown finger body name: {finger_body_name}. Available: {list(self.finger_body_to_index.keys())}")
        
        body_type, finger_idx = self.finger_body_to_index[finger_body_name]
        
        # Determine which observation component to use
        if body_type == "fingertip":
            component_key = f"fingertip_poses_{frame}"
        elif body_type == "fingerpad":
            component_key = f"fingerpad_poses_{frame}"
        else:
            raise ValueError(f"Unknown body type: {body_type}")
        
        if component_key not in self.component_slice_indices:
            raise ValueError(f"Observation component {component_key} not found in observation keys")
        
        component_start, _ = self.component_slice_indices[component_key]
        pose_start = component_start + finger_idx * 7
        pose_end = pose_start + 7
        
        return (pose_start, pose_end)

    def get_obs_range_for_contact_force(self, finger_idx: int):
        """
        Get the index range in the concatenated observation tensor for contact force.
        
        Args:
            finger_idx: Finger index (0-4 for 5 fingers)
            
        Returns:
            Tuple of (start_index, end_index) for the 3D force in concatenated tensor
        """
        if finger_idx < 0 or finger_idx >= 5:
            raise ValueError(f"Finger index must be 0-4, got {finger_idx}")
        
        if "contact_forces" not in self.component_slice_indices:
            raise ValueError("Contact forces not found in observation keys")
        
        component_start, _ = self.component_slice_indices["contact_forces"]
        force_start = component_start + finger_idx * 3
        force_end = force_start + 3
        
        return (force_start, force_end)

    def get_base_dof_value(self, joint_name: str, obs_type: str = "pos", obs_data = None, env_idx: int = None):
        """
        Get base DOF value by joint name.
        
        Args:
            joint_name: Name of the base joint (e.g., "ARTx", "ARTy", etc.)
            obs_type: Type of observation ("pos", "vel", "target")
            obs_data: Either observation tensor OR observation dictionary (not both)
            env_idx: Environment index (if None, returns data for all environments)
            
        Returns:
            DOF value(s) - scalar if env_idx specified, tensor if env_idx is None
        """
        if joint_name not in self.base_joint_to_index:
            raise ValueError(f"Unknown base joint name: {joint_name}. Available: {list(self.base_joint_to_index.keys())}")
        
        if obs_data is None:
            raise ValueError("obs_data must be provided")
        
        joint_idx = self.base_joint_to_index[joint_name]
        
        # Determine which observation component to use
        if obs_type == "pos":
            component_key = "base_dof_pos"
        elif obs_type == "vel":
            component_key = "base_dof_vel"
        elif obs_type == "target":
            component_key = "base_dof_target"
        else:
            raise ValueError(f"Unknown obs_type: {obs_type}. Available: pos, vel, target")
        
        # Check if obs_data is a dictionary
        if isinstance(obs_data, dict):
            if component_key not in obs_data:
                raise ValueError(f"Component '{component_key}' not found in obs_dict")
            
            data = obs_data[component_key][:, joint_idx]
            return data[env_idx].item() if env_idx is not None else data
        
        # obs_data is a tensor
        else:
            if component_key not in self.component_slice_indices:
                raise ValueError(f"Component '{component_key}' not found in observation tensor")
            
            component_start, _ = self.component_slice_indices[component_key]
            tensor_idx = component_start + joint_idx
            
            if env_idx is not None:
                return obs_data[env_idx, tensor_idx].item()
            else:
                return obs_data[:, tensor_idx]

    def get_active_finger_dof_value(self, control_name: str, obs_type: str = "pos", obs_data = None, env_idx: int = None):
        """
        Get active finger DOF value by control name.
        
        Args:
            control_name: Name of the finger control (e.g., "th_dip", "if_pip", etc.)
            obs_type: Type of observation ("pos", "vel", "target")
            obs_data: Either observation tensor OR observation dictionary (not both)
            env_idx: Environment index (if None, returns data for all environments)
            
        Returns:
            DOF value(s) - scalar if env_idx specified, tensor if env_idx is None
        """
        if control_name not in self.control_name_to_index:
            raise ValueError(f"Unknown control name: {control_name}. Available: {list(self.control_name_to_index.keys())}")
        
        if obs_data is None:
            raise ValueError("obs_data must be provided")
        
        control_idx = self.control_name_to_index[control_name]
        
        # Determine which observation component to use
        if obs_type == "pos":
            component_key = "active_finger_dof_pos"
        elif obs_type == "vel":
            component_key = "active_finger_dof_vel"
        elif obs_type == "target":
            component_key = "active_finger_dof_target"
        else:
            raise ValueError(f"Unknown obs_type: {obs_type}. Available: pos, vel, target")
        
        # Check if obs_data is a dictionary
        if isinstance(obs_data, dict):
            if component_key not in obs_data:
                raise ValueError(f"Component '{component_key}' not found in obs_dict")
            
            data = obs_data[component_key][:, control_idx]
            return data[env_idx].item() if env_idx is not None else data
        
        # obs_data is a tensor
        else:
            if component_key not in self.component_slice_indices:
                raise ValueError(f"Component '{component_key}' not found in observation tensor")
            
            component_start, _ = self.component_slice_indices[component_key]
            tensor_idx = component_start + control_idx
            
            if env_idx is not None:
                return obs_data[env_idx, tensor_idx].item()
            else:
                return obs_data[:, tensor_idx]

    def get_finger_pose_value(self, finger_body_name: str, frame: str = "world", obs_data = None, env_idx: int = None):
        """
        Get finger pose by body name.
        
        Args:
            finger_body_name: Name of the finger body (e.g., "r_f_link1_tip", "r_f_link2_pad", etc.)
            frame: Reference frame ("world" or "hand")
            obs_data: Either observation tensor OR observation dictionary (not both)
            env_idx: Environment index (if None, returns data for all environments)
            
        Returns:
            Pose data - dict with 'position' and 'orientation' (numpy arrays if env_idx specified, tensors if None)
        """
        if finger_body_name not in self.finger_body_to_index:
            raise ValueError(f"Unknown finger body name: {finger_body_name}. Available: {list(self.finger_body_to_index.keys())}")
        
        if obs_data is None:
            raise ValueError("obs_data must be provided")
        
        body_type, finger_idx = self.finger_body_to_index[finger_body_name]
        
        # Determine which observation component to use
        if body_type == "fingertip":
            component_key = f"fingertip_poses_{frame}"
        elif body_type == "fingerpad":
            component_key = f"fingerpad_poses_{frame}"
        else:
            raise ValueError(f"Unknown body type: {body_type}")
        
        # Check if obs_data is a dictionary
        if isinstance(obs_data, dict):
            if component_key not in obs_data:
                raise ValueError(f"Component '{component_key}' not found in obs_dict")
            
            pose_start = finger_idx * 7
            pose_end = pose_start + 7
            pose_data = obs_data[component_key][:, pose_start:pose_end]
            
            if env_idx is not None:
                return {
                    'position': pose_data[env_idx, :3].cpu().numpy(),
                    'orientation': pose_data[env_idx, 3:7].cpu().numpy()
                }
            else:
                return {
                    'position': pose_data[:, :3],
                    'orientation': pose_data[:, 3:7]
                }
        
        # obs_data is a tensor
        else:
            if component_key not in self.component_slice_indices:
                raise ValueError(f"Component '{component_key}' not found in observation tensor")
            
            component_start, _ = self.component_slice_indices[component_key]
            pose_start = component_start + finger_idx * 7
            pose_end = pose_start + 7
            
            if env_idx is not None:
                pose_data = obs_data[env_idx, pose_start:pose_end]
                return {
                    'position': pose_data[:3].cpu().numpy(),
                    'orientation': pose_data[3:7].cpu().numpy()
                }
            else:
                pose_data = obs_data[:, pose_start:pose_end]
                return {
                    'position': pose_data[:, :3],
                    'orientation': pose_data[:, 3:7]
                }

    def get_contact_force_value(self, finger_idx: int, obs_data = None, env_idx: int = None):
        """
        Get contact force by finger index.
        
        Args:
            finger_idx: Finger index (0-4 for 5 fingers)
            obs_data: Either observation tensor OR observation dictionary (not both)
            env_idx: Environment index (if None, returns data for all environments)
            
        Returns:
            Contact force - numpy array if env_idx specified, tensor if env_idx is None
        """
        if finger_idx < 0 or finger_idx >= 5:
            raise ValueError(f"Finger index must be 0-4, got {finger_idx}")
        
        if obs_data is None:
            raise ValueError("obs_data must be provided")
        
        component_key = "contact_forces"
        
        # Check if obs_data is a dictionary
        if isinstance(obs_data, dict):
            if component_key not in obs_data:
                raise ValueError(f"Component '{component_key}' not found in obs_dict")
            
            force_start = finger_idx * 3
            force_end = force_start + 3
            force_data = obs_data[component_key][:, force_start:force_end]
            
            if env_idx is not None:
                return force_data[env_idx].cpu().numpy()
            else:
                return force_data
        
        # obs_data is a tensor
        else:
            if component_key not in self.component_slice_indices:
                raise ValueError(f"Component '{component_key}' not found in observation tensor")
            
            component_start, _ = self.component_slice_indices[component_key]
            force_start = component_start + finger_idx * 3
            force_end = force_start + 3
            
            if env_idx is not None:
                return obs_data[env_idx, force_start:force_end].cpu().numpy()
            else:
                return obs_data[:, force_start:force_end]

    def parse_obs_tensor(self, obs_tensor: torch.Tensor = None, obs_dict: Dict[str, torch.Tensor] = None, env_idx: int = 0):
        """
        Parse observation tensor and provide convenient access to components.
        
        Args:
            obs_tensor: Observation tensor to parse (if None, uses current observation)
            obs_dict: Observation dictionary for accessing all_finger_dof components (if None, uses limited access)
            env_idx: Environment index
            
        Returns:
            ObservationParser instance with convenient accessor methods
        """
        if obs_tensor is None:
            obs_tensor = self.obs_buf
        
        if obs_tensor is None:
            raise ValueError("No observation tensor available")
        
        # Extract the observation for the specified environment
        if len(obs_tensor.shape) == 1:
            obs_data = obs_tensor  # Already single environment
        else:
            obs_data = obs_tensor[env_idx]
        
        return ObservationParser(obs_data, self, obs_dict)


class ObservationParser:
    """
    Helper class to parse and access observation tensor components.
    """
    
    def __init__(self, obs_data: torch.Tensor, encoder: 'ObservationEncoder', obs_dict: Dict[str, torch.Tensor] = None):
        self.obs_data = obs_data
        self.encoder = encoder
        self.obs_dict = obs_dict
    
    def get_base_dof(self, joint_name: str, obs_type: str = "pos"):
        """Get base DOF value by joint name."""
        idx = self.encoder.get_obs_index_for_base_joint(joint_name, obs_type)
        return self.obs_data[idx].item()
    
    def get_finger_dof(self, control_name: str, obs_type: str = "pos"):
        """Get finger DOF value by control name."""
        idx = self.encoder.get_obs_index_for_finger_control(control_name, obs_type)
        return self.obs_data[idx].item()
    
    def get_finger_pose(self, finger_body_name: str, frame: str = "world"):
        """Get finger pose by body name."""
        start_idx, end_idx = self.encoder.get_obs_range_for_finger_pose(finger_body_name, frame)
        pose_data = self.obs_data[start_idx:end_idx]
        return {
            'position': pose_data[:3].cpu().numpy(),
            'orientation': pose_data[3:7].cpu().numpy()
        }
    
    def get_contact_force(self, finger_idx: int):
        """Get contact force by finger index."""
        start_idx, end_idx = self.encoder.get_obs_range_for_contact_force(finger_idx)
        return self.obs_data[start_idx:end_idx].cpu().numpy()
    
    def get_raw_finger_dof(self, dof_name: str, obs_type: str = "pos"):
        """
        Get any finger DOF value by raw DOF name (including inactive DOFs).
        Requires obs_dict to be provided during parsing.
        
        Args:
            dof_name: Name of the raw finger DOF (e.g., "r_f_joint1_1", "r_f_joint3_1", etc.)
            obs_type: Type of observation ("pos", "vel", "target")
            
        Returns:
            DOF value for the specified finger DOF (single environment)
        """
        if self.obs_dict is None:
            raise ValueError("obs_dict must be provided during parsing to access raw finger DOF values")
        
        return self.encoder.get_raw_finger_dof(dof_name, obs_type, self.obs_dict, env_idx=0)