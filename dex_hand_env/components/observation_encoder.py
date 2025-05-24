"""
Observation encoder component for DexHand environment.

This module provides observation encoding functionality for the DexHand environment,
including proprioceptive states, sensor readings, and task-specific observations.
"""

# Import standard libraries
import torch
import numpy as np

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
    """
    
    def __init__(self, gym, sim, num_envs, device):
        """
        Initialize the observation encoder.
        
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
        
        # Configuration flags
        self.include_dof_pos = True
        self.include_dof_vel = True
        self.include_hand_pose = True
        self.include_contact_forces = True
        self.include_actions = False
        
        # Constants for observation dimensions
        self.NUM_BASE_DOFS = 6
        self.NUM_ACTIVE_FINGER_DOFS = 12
        self.NUM_FINGERS = 5
        
        # Initialize observation buffers
        self.obs_buf = None
        self.states_buf = None
        self.obs_dict = {}
        
        # Track observation keys and dimensions
        self.obs_keys = []
        self.num_observations = 0
        
        # Cached tensors
        self.dof_pos = None
        self.dof_vel = None
        self.root_state_tensor = None
        self.contact_forces = None
        self.prev_actions = None
    
    def configure(self, include_dof_pos=None, include_dof_vel=None, 
                 include_hand_pose=None, include_contact_forces=None,
                 include_actions=None):
        """
        Configure which observations to include.
        
        Args:
            include_dof_pos: Include DOF positions
            include_dof_vel: Include DOF velocities
            include_hand_pose: Include hand root pose
            include_contact_forces: Include contact force sensors
            include_actions: Include previous actions
        """
        if include_dof_pos is not None:
            self.include_dof_pos = include_dof_pos
        if include_dof_vel is not None:
            self.include_dof_vel = include_dof_vel
        if include_hand_pose is not None:
            self.include_hand_pose = include_hand_pose
        if include_contact_forces is not None:
            self.include_contact_forces = include_contact_forces
        if include_actions is not None:
            self.include_actions = include_actions
        
        # Recalculate observation space
        self._calculate_obs_dim()
    
    def _calculate_obs_dim(self):
        """
        Calculate observation dimension based on configuration.
        
        Returns:
            Total observation dimension
        """
        self.obs_keys = []
        self.num_observations = 0
        
        if self.include_dof_pos:
            # DOF positions (base + fingers)
            self.obs_keys.append("dof_pos")
            self.num_observations += self.NUM_BASE_DOFS + self.NUM_ACTIVE_FINGER_DOFS
            
        if self.include_dof_vel:
            # DOF velocities (base + fingers)
            self.obs_keys.append("dof_vel")
            self.num_observations += self.NUM_BASE_DOFS + self.NUM_ACTIVE_FINGER_DOFS
            
        if self.include_hand_pose:
            # Hand root pose (position + orientation)
            self.obs_keys.append("hand_pose")
            self.num_observations += 7  # 3 for position, 4 for quaternion
            
        if self.include_contact_forces:
            # Contact forces (3D force for each finger)
            self.obs_keys.append("contact_forces")
            self.num_observations += self.NUM_FINGERS * 3
            
        if self.include_actions:
            # Previous actions
            self.obs_keys.append("prev_actions")
            self.num_observations += self.NUM_BASE_DOFS + self.NUM_ACTIVE_FINGER_DOFS
        
        return self.num_observations
    
    def initialize_buffers(self, num_dof):
        """
        Initialize observation buffers.
        
        Args:
            num_dof: Total number of DOFs in the model
        """
        # Calculate observation dimension
        self._calculate_obs_dim()
        
        # Create observation buffer
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_observations), device=self.device
        )
        
        # Create state buffer (full state, not just observation)
        # This can be used for asymmetric actor-critic algorithms
        self.states_buf = torch.zeros(
            (self.num_envs, self.num_observations), device=self.device
        )
        
        # Initialize dictionary to store individual observation components
        self.obs_dict = {}
        
        # Initialize contact forces tensor
        self.contact_forces = torch.zeros(
            (self.num_envs, self.NUM_FINGERS, 3), device=self.device
        )
        
        # Initialize previous actions tensor
        self.prev_actions = torch.zeros(
            (self.num_envs, self.NUM_BASE_DOFS + self.NUM_ACTIVE_FINGER_DOFS), 
            device=self.device
        )
    
    def update_cached_tensors(self, dof_pos=None, dof_vel=None, root_state_tensor=None):
        """
        Update cached tensors with latest values from simulation.
        
        Args:
            dof_pos: DOF position tensor
            dof_vel: DOF velocity tensor
            root_state_tensor: Root state tensor
        """
        if dof_pos is not None:
            self.dof_pos = dof_pos
        if dof_vel is not None:
            self.dof_vel = dof_vel
        if root_state_tensor is not None:
            self.root_state_tensor = root_state_tensor
    
    def update_contact_forces(self, contact_forces):
        """
        Update contact force readings.
        
        Args:
            contact_forces: Contact force tensor
        """
        self.contact_forces = contact_forces
    
    def update_prev_actions(self, actions):
        """
        Update previous actions.
        
        Args:
            actions: Current action tensor
        """
        if actions is not None and hasattr(actions, 'shape'):
            self.prev_actions = actions.clone()
    
    def compute_observations(self, hand_indices=None, fingertip_indices=None, joint_to_control=None, active_joint_names=None):
        """
        Compute the observation vector.
        
        Args:
            hand_indices: Indices of hand actors
            fingertip_indices: Indices of fingertips
            joint_to_control: Mapping from joint names to control names
            active_joint_names: List of active joint names
            
        Returns:
            Observation tensor and dictionary of components
        """
        # Reset observation dictionary
        self.obs_dict = {}
        
        # Current buffer index for filling observations
        obs_idx = 0
        
        # Safety check - make sure tensors are initialized
        if self.dof_pos is None or self.dof_vel is None or self.root_state_tensor is None:
            print("Warning: Tensor handles not initialized. Cannot compute observations.")
            return self.obs_buf, self.obs_dict
            
        if joint_to_control is None or active_joint_names is None:
            print("Warning: Joint mappings not provided. Some observations may be incorrect.")
        
        # DOF positions
        if self.include_dof_pos:
            # Select joint positions (base + active finger DOFs)
            active_positions = torch.zeros(
                (self.num_envs, self.NUM_BASE_DOFS + self.NUM_ACTIVE_FINGER_DOFS),
                device=self.device
            )
            
            # Copy base DOF positions
            active_positions[:, :self.NUM_BASE_DOFS] = self.dof_pos[:, :self.NUM_BASE_DOFS]
            
            # Map finger DOF positions to active DOFs
            if joint_to_control is not None and active_joint_names is not None:
                for i, name in enumerate(self.gym.get_actor_dof_names(self.sim)[self.NUM_BASE_DOFS:]):
                    # Skip if not in joint_to_control mapping
                    if name not in joint_to_control:
                        continue
                        
                    # Map from joint name to control index
                    try:
                        control_name = joint_to_control[name]
                        try:
                            control_idx = active_joint_names.index(control_name)
                            active_idx = self.NUM_BASE_DOFS + control_idx
                            
                            if active_idx < active_positions.shape[1]:
                                # DOF index in the full state
                                dof_idx = i + self.NUM_BASE_DOFS
                                
                                # Copy the position value
                                if dof_idx < self.dof_pos.shape[1]:
                                    active_positions[:, active_idx] = self.dof_pos[:, dof_idx]
                        except (ValueError, IndexError) as e:
                            pass  # Safely continue if mapping fails
                    except KeyError:
                        pass  # Safely continue if joint not in mapping
            
            # Store in observation buffer
            self.obs_buf[:, obs_idx:obs_idx + active_positions.shape[1]] = active_positions
            obs_idx += active_positions.shape[1]
            
            # Store in observation dictionary
            self.obs_dict["dof_pos"] = active_positions
        
        # DOF velocities
        if self.include_dof_vel:
            # Select joint velocities (base + active finger DOFs)
            active_velocities = torch.zeros(
                (self.num_envs, self.NUM_BASE_DOFS + self.NUM_ACTIVE_FINGER_DOFS),
                device=self.device
            )
            
            # Copy base DOF velocities
            active_velocities[:, :self.NUM_BASE_DOFS] = self.dof_vel[:, :self.NUM_BASE_DOFS]
            
            # Map finger DOF velocities to active DOFs
            if joint_to_control is not None and active_joint_names is not None:
                for i, name in enumerate(self.gym.get_actor_dof_names(self.sim)[self.NUM_BASE_DOFS:]):
                    # Skip if not in joint_to_control mapping
                    if name not in joint_to_control:
                        continue
                        
                    # Map from joint name to control index
                    try:
                        control_name = joint_to_control[name]
                        try:
                            control_idx = active_joint_names.index(control_name)
                            active_idx = self.NUM_BASE_DOFS + control_idx
                            
                            if active_idx < active_velocities.shape[1]:
                                # DOF index in the full state
                                dof_idx = i + self.NUM_BASE_DOFS
                                
                                # Copy the velocity value
                                if dof_idx < self.dof_vel.shape[1]:
                                    active_velocities[:, active_idx] = self.dof_vel[:, dof_idx]
                        except (ValueError, IndexError) as e:
                            pass  # Safely continue if mapping fails
                    except KeyError:
                        pass  # Safely continue if joint not in mapping
            
            # Store in observation buffer
            self.obs_buf[:, obs_idx:obs_idx + active_velocities.shape[1]] = active_velocities
            obs_idx += active_velocities.shape[1]
            
            # Store in observation dictionary
            self.obs_dict["dof_vel"] = active_velocities
        
        # Hand pose (position and orientation)
        if self.include_hand_pose and hand_indices is not None:
            if len(hand_indices) > 0:
                hand_poses = torch.zeros((self.num_envs, 7), device=self.device)
                
                # Extract root state for hand actors
                for i, hand_idx in enumerate(hand_indices):
                    if i >= self.num_envs:
                        break
                        
                    # Get actor index in the root state tensor
                    hand_root_idx = hand_idx * 13
                    
                    if hand_root_idx + 7 <= self.root_state_tensor.shape[1]:
                        # Position (3) and orientation (4)
                        hand_poses[i, :7] = self.root_state_tensor[i, hand_root_idx:hand_root_idx + 7]
                
                # Store in observation buffer
                self.obs_buf[:, obs_idx:obs_idx + 7] = hand_poses
                obs_idx += 7
                
                # Store in observation dictionary
                self.obs_dict["hand_pose"] = hand_poses
            else:
                # No hand indices provided, fill with zeros
                self.obs_buf[:, obs_idx:obs_idx + 7] = 0
                obs_idx += 7
                
                # Store in observation dictionary
                self.obs_dict["hand_pose"] = torch.zeros((self.num_envs, 7), device=self.device)
        
        # Contact forces
        if self.include_contact_forces:
            # Reshape contact forces to flat tensor
            flat_contacts = self.contact_forces.reshape(self.num_envs, -1)
            
            # Store in observation buffer
            self.obs_buf[:, obs_idx:obs_idx + flat_contacts.shape[1]] = flat_contacts
            obs_idx += flat_contacts.shape[1]
            
            # Store in observation dictionary
            self.obs_dict["contact_forces"] = flat_contacts
        
        # Previous actions
        if self.include_actions:
            # Store in observation buffer
            self.obs_buf[:, obs_idx:obs_idx + self.prev_actions.shape[1]] = self.prev_actions
            obs_idx += self.prev_actions.shape[1]
            
            # Store in observation dictionary
            self.obs_dict["prev_actions"] = self.prev_actions
        
        # Copy to states buffer (for asymmetric actor-critic)
        self.states_buf = self.obs_buf.clone()
        
        return self.obs_buf, self.obs_dict
    
    def add_task_observations(self, task_obs_dict):
        """
        Add task-specific observations to the observation buffer.
        
        Args:
            task_obs_dict: Dictionary of task-specific observations
            
        Returns:
            Updated observation tensor and dictionary
        """
        # Start from current observation index
        obs_idx = self._get_current_obs_index()
        
        # Add each task observation to the buffer
        for key, value in task_obs_dict.items():
            if isinstance(value, torch.Tensor):
                # Reshape to flat tensor if needed
                if len(value.shape) > 2:
                    flat_value = value.reshape(self.num_envs, -1)
                else:
                    flat_value = value
                
                # Check if buffer needs resizing
                if obs_idx + flat_value.shape[1] > self.obs_buf.shape[1]:
                    # Resize observation buffer
                    new_buf = torch.zeros(
                        (self.num_envs, obs_idx + flat_value.shape[1]),
                        device=self.device
                    )
                    new_buf[:, :self.obs_buf.shape[1]] = self.obs_buf
                    self.obs_buf = new_buf
                    
                    # Resize state buffer
                    new_states = torch.zeros(
                        (self.num_envs, obs_idx + flat_value.shape[1]),
                        device=self.device
                    )
                    new_states[:, :self.states_buf.shape[1]] = self.states_buf
                    self.states_buf = new_states
                
                # Add to observation buffer
                self.obs_buf[:, obs_idx:obs_idx + flat_value.shape[1]] = flat_value
                obs_idx += flat_value.shape[1]
                
                # Add to observation dictionary
                self.obs_dict[key] = flat_value
                
                # Add to obs_keys if not already present
                if key not in self.obs_keys:
                    self.obs_keys.append(key)
        
        # Update observation dimension
        self.num_observations = self.obs_buf.shape[1]
        
        # Copy to states buffer
        self.states_buf = self.obs_buf.clone()
        
        return self.obs_buf, self.obs_dict
    
    def _get_current_obs_index(self):
        """
        Get the current index in the observation buffer.
        
        Returns:
            Current observation index
        """
        obs_idx = 0
        
        if self.include_dof_pos:
            obs_idx += self.NUM_BASE_DOFS + self.NUM_ACTIVE_FINGER_DOFS
            
        if self.include_dof_vel:
            obs_idx += self.NUM_BASE_DOFS + self.NUM_ACTIVE_FINGER_DOFS
            
        if self.include_hand_pose:
            obs_idx += 7  # 3 for position, 4 for quaternion
            
        if self.include_contact_forces:
            obs_idx += self.NUM_FINGERS * 3
            
        if self.include_actions:
            obs_idx += self.NUM_BASE_DOFS + self.NUM_ACTIVE_FINGER_DOFS
        
        return obs_idx
    
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