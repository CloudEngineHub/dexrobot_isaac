"""
Hand initializer component for DexHand environment.

This module provides hand model initialization functionality for the DexHand environment,
including loading assets, creating actors, and setting up initial states.
"""

# Import standard libraries
import os
import torch
import numpy as np

# Import IsaacGym
from isaacgym import gymapi, gymtorch


class HandInitializer:
    """
    Handles initialization of the robotic hand model in the environment.
    
    This component provides functionality to:
    - Load hand assets and meshes
    - Create hand instances in each environment
    - Set up initial joint and position states
    - Configure joint properties like limits and stiffness
    """
    
    def __init__(self, gym, sim, num_envs, device, asset_root=None):
        """
        Initialize the hand initializer.
        
        Args:
            gym: The isaacgym gym instance
            sim: The isaacgym simulation instance
            num_envs: Number of environments
            device: PyTorch device
            asset_root: Root directory for assets
        """
        self.gym = gym
        self.sim = sim
        self.num_envs = num_envs
        self.device = device
        
        # Set asset root
        if asset_root is None:
            # Default to assets directory in parent of current file
            current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.asset_root = os.path.join(current_dir, "assets")
        else:
            self.asset_root = asset_root
            
        # Default hand model path
        self.hand_asset_file = "dexrobot_mujoco/dexrobot_mujoco/models/dexhand021_right_simplified_floating.xml"
        
        # Define joint names for use in hand creation
        self.base_joint_names = [
            "ARTx", "ARTy", "ARTz", "ARRx", "ARRy", "ARRz"
        ]
        
        self.finger_joint_names = [
            "r_f_joint1_1", "r_f_joint1_2", "r_f_joint1_3", "r_f_joint1_4",
            "r_f_joint2_1", "r_f_joint2_2", "r_f_joint2_3", "r_f_joint2_4",
            "r_f_joint3_1", "r_f_joint3_2", "r_f_joint3_3", "r_f_joint3_4",
            "r_f_joint4_1", "r_f_joint4_2", "r_f_joint4_3", "r_f_joint4_4",
            "r_f_joint5_1", "r_f_joint5_2", "r_f_joint5_3", "r_f_joint5_4"
        ]
        
        # Map from hardware active DoFs to full finger joint space
        # based on HardwareMapping from dexhand_ros.py
        self.active_joint_mapping = {
            "th_adduction": ["r_f_joint1_1"],
            "th_mcp": ["r_f_joint1_2"],
            "th_pip": ["r_f_joint1_3", "r_f_joint1_4"],
            "ix_adduction": ["r_f_joint2_1"],
            "ix_mcp": ["r_f_joint2_2"],
            "ix_pip": ["r_f_joint2_3", "r_f_joint2_4"],
            "mf_adduction": ["r_f_joint3_1"],
            "mf_mcp": ["r_f_joint3_2"],
            "mf_pip": ["r_f_joint3_3", "r_f_joint3_4"],
            "rf_adduction": ["r_f_joint4_1"],
            "rf_mcp": ["r_f_joint4_2"],
            "rf_pip": ["r_f_joint4_3", "r_f_joint4_4"],
            "lf_adduction": ["r_f_joint5_1"],
            "lf_mcp": ["r_f_joint5_2"],
            "lf_pip": ["r_f_joint5_3", "r_f_joint5_4"]
        }
        
        # Create reverse mapping from joint name to controller
        self.joint_to_control = {}
        for control, joints in self.active_joint_mapping.items():
            for joint in joints:
                self.joint_to_control[joint] = control
        
        # Active joint names (12 DoFs that can be controlled directly)
        self.active_joint_names = list(self.active_joint_mapping.keys())
        
        # Body names for fingertips and fingerpads in the MJCF model
        self.fingertip_body_names = [
            "r_f_link1_tip", "r_f_link2_tip", "r_f_link3_tip",
            "r_f_link4_tip", "r_f_link5_tip"
        ]
        
        self.fingerpad_body_names = [
            "r_f_link1_pad", "r_f_link2_pad", "r_f_link3_pad",
            "r_f_link4_pad", "r_f_link5_pad"
        ]
        
        # Storage for handles and indices
        self.hand_handles = []
        self.fingertip_body_handles = []
        self.fingerpad_body_handles = []
        self.hand_indices = []
        self.fingertip_indices = []
        
        # Initial pose settings
        self.initial_hand_pos = [0.0, 0.0, 0.5]
        self.initial_hand_rot = [0.0, 0.0, 0.0, 1.0]
        
        # Joint control settings
        self.base_stiffness = 400.0
        self.base_damping = 40.0
        self.finger_stiffness = 100.0
        self.finger_damping = 10.0
    
    def set_initial_pose(self, pos, rot=None):
        """
        Set the initial pose for the hand.
        
        Args:
            pos: List or tensor with [x, y, z] position
            rot: List or tensor with [x, y, z, w] quaternion rotation
        """
        self.initial_hand_pos = pos
        if rot is not None:
            self.initial_hand_rot = rot
    
    def set_joint_properties(self, base_stiffness=None, base_damping=None, 
                           finger_stiffness=None, finger_damping=None):
        """
        Set joint control properties for the hand.
        
        Args:
            base_stiffness: Stiffness for base joints
            base_damping: Damping for base joints
            finger_stiffness: Stiffness for finger joints
            finger_damping: Damping for finger joints
        """
        if base_stiffness is not None:
            self.base_stiffness = base_stiffness
        if base_damping is not None:
            self.base_damping = base_damping
        if finger_stiffness is not None:
            self.finger_stiffness = finger_stiffness
        if finger_damping is not None:
            self.finger_damping = finger_damping
    
    def load_hand_asset(self, asset_file=None):
        """
        Load the hand asset from file.
        
        Args:
            asset_file: Optional override for asset file path
            
        Returns:
            The loaded asset
        """
        if asset_file is not None:
            self.hand_asset_file = asset_file
            
        # Verify asset file exists
        asset_full_path = os.path.join(self.asset_root, self.hand_asset_file)
        if not os.path.exists(asset_full_path):
            raise FileNotFoundError(f"Hand asset file not found: {asset_full_path}")
        
        # Set asset options
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = False
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        
        try:
            # Load the hand asset
            hand_asset = self.gym.load_asset(
                self.sim, self.asset_root, self.hand_asset_file, asset_options
            )
            return hand_asset
        except Exception as e:
            print(f"Error loading hand asset: {e}")
            raise
    
    def create_hands(self, envs, hand_asset):
        """
        Create hand instances in all environments.
        
        Args:
            envs: List of environment instances
            hand_asset: The hand asset to instantiate
            
        Returns:
            Lists of handles to the created hands and their components
        """
        self.hand_handles = []
        self.fingertip_body_handles = []
        self.fingerpad_body_handles = []
        self.hand_indices = []
        self.fingertip_indices = []
        
        # Get DOF properties
        hand_dof_props = self.gym.get_asset_dof_properties(hand_asset)
        
        # Configure DOF properties for PD control
        for i, name in enumerate(self.gym.get_asset_dof_names(hand_asset)):
            # Base joints (translational and rotational)
            if any(base_name in name for base_name in self.base_joint_names):
                hand_dof_props["stiffness"][i] = self.base_stiffness
                hand_dof_props["damping"][i] = self.base_damping
            # Finger joints
            elif any(finger_name in name for finger_name in self.finger_joint_names):
                hand_dof_props["stiffness"][i] = self.finger_stiffness
                hand_dof_props["damping"][i] = self.finger_damping
        
        # Set drive mode for PD control
        hand_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
        
        # Initial pose
        hand_pose = gymapi.Transform()
        hand_pose.p = gymapi.Vec3(*self.initial_hand_pos)
        hand_pose.r = gymapi.Quat(*self.initial_hand_rot)
        
        # Create hands in all environments
        for i, env in enumerate(envs):
            # Create hand
            hand_handle = self.gym.create_actor(
                env, hand_asset, hand_pose, f"hand_{i}", i, 0
            )
            
            # Set DOF properties
            self.gym.set_actor_dof_properties(env, hand_handle, hand_dof_props)
            
            # Get global index
            hand_idx = self.gym.get_actor_index(env, hand_handle, gymapi.DOMAIN_SIM)
            
            # Store handles and indices
            self.hand_handles.append(hand_handle)
            self.hand_indices.append(hand_idx)
            
            # Get fingertip body handles
            fingertip_body_handles = []
            for name in self.fingertip_body_names:
                fingertip_body_handles.append(
                    self.gym.find_actor_rigid_body_handle(env, hand_handle, name)
                )
            self.fingertip_body_handles.append(fingertip_body_handles)
            
            # Get fingerpad body handles
            fingerpad_body_handles = []
            for name in self.fingerpad_body_names:
                fingerpad_body_handles.append(
                    self.gym.find_actor_rigid_body_handle(env, hand_handle, name)
                )
            self.fingerpad_body_handles.append(fingerpad_body_handles)
            
            # Get fingertip indices
            fingertip_indices = []
            for name in self.fingertip_body_names:
                body_idx = self.gym.find_actor_rigid_body_index(env, hand_handle, name, gymapi.DOMAIN_SIM)
                fingertip_indices.append(body_idx)
            self.fingertip_indices.append(fingertip_indices)
        
        return {
            "hand_handles": self.hand_handles,
            "fingertip_body_handles": self.fingertip_body_handles,
            "fingerpad_body_handles": self.fingerpad_body_handles,
            "hand_indices": self.hand_indices,
            "fingertip_indices": self.fingertip_indices
        }
    
    def get_dof_mapping(self):
        """
        Get mapping of joint names to active DOFs.
        
        Returns:
            Dictionary with mapping information
        """
        return {
            "base_joint_names": self.base_joint_names,
            "finger_joint_names": self.finger_joint_names,
            "active_joint_mapping": self.active_joint_mapping,
            "joint_to_control": self.joint_to_control,
            "active_joint_names": self.active_joint_names
        }