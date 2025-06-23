"""
Hand initializer component for DexHand environment.

This module provides hand model initialization functionality for the DexHand environment,
including loading assets, creating actors, and setting up initial states.
"""

# Import standard libraries
import os
from enum import Enum
from typing import List
from loguru import logger

# Import IsaacGym
from isaacgym import gymapi

# Import PyTorch for tensor operations


class HardwareMapping(Enum):
    """Mapping between URDF and hardware joints"""

    th_dip = ("th_dip", ["r_f_joint1_3", "r_f_joint1_4"])
    th_mcp = ("th_mcp", ["r_f_joint1_2"])
    th_rot = ("th_rot", ["r_f_joint1_1"])
    ff_spr = ("ff_spr", ["r_f_joint2_1", "r_f_joint4_1", "r_f_joint5_1"])
    ff_dip = ("ff_dip", ["r_f_joint2_3", "r_f_joint2_4"])
    ff_mcp = ("ff_mcp", ["r_f_joint2_2"])
    mf_dip = ("mf_dip", ["r_f_joint3_3", "r_f_joint3_4"])
    mf_mcp = ("mf_mcp", ["r_f_joint3_2"])
    rf_dip = ("rf_dip", ["r_f_joint4_3", "r_f_joint4_4"])
    rf_mcp = ("rf_mcp", ["r_f_joint4_2"])
    lf_dip = ("lf_dip", ["r_f_joint5_3", "r_f_joint5_4"])
    lf_mcp = ("lf_mcp", ["r_f_joint5_2"])

    def __init__(self, control_name: str, joint_names: List[str]):
        self.control_name = control_name
        self.joint_names = joint_names


class HandInitializer:
    """
    Handles initialization of the robotic hand model in the environment.

    This component provides functionality to:
    - Load hand assets and meshes
    - Create hand instances in each environment
    - Set up initial joint and position states
    - Configure joint properties like limits and stiffness
    """

    def __init__(self, parent, asset_root=None):
        """
        Initialize the hand initializer.

        Args:
            parent: Parent DexHandBase instance
            asset_root: Root directory for assets
        """
        self.parent = parent

        # Access parent properties directly for gym and sim (immutable)
        self.gym = parent.gym
        self.sim = parent.sim

        # Set asset root
        if asset_root is None:
            # Default to assets directory in parent of current file
            current_dir = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            self.asset_root = os.path.join(current_dir, "assets")
        else:
            self.asset_root = asset_root

        # Default hand model path
        self.hand_asset_file = "dexrobot_mujoco/dexrobot_mujoco/models/dexhand021_right_simplified_floating.xml"

        # Define joint names for use in hand creation
        self.base_joint_names = ["ARTx", "ARTy", "ARTz", "ARRx", "ARRy", "ARRz"]

        self.finger_joint_names = [
            "r_f_joint1_1",
            "r_f_joint1_2",
            "r_f_joint1_3",
            "r_f_joint1_4",
            "r_f_joint2_1",
            "r_f_joint2_2",
            "r_f_joint2_3",
            "r_f_joint2_4",
            "r_f_joint3_1",
            "r_f_joint3_2",
            "r_f_joint3_3",
            "r_f_joint3_4",
            "r_f_joint4_1",
            "r_f_joint4_2",
            "r_f_joint4_3",
            "r_f_joint4_4",
            "r_f_joint5_1",
            "r_f_joint5_2",
            "r_f_joint5_3",
            "r_f_joint5_4",
        ]

        # Use the authoritative hardware mapping
        self.active_joint_mapping = {
            member.control_name: member.joint_names for member in HardwareMapping
        }

        # Create reverse mapping from joint name to controller
        self.joint_to_control = {}
        for control, joints in self.active_joint_mapping.items():
            for joint in joints:
                self.joint_to_control[joint] = control

        # Active joint names (12 controls mapping to finger DOFs with coupling)
        self.active_joint_names = list(self.active_joint_mapping.keys())

        # Body names for fingertips and fingerpads in the MJCF model
        self.fingertip_body_names = [
            "r_f_link1_tip",
            "r_f_link2_tip",
            "r_f_link3_tip",
            "r_f_link4_tip",
            "r_f_link5_tip",
        ]

        self.fingerpad_body_names = [
            "r_f_link1_pad",
            "r_f_link2_pad",
            "r_f_link3_pad",
            "r_f_link4_pad",
            "r_f_link5_pad",
        ]

        # Contact force body names will be set from config during initialize()
        self.contact_force_body_names = []

        # Storage for handles and indices
        self.hand_handles = []
        self.fingertip_body_handles = []
        self.fingerpad_body_handles = []
        self.hand_actor_indices = []  # Actor indices for DOF operations
        self.hand_rigid_body_indices = (
            []
        )  # Temporary storage for verification - must be identical across envs
        self.fingertip_indices = []
        self.fingerpad_indices = []
        self.contact_force_body_indices = []  # Indices for contact force sensing bodies

        # Optimized single index (after verification that all envs have same index)
        self.hand_rigid_body_index = None  # Single scalar index (same across envs)

        # Rigid body index to name mapping for debug logging
        self.rigid_body_index_to_name = {}  # Maps rigid body index -> body name

        # DOF names - will be populated during asset loading
        self._dof_names = None

        # Initial pose settings
        self.initial_hand_pos = [0.0, 0.0, 0.5]
        self.initial_hand_rot = [0.0, 0.0, 0.0, 1.0]

        # Joint control settings removed - now using MJCF values directly

    @property
    def num_envs(self):
        """Access num_envs from parent (single source of truth)."""
        return self.parent.num_envs

    @property
    def device(self):
        """Access device from parent (single source of truth)."""
        return self.parent.device

    @property
    def dof_names(self):
        """Access DOF names (single source of truth)."""
        if self._dof_names is None:
            raise RuntimeError(
                "DOF names not available. load_hand_asset() must be called first."
            )
        return self._dof_names

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

    def set_contact_force_bodies(self, body_names):
        """
        Set the contact force body names from config.

        Args:
            body_names: List of body names to monitor for contact forces
        """
        self.contact_force_body_names = body_names

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
        asset_options.fix_base_link = True
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

            # Debug: Check DOF count in the asset
            dof_count = self.gym.get_asset_dof_count(hand_asset)
            logger.debug(f"Asset DOF count: {dof_count}")

            # Debug: Get asset DOF names to see what Isaac Gym sees in the asset
            asset_dof_names = self.gym.get_asset_dof_names(hand_asset)
            logger.debug(f"Asset DOF names ({len(asset_dof_names)}): {asset_dof_names}")

            # Store DOF names as single source of truth
            self._dof_names = asset_dof_names

            return hand_asset
        except Exception as e:
            logger.error(f"Error loading hand asset: {e}")
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
        self.hand_actor_indices = []  # Actor indices for DOF operations
        self.hand_rigid_body_indices = []  # Rigid body indices
        self.fingertip_indices = []
        self.fingerpad_indices = []

        # Note: We'll get DOF properties from the first actor after creation
        # to avoid issues with GPU pipeline
        self.original_dof_props = None

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

            # Get actor index (for DOF operations)
            actor_idx = self.gym.get_actor_index(env, hand_handle, gymapi.DOMAIN_SIM)
            self.hand_actor_indices.append(actor_idx)

            # Get DOF properties from actor (not asset) for GPU pipeline compatibility
            hand_dof_props = self.gym.get_actor_dof_properties(env, hand_handle)

            # Configure DOF properties for PD control
            dof_names = self.gym.get_actor_dof_names(env, hand_handle)

            # Save a copy of the original DOF properties for tensor manager (from first actor)
            if i == 0:
                self.original_dof_props = hand_dof_props.copy()
                # Debug: Print DOF limits for first 6 joints
                logger.debug("===== BASE DOF LIMITS FROM ACTOR =====")
                for j in range(min(6, len(dof_names))):
                    logger.debug(
                        f"DOF {j} ({dof_names[j]}): lower={hand_dof_props['lower'][j]:.6f}, upper={hand_dof_props['upper'][j]:.6f}"
                    )
                logger.debug("=====================================")

            # Log DOF names for the first actor to verify all 25 DOFs
            if i == 0:
                logger.debug("===== DOF NAMES VERIFICATION =====")
                logger.debug(f"Total DOFs found: {len(dof_names)}")
                logger.debug("DOF Index -> Joint Name:")
                for j, name in enumerate(dof_names):
                    # Determine joint type for classification
                    joint_type = "UNKNOWN"
                    if any(base_name in name for base_name in self.base_joint_names):
                        joint_type = "BASE"
                    elif any(
                        finger_name in name for finger_name in self.finger_joint_names
                    ):
                        joint_type = "FINGER"

                    logger.debug(f"  {j:2d}: {name:<20} ({joint_type})")
                logger.debug("=====================================")

            for j, name in enumerate(dof_names):
                # Set drive mode (keep using position control)
                hand_dof_props["driveMode"][j] = gymapi.DOF_MODE_POS
                # Note: stiffness and damping now come directly from MJCF model

            # Set DOF properties
            self.gym.set_actor_dof_properties(env, hand_handle, hand_dof_props)

            # Store handle (indices will be acquired after all actors are created)
            self.hand_handles.append(hand_handle)

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

            # Indices will be acquired after all actors are created

        return {
            "hand_handles": self.hand_handles,
            "fingertip_body_handles": self.fingertip_body_handles,
            "fingerpad_body_handles": self.fingerpad_body_handles,
            "dof_properties": self.original_dof_props,  # Add DOF properties to return value
            "hand_actor_indices": self.hand_actor_indices,  # Actor indices for DOF operations
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
            "active_joint_names": self.active_joint_names,
        }

    def initialize_rigid_body_indices(self, envs):
        """
        Initialize rigid body indices after all actors have been created.

        This method should be called once after all actors (including task-specific
        actors) have been added to the simulation. Isaac Gym uses global indices
        that depend on the total number of actors, so this must happen after
        environment setup is complete. These indices are immutable once initialized.

        Args:
            envs: List of environment instances
        """
        from loguru import logger

        logger.debug("Initializing rigid body indices after all actor creation...")

        # Ensure we only initialize once
        if self.hand_rigid_body_indices:
            raise RuntimeError(
                "Rigid body indices have already been initialized. They should only be initialized once."
            )

        # Build rigid body index to name mapping (only for first environment since all are identical)
        if envs and self.hand_handles:
            # Get all rigid body names for the hand
            rigid_body_names = self.gym.get_actor_rigid_body_names(
                envs[0], self.hand_handles[0]
            )

            # Build the mapping for all hand rigid bodies
            for body_name in rigid_body_names:
                body_idx = self.gym.find_actor_rigid_body_index(
                    envs[0], self.hand_handles[0], body_name, gymapi.DOMAIN_SIM
                )
                self.rigid_body_index_to_name[body_idx] = body_name

            logger.debug(
                f"Built rigid body index mapping with {len(self.rigid_body_index_to_name)} bodies"
            )

        # Acquire indices for each environment
        for i, (env, hand_handle) in enumerate(zip(envs, self.hand_handles)):
            # Get hand base rigid body index
            hand_base_idx = self.gym.find_actor_rigid_body_index(
                env, hand_handle, "right_hand_base", gymapi.DOMAIN_SIM
            )
            self.hand_rigid_body_indices.append(hand_base_idx)

            # Get fingertip indices
            fingertip_indices = []
            for name in self.fingertip_body_names:
                body_idx = self.gym.find_actor_rigid_body_index(
                    env, hand_handle, name, gymapi.DOMAIN_SIM
                )
                fingertip_indices.append(body_idx)
            self.fingertip_indices.append(fingertip_indices)

            # Get fingerpad indices
            fingerpad_indices = []
            for name in self.fingerpad_body_names:
                body_idx = self.gym.find_actor_rigid_body_index(
                    env, hand_handle, name, gymapi.DOMAIN_SIM
                )
                fingerpad_indices.append(body_idx)
            self.fingerpad_indices.append(fingerpad_indices)

            # Get contact force body indices
            contact_force_indices = []
            for name in self.contact_force_body_names:
                body_idx = self.gym.find_actor_rigid_body_index(
                    env, hand_handle, name, gymapi.DOMAIN_SIM
                )
                contact_force_indices.append(body_idx)
            self.contact_force_body_indices.append(contact_force_indices)

        # Convert global indices to local index
        # Since all environments have identical structure, the local index within each environment
        # is the same. We use the first environment's global index as the local index.
        if not self.hand_rigid_body_indices:
            raise RuntimeError(
                "No hand rigid body indices found. This indicates a critical initialization error."
            )

        # The local index is simply the first environment's global index
        # This works because environment 0 starts at global index 0
        self.hand_rigid_body_index = self.hand_rigid_body_indices[0]

        logger.debug(
            f"Using hand rigid body index {self.hand_rigid_body_index} (from env 0 global index)"
        )
        logger.debug(f"All environment global indices: {self.hand_rigid_body_indices}")

        logger.debug(f"Initialized rigid body indices for {len(envs)} environments")
        logger.debug(f"Hand rigid body index: {self.hand_rigid_body_index}")
        logger.debug(f"Hand actor indices: {self.hand_actor_indices}")
        logger.debug(
            f"First env fingertip indices: {self.fingertip_indices[0] if self.fingertip_indices else 'None'}"
        )

        # Return rigid body indices as constants since all environments must be identical
        return {
            "hand_rigid_body_index": self.hand_rigid_body_index,  # Single index (constant)
            "hand_actor_indices": self.hand_actor_indices,  # Actor indices for DOF operations
            "fingertip_indices": self.fingertip_indices,
            "fingerpad_indices": self.fingerpad_indices,
            "contact_force_body_indices": self.contact_force_body_indices,  # Indices for force sensing
        }
