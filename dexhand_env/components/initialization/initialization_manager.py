"""
Initialization Manager component for DexHand environment.

This module manages the complex initialization process for DexHand environment components.
It handles the two-stage initialization pattern and coordinates component setup.
"""

from loguru import logger
import torch


class InitializationManager:
    """
    Manages the complex initialization process for DexHand environment components.

    This component handles:
    - Two-stage initialization coordination
    - Component creation and setup
    - Tensor initialization
    - Index mapping creation
    """

    def __init__(self, parent):
        """Initialize InitializationManager with parent environment reference."""
        self.parent = parent
        self.gym = parent.gym
        self.sim = parent.sim

    @property
    def device(self):
        """Access device from parent (single source of truth)."""
        return self.parent.device

    def setup_additional_tensors(self):
        """
        Set up additional tensors needed after component initialization.
        """
        # Create reward buffer
        self.parent.rew_buf = torch.zeros((self.parent.num_envs,), device=self.device)

        # Create reset buffer
        self.parent.reset_buf = torch.zeros(
            (self.parent.num_envs,), device=self.device, dtype=torch.bool
        )

        # Create episode step count buffer
        self.parent.episode_step_count = torch.zeros(
            (self.parent.num_envs,), device=self.device, dtype=torch.long
        )

        # Set up action space
        # action_processor must exist after _init_components()
        # Calculate action space size
        num_actions = 0
        if self.parent.action_processor.policy_controls_hand_base:
            num_actions += self.parent.action_processor.NUM_BASE_DOFS

        if self.parent.action_processor.policy_controls_fingers:
            num_actions += self.parent.action_processor.NUM_ACTIVE_FINGER_DOFS

        # Now set the property once we have the final value
        self.parent.num_actions = num_actions

        # Create the action space
        self.parent.actions = torch.zeros(
            (self.parent.num_envs, self.parent.num_actions), device=self.device
        )

        # Initialize observation encoder now that we know the action space size
        observation_keys = self.parent.task_cfg.get(
            "policy_observation_keys",
            [
                "base_dof_pos",
                "base_dof_vel",
                "finger_dof_pos",
                "finger_dof_vel",
                "hand_pose",
                "contact_forces",
            ],
        )

        # Initialize task states before observation encoder setup
        # This ensures task states are registered when computing observation dimensions
        self.parent.task.initialize_task_states()

        self.parent.observation_encoder.initialize(
            observation_keys=observation_keys,
            joint_to_control=self.parent.hand_initializer.joint_to_control,
            active_joint_names=self.parent.hand_initializer.active_joint_names,
            num_actions=self.parent.num_actions,
            action_processor=self.parent.action_processor,
            index_mappings={
                "base_joint_to_index": self.parent.base_joint_to_index,
                "control_name_to_index": self.parent.control_name_to_index,
                "raw_dof_name_to_index": self.parent.raw_dof_name_to_index,
                "finger_body_to_index": self.parent.finger_body_to_index,
            },
        )

        # Set observation space dimensions needed by VecTask
        self.parent.num_observations = self.parent.observation_encoder.num_observations

        # Now initialize observation and state buffers with correct size
        self.parent.obs_buf = torch.zeros(
            (self.parent.num_envs, self.parent.num_observations),
            device=self.device,
        )
        self.parent.states_buf = torch.zeros(
            (self.parent.num_envs, self.parent.num_observations),
            device=self.device,
        )

        # Create extras dictionary for additional info
        self.parent.extras = {}

        # Initialize reward components to ensure it always exists
        self.parent.last_reward_components = {}

    def create_index_mappings(self):
        """
        Create index mappings for convenient access to tensors by key names.
        """
        logger.debug("Creating index mappings...")

        # 1. Base joint name to index mapping (ARTx, ARTy, etc. -> 0-5)
        self.parent.base_joint_to_index = {}
        for i, joint_name in enumerate(self.parent.base_joint_names):
            self.parent.base_joint_to_index[joint_name] = i

        # 2. Control name to active finger DOF index mapping (th_dip, etc. -> 0-11)
        self.parent.control_name_to_index = {}
        # hand_initializer must exist and have active_joint_names after _init_components()
        for i, control_name in enumerate(
            self.parent.hand_initializer.active_joint_names
        ):
            self.parent.control_name_to_index[control_name] = i

        # 3. Raw finger DOF name to raw DOF tensor index mapping (r_f_joint1_1, etc. -> 0-25)
        self.parent.raw_dof_name_to_index = {}
        # observation_encoder must exist but might not have dof_names yet
        if self.parent.observation_encoder.dof_names:
            for i, dof_name in enumerate(self.parent.observation_encoder.dof_names):
                self.parent.raw_dof_name_to_index[dof_name] = i

        # 4. Finger name + pad/tip to body tensor index mapping
        self.parent.finger_body_to_index = {}

        # Map fingertip body names to indices
        for i, tip_name in enumerate(self.parent.fingertip_body_names):
            finger_name = tip_name.replace(
                "_tip", ""
            )  # e.g., "r_f_link1_tip" -> "r_f_link1"
            self.parent.finger_body_to_index[f"{finger_name}_tip"] = ("fingertip", i)

        # Map fingerpad body names to indices
        for i, pad_name in enumerate(self.parent.fingerpad_body_names):
            finger_name = pad_name.replace(
                "_pad", ""
            )  # e.g., "r_f_link1_pad" -> "r_f_link1"
            self.parent.finger_body_to_index[f"{finger_name}_pad"] = ("fingerpad", i)

        logger.debug("Created mappings:")
        logger.debug(f"  Base joints: {len(self.parent.base_joint_to_index)} entries")
        logger.debug(
            f"  Control names: {len(self.parent.control_name_to_index)} entries"
        )
        logger.debug(
            f"  Raw DOF names: {len(self.parent.raw_dof_name_to_index)} entries"
        )
        logger.debug(
            f"  Finger bodies: {len(self.parent.finger_body_to_index)} entries"
        )
