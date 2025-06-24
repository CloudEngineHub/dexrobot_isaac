"""
Reset manager component for DexHand environment.

This module provides reset functionality for the DexHand environment,
including environment resets, randomization, and initialization.
"""

# Import standard libraries
import torch
from loguru import logger

# Import IsaacGym


class ResetManager:
    """
    Manages environment resets for the DexHand environment.

    This component provides functionality to:
    - Handle environment resets based on episode termination
    - Manage task-specific reset conditions
    - Track episode progress
    - Apply randomization during resets
    """

    def __init__(
        self,
        parent,
        dof_state,
        root_state_tensor,
        hand_local_actor_index,
        hand_local_rigid_body_index,
        task,
        max_episode_length,
    ):
        """
        Initialize the reset manager.

        Args:
            parent: Parent DexHandBase instance
            dof_state: DOF state tensor reference
            root_state_tensor: Root state tensor reference
            hand_local_actor_index: Local actor index within each environment (typically 0)
            hand_local_rigid_body_index: Local rigid body index for hand base
            task: Task instance (may have reset_task method)
            max_episode_length: Maximum episode length from config
        """
        self.parent = parent
        self.gym = parent.gym
        self.sim = parent.sim
        self.max_episode_length = max_episode_length

        # Store dependencies
        self.dof_state = dof_state
        self.root_state_tensor = root_state_tensor
        self.hand_local_actor_index = hand_local_actor_index
        self.hand_local_rigid_body_index = hand_local_rigid_body_index
        self.task = task

        # Episode step count buffer - will be set by set_episode_step_count_buffer()
        self.episode_step_count = None

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
        self.default_hand_pos = torch.tensor([0.0, 0.0, 0.5], device=self.device)
        self.default_hand_rot = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)

    @property
    def num_envs(self):
        """Access num_envs from parent (single source of truth)."""
        return self.parent.num_envs

    @property
    def device(self):
        """Access device from parent (single source of truth)."""
        return self.parent.device

    @property
    def physics_manager(self):
        """Access physics_manager from parent (single source of truth)."""
        return self.parent.physics_manager

    @property
    def action_processor(self):
        """Access action_processor from parent (single source of truth)."""
        return self.parent.action_processor

    def set_episode_length(self, max_episode_length):
        """
        Set the maximum episode length.

        Args:
            max_episode_length: Maximum episode length
        """
        self.max_episode_length = max_episode_length

    def set_randomization(
        self,
        randomize_positions=False,
        randomize_orientations=False,
        randomize_dofs=False,
        position_range=None,
        orientation_range=None,
        dof_range=None,
    ):
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

    def set_episode_step_count_buffer(self, episode_step_count):
        """
        Set the shared episode step count buffer.

        Args:
            episode_step_count: Shared episode step count buffer from main environment
        """
        self.episode_step_count = episode_step_count

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

    def reset_idx(self, env_ids):
        """
        Reset specified environments.

        Args:
            env_ids: Tensor of environment IDs to reset

        Returns:
            Boolean indicating success
        """
        try:
            # Removed excessive DEBUG logging - env_ids already validated
            if len(env_ids) == 0:
                # No environments to reset
                return True

            # No need to validate required parameters - they are now non-optional

            # Removed excessive DEBUG logging - tensor shapes and devices

            # Reset progress buffer for reset environments
            # Reset progress buffer
            self.episode_step_count[env_ids] = 0
            # Progress buffer reset

            # Reset DOF states
            if self.default_dof_pos is not None:
                # Reset DOF states
                # Use default DOF positions
                dof_pos = self.default_dof_pos.clone()
                # Default DOF positions initialized

                # Apply randomization if enabled
                if (
                    self.randomize_dof_positions
                    and self.dof_position_randomization_range > 0
                ):
                    # Apply DOF randomization
                    dof_pos = (
                        dof_pos
                        + torch.rand(dof_pos.shape, device=self.device)
                        * self.dof_position_randomization_range
                        - self.dof_position_randomization_range / 2
                    )

                # Set DOF positions for reset environments
                # Set DOF positions
                self.dof_state[env_ids, :, 0] = dof_pos

                # Zero DOF velocities
                # Zero DOF velocities
                self.dof_state[env_ids, :, 1] = 0
                # DOF states reset

                # Debug: Log what we're setting
                logger.debug(
                    f"Reset DOF positions to: {dof_pos[:6]}"
                )  # Show first 6 DOFs

            # Reset hand pose in root state tensor
            # For fixed-base hands, we don't need to reset root state
            # The hand position is controlled by DOFs, not by root state
            # Skip this section for fixed-base configuration
            if False:  # Disabled for fixed-base hands
                # Vectorized root state reset for future use
                # Check if any env_ids are out of bounds
                max_env_id = env_ids.max().item() if len(env_ids) > 0 else -1
                if max_env_id >= self.num_envs:
                    raise RuntimeError(
                        f"Environment ID {max_env_id} exceeds number of environments {self.num_envs}"
                    )

                # All environments have the same local rigid body index
                # The root_state_tensor is shaped (num_envs, num_bodies_per_env, 13)
                # so we can use the local index directly
                if self.hand_local_rigid_body_index >= self.root_state_tensor.shape[1]:
                    raise RuntimeError(
                        f"Hand local rigid body index {self.hand_local_rigid_body_index} exceeds root_state_tensor bodies dimension {self.root_state_tensor.shape[1]}"
                    )

                # Set positions - vectorized
                # Start with default positions expanded for all environments being reset
                positions = (
                    self.default_hand_pos.unsqueeze(0).expand(len(env_ids), -1).clone()
                )

                # Apply position randomization if enabled
                if self.randomize_initial_positions:
                    # Generate random offsets for all environments at once
                    random_offsets = (
                        torch.rand((len(env_ids), 3), device=self.device) * 2 - 1
                    )
                    # Scale by randomization range
                    position_range = torch.tensor(
                        self.position_randomization_range, device=self.device
                    )
                    random_offsets *= position_range
                    positions += random_offsets

                # Set positions using local index
                self.root_state_tensor[
                    env_ids, self.hand_local_rigid_body_index, 0:3
                ] = positions

                # Set rotations - vectorized
                # Start with default rotations
                rotations = (
                    self.default_hand_rot.unsqueeze(0).expand(len(env_ids), -1).clone()
                )

                # Apply orientation randomization if enabled
                if (
                    self.randomize_initial_orientations
                    and self.orientation_randomization_range > 0
                ):
                    # Generate random angles for all environments
                    rand_angles = (
                        torch.rand(len(env_ids), device=self.device) * 2 - 1
                    ) * self.orientation_randomization_range
                    half_angles = rand_angles / 2

                    # Create quaternions for z-axis rotation [x, y, z, w]
                    rotations = torch.zeros((len(env_ids), 4), device=self.device)
                    rotations[:, 2] = torch.sin(half_angles)  # z component
                    rotations[:, 3] = torch.cos(half_angles)  # w component

                # Set rotations using local index
                self.root_state_tensor[
                    env_ids, self.hand_local_rigid_body_index, 3:7
                ] = rotations

                # Zero velocities using local index
                self.root_state_tensor[
                    env_ids, self.hand_local_rigid_body_index, 7:13
                ] = 0
            else:
                # No hand indices provided, skipping hand pose reset
                pass

            # Call task-specific reset function - use duck typing instead of hasattr
            try:
                self.task.reset_task(env_ids)
            except AttributeError:
                # Task doesn't implement reset_task - this is acceptable
                pass

            # Apply DOF states to simulation
            # Use the local actor index for the hand
            self.physics_manager.apply_dof_states(
                self.gym,
                self.sim,
                env_ids,
                self.dof_state,
                actor_index=self.hand_local_actor_index,
            )

            # Reset action processor targets to avoid jumps after reset
            self.action_processor.reset_targets(env_ids)

            # Note: reset_buf is now managed by the calling code (BaseTask)

            # Reset completed successfully
            return True

        except Exception as e:
            logger.critical(f"CRITICAL ERROR in reset_manager.reset_idx: {e}")
            import traceback

            traceback.print_exc()
            raise

    def reset_all(self):
        """
        Reset all environments.

        Returns:
            Boolean indicating success
        """
        env_ids = torch.arange(self.num_envs, device=self.device)
        return self.reset_idx(env_ids)
