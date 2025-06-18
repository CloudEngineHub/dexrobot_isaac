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
        hand_indices,
        task,
        max_episode_length,
    ):
        """
        Initialize the reset manager.

        Args:
            parent: Parent DexHandBase instance
            dof_state: DOF state tensor reference
            root_state_tensor: Root state tensor reference
            hand_indices: Hand actor indices for each environment
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
        self.hand_indices = hand_indices
        self.task = task

        # Reset and progress buffers - will be set by set_buffers()
        self.reset_buf = None
        self.progress_buf = None

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

    def set_buffers(self, reset_buf, progress_buf):
        """
        Set the shared reset and progress buffers.

        Args:
            reset_buf: Shared reset buffer from main environment
            progress_buf: Shared progress buffer from main environment
        """
        self.reset_buf = reset_buf
        self.progress_buf = progress_buf

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
        try:
            # Reset environments that have reached max episode length
            condition = self.progress_buf >= self.max_episode_length - 1

            self.reset_buf = torch.where(
                condition, torch.ones_like(self.reset_buf), self.reset_buf
            )

            # Apply task-specific reset conditions if provided
            if task_reset is not None:
                self.reset_buf = torch.logical_or(self.reset_buf, task_reset)

            # Convert to boolean
            self.reset_buf = self.reset_buf.bool()

            return self.reset_buf
        except Exception as e:
            logger.critical(f"CRITICAL ERROR in reset_manager.check_termination: {e}")
            import traceback

            traceback.print_exc()
            raise

    def increment_progress(self):
        """
        Increment progress buffers.

        Returns:
            Updated progress buffer
        """
        try:
            # Check if progress buffer is set
            if self.progress_buf is None:
                raise RuntimeError("Progress buffer not set. Call set_buffers() first.")

            # Increment progress
            self.progress_buf += 1
            return self.progress_buf
        except Exception as e:
            logger.error(f"ERROR in increment_progress: {e}")
            import traceback

            traceback.print_exc()
            raise

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
            self.progress_buf[env_ids] = 0
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
                # Validate input parameters
                if len(self.hand_indices) != self.num_envs:
                    raise RuntimeError(
                        f"hand_indices length {len(self.hand_indices)} doesn't match num_envs {self.num_envs}"
                    )

                # Check if any env_ids are out of bounds
                max_env_id = env_ids.max().item() if len(env_ids) > 0 else -1
                if max_env_id >= self.num_envs:
                    raise RuntimeError(
                        f"Environment ID {max_env_id} exceeds number of environments {self.num_envs}"
                    )

                # Get hand indices for environments being reset
                hand_indices_to_reset = torch.tensor(
                    [self.hand_indices[env_id] for env_id in env_ids],
                    device=self.device,
                    dtype=torch.long,
                )

                # Validate all hand indices
                if torch.any(hand_indices_to_reset >= self.root_state_tensor.shape[1]):
                    raise RuntimeError(
                        f"One or more hand indices exceed root_state_tensor bodies dimension {self.root_state_tensor.shape[1]}"
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

                # Set positions using advanced indexing
                self.root_state_tensor[
                    env_ids[:, None], hand_indices_to_reset[:, None], 0:3
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

                # Set rotations using advanced indexing
                self.root_state_tensor[
                    env_ids[:, None], hand_indices_to_reset[:, None], 3:7
                ] = rotations

                # Zero velocities - vectorized
                self.root_state_tensor[
                    env_ids[:, None], hand_indices_to_reset[:, None], 7:13
                ] = 0
            else:
                # No hand indices provided, skipping hand pose reset
                pass

            # Call task-specific reset function if provided
            if hasattr(self.task, "reset_task"):
                # Call task reset function
                self.task.reset_task(env_ids)
                # Task reset function completed
            else:
                # No task reset function provided
                pass

            # Apply DOF states to simulation
            # For fixed-base hands, we only need to apply DOF states
            self.physics_manager.apply_dof_states(
                self.gym, self.sim, env_ids, self.dof_state, self.hand_indices
            )

            # Reset action processor targets to avoid jumps after reset
            self.action_processor.reset_targets(env_ids)

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
