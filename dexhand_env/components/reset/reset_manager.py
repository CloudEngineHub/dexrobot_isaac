"""
Reset manager component for DexHand environment.

This module provides reset functionality for the DexHand environment,
handling environment resets and DOF state restoration.
"""

# Import standard libraries
import torch
from loguru import logger

# Import IsaacGym
from isaacgym import gymtorch


class ResetManager:
    """
    Manages environment resets for the DexHand environment.

    This component provides functionality to:
    - Reset DOF states to default positions
    - Apply root states for all actors during reset
    - Coordinate with task-specific reset logic
    - Reset action processor targets
    """

    def __init__(
        self,
        parent,
        dof_state,
        root_state_tensor,
        hand_local_actor_index,
        default_dof_pos,
        task,
    ):
        """
        Initialize the reset manager.

        Args:
            parent: Parent DexHandBase instance
            dof_state: DOF state tensor reference
            root_state_tensor: Root state tensor reference
            hand_local_actor_index: Local actor index within each environment (typically 0)
            default_dof_pos: Default DOF positions tensor (required)
            task: Task instance (may have reset_task_state method)
        """
        self.parent = parent
        self.gym = parent.gym
        self.sim = parent.sim

        # Store dependencies
        self.dof_state = dof_state
        self.root_state_tensor = root_state_tensor
        self.hand_local_actor_index = hand_local_actor_index
        self.task = task

        # Store default DOF positions - fail fast if None
        if default_dof_pos is None:
            raise RuntimeError(
                "default_dof_pos is None - ResetManager requires default DOF positions"
            )
        self.default_dof_pos = default_dof_pos

        # Pre-computed global actor indices - will be set when we know num_actors_per_env
        self.all_actor_indices = None

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

    @property
    def episode_step_count(self):
        """Access episode_step_count from parent (single source of truth)."""
        return self.parent.episode_step_count

    def reset_idx(self, env_ids):
        """
        Reset specified environments.

        Args:
            env_ids: Tensor of environment IDs to reset

        Returns:
            Boolean indicating success
        """
        try:
            if len(env_ids) == 0:
                # No environments to reset
                return True

            # Reset episode step count for reset environments
            self.episode_step_count[env_ids] = 0

            # Reset DOF states to default positions
            # Clone to avoid modifying the stored default
            dof_pos = self.default_dof_pos.clone()

            # Set DOF positions for reset environments
            self.dof_state[env_ids, :, 0] = dof_pos

            # Zero DOF velocities
            self.dof_state[env_ids, :, 1] = 0

            # Call task-specific reset function - use duck typing instead of hasattr
            try:
                self.task.reset_task_state(env_ids)
            except AttributeError:
                # Task doesn't implement reset_task_state - this is acceptable
                pass

            # Apply root states for all actors in the reset environments
            # This ensures any task-specific object resets are applied to the simulation
            # We apply root states for ALL actors to maintain architectural cleanliness
            # without requiring knowledge of specific task implementations

            # Initialize pre-computed global actor indices if not done yet
            if self.all_actor_indices is None:
                num_actors_per_env = self.root_state_tensor.shape[1]
                self.all_actor_indices = torch.arange(
                    self.num_envs * num_actors_per_env,
                    device=self.device,
                    dtype=torch.int32,
                ).reshape(self.num_envs, num_actors_per_env)

            # Extract global indices for ALL actors in the environments being reset
            actor_indices_to_reset = self.all_actor_indices[env_ids].flatten()

            # Apply all actor root states at once
            # Use the full tensor - Isaac Gym API expects the full tensor, not sliced data
            # The actor indices tell Isaac Gym which actors to update within the full tensor
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_state_tensor),
                gymtorch.unwrap_tensor(actor_indices_to_reset),
                len(actor_indices_to_reset),
            )

            # Apply DOF states to simulation
            # CRITICAL: Do NOT refresh tensors before applying DOF states!
            # Refreshing would overwrite our reset values with current simulation state
            # Use the local actor index for the hand
            self.physics_manager.apply_dof_states(
                env_ids,
                self.dof_state,
                actor_index=self.hand_local_actor_index,
            )

            # Reset action processor targets to match current DOF positions
            # This ensures that any DOF position changes made by the task
            # have corresponding target updates to prevent PD control conflicts
            current_dof_positions = self.dof_state[env_ids, :, 0]  # Position column
            self.action_processor.reset_targets(env_ids, current_dof_positions)

            # Run a physics step to integrate the DOF changes
            # This is critical to ensure rigid body positions are updated to match DOF states
            self.physics_manager.step_physics(refresh_tensors=True)

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
