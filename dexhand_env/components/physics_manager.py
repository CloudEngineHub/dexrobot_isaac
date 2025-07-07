"""
Physics manager component for DexHand environment.

This module provides physics simulation management for the DexHand environment,
including physics stepping, synchronization, and auto-detection of physics steps.
"""

# Import IsaacGym first
from isaacgym import gymtorch

# Then import PyTorch
import torch

# Import loguru
from loguru import logger


class PhysicsManager:
    """
    Manages physics simulation for the DexHand environment.

    This component provides functionality to:
    - Step physics simulation uniformly
    - Auto-detect required physics steps for stable resets
    - Handle GPU pipeline specific requirements
    - Apply tensor states to simulation
    """

    def __init__(self, parent, physics_dt):
        """
        Initialize the physics manager.

        Args:
            parent: Parent DexHandBase instance
            physics_dt: Physics timestep from config
        """
        self.parent = parent
        self.gym = parent.gym
        self.sim = parent.sim
        self.physics_dt = physics_dt

        # Physics tracking variables
        self.physics_step_count = 0
        self.last_control_step_count = 0
        self.physics_steps_per_control_step = 1
        self.auto_detected_physics_steps = False
        self.measuring_control_cycle = False

        # control_dt will be set by measurement
        self.control_dt = None

    @property
    def device(self):
        """Access device from parent (single source of truth)."""
        return self.parent.device

    @property
    def tensor_manager(self):
        """Access tensor_manager from parent (single source of truth)."""
        return self.parent.tensor_manager

    @property
    def num_actors_per_env(self):
        """Get number of actors per environment dynamically from tensor shape."""
        if self.tensor_manager is None:
            raise RuntimeError("tensor_manager is None - initialization failed")
        if self.tensor_manager.actor_root_state_tensor is None:
            raise RuntimeError(
                "actor_root_state_tensor is None - tensor setup not complete"
            )
        return self.tensor_manager.actor_root_state_tensor.shape[1]

    def step_physics(self, refresh_tensors=True):
        """
        Physics stepping wrapper function.

        This function handles physics stepping and auto-detection of how many
        physics steps are needed per control step for stable resets.
        It is used by both the main stepping function and reset_idx.

        Args:
            refresh_tensors: Whether to refresh tensor data after stepping

        Returns:
            Boolean indicating whether physics was successfully stepped
        """
        try:
            # Increment physics step counter
            self.physics_step_count += 1

            # Simulate physics
            self.gym.simulate(self.sim)

            # Step graphics - required for proper physics simulation even in headless mode
            self.gym.step_graphics(self.sim)

            # Fetch results - ALWAYS call this to update tensor data
            # This is crucial for headless mode to work properly
            self.gym.fetch_results(self.sim, True)

            # Always refresh tensors after physics step to ensure data is current
            # This is especially important for headless mode
            if refresh_tensors:
                # Refresh all the simulation tensors to ensure data is current
                try:
                    self.gym.refresh_dof_state_tensor(self.sim)
                    self.gym.refresh_actor_root_state_tensor(self.sim)
                    self.gym.refresh_rigid_body_state_tensor(self.sim)
                    self.gym.refresh_net_contact_force_tensor(self.sim)
                except Exception as refresh_err:
                    logger.warning(f"Error refreshing tensors: {refresh_err}")

            # No passive auto-detection needed - we use active measurement

            return True
        except Exception as e:
            logger.error(f"Error in step_physics: {e}")
            logger.exception("Traceback:")
            return False

    def apply_dof_states(self, gym, sim, env_ids, dof_state, actor_index=0):
        """
        Apply DOF states to specified actor in given environments.

        Args:
            gym: The isaacgym gym instance
            sim: The isaacgym simulation instance
            env_ids: Tensor of environment IDs to update
            dof_state: Tensor of DOF states [num_envs, num_dofs, 2]
            actor_index: Local actor index within each environment (default: 0)

        Returns:
            Boolean indicating success
        """
        try:
            # Compute global actor indices
            # For environment i with local actor j, global index = i * num_actors_per_env + j
            global_actor_indices = env_ids * self.num_actors_per_env + actor_index

            # Ensure indices are int32 for Isaac Gym
            global_actor_indices = global_actor_indices.to(torch.int32)

            # Pass the full DOF state tensor - Isaac Gym uses indices to update only specified actors
            # Based on Isaac Gym examples (e.g., ant.py), indexed operations expect the full tensor
            dof_states_to_apply = dof_state.reshape(-1, 2)

            # Apply DOF states using Isaac Gym API
            self.gym.set_dof_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(dof_states_to_apply),
                gymtorch.unwrap_tensor(global_actor_indices),
                len(env_ids),
            )

            # Refresh tensors to make changes visible
            self.refresh_tensors()

            return True
        except Exception as e:
            logger.critical(f"Error in physics_manager.apply_dof_states: {e}")
            logger.exception("Traceback:")
            return False

    def apply_root_states(self, gym, sim, env_ids, root_state_tensor, actor_indices):
        """
        Apply root states to specified actors.

        Args:
            gym: The isaacgym gym instance
            sim: The isaacgym simulation instance
            env_ids: Tensor of environment IDs
            root_state_tensor: Tensor of root states [num_envs, num_bodies, 13]
            actor_indices: Tensor of actor indices for each environment

        Returns:
            Boolean indicating success
        """
        try:
            # Vectorized extraction of actor indices
            # Check bounds once
            max_env_id = env_ids.max() if len(env_ids) > 0 else -1
            if max_env_id >= len(actor_indices):
                raise RuntimeError(
                    f"Environment ID {max_env_id} out of range for actor_indices (size: {len(actor_indices)})"
                )

            # Use advanced indexing to extract indices
            indices_to_use = actor_indices[env_ids].to(torch.int32)

            # Validate tensor bounds
            if max_env_id >= root_state_tensor.shape[0]:
                raise RuntimeError(
                    f"Environment ID {max_env_id} out of range for root_state_tensor (shape: {root_state_tensor.shape})"
                )

            max_actor_idx = indices_to_use.max() if len(indices_to_use) > 0 else -1
            if max_actor_idx >= root_state_tensor.shape[1]:
                raise RuntimeError(
                    f"Actor index {max_actor_idx} out of range for root_state_tensor (shape: {root_state_tensor.shape})"
                )

            # Apply root states using the full tensor
            # Isaac Gym API expects the full tensor, not sliced data
            # The actor indices tell Isaac Gym which actors to update within the full tensor
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(root_state_tensor),
                gymtorch.unwrap_tensor(indices_to_use),
                len(indices_to_use),
            )

            # Refresh tensors to make changes visible
            self.refresh_tensors()

            return True
        except Exception as e:
            logger.critical(f"Error in physics_manager.apply_root_states: {e}")
            logger.exception("Traceback:")
            return False

    def mark_control_step(self):
        """
        Mark that a control step has occurred.

        This helps track physics vs control steps for auto-detection.
        """
        logger.debug(
            f"Marking control step: last_control_step_count {self.last_control_step_count} -> {self.physics_step_count}"
        )
        self.last_control_step_count = self.physics_step_count

    def start_control_cycle_measurement(self):
        """
        Start measuring physics steps for auto-detection.

        This should be called at the beginning of the first control cycle
        to accurately measure how many physics steps occur including resets.
        """
        if not self.auto_detected_physics_steps and self.control_dt is None:
            logger.info("Starting control cycle measurement for auto-detection")
            self.measuring_control_cycle = True
            self.physics_step_count = 0
            self.last_control_step_count = 0
            return True
        return False

    def finish_control_cycle_measurement(self):
        """
        Finish measuring and set control_dt based on measured steps.
        """
        if self.measuring_control_cycle:
            measured_steps = self.physics_step_count
            logger.info(
                f"Control cycle measurement complete: {measured_steps} physics steps detected"
            )

            # Update physics steps per control
            self.physics_steps_per_control_step = measured_steps
            self.auto_detected_physics_steps = True
            self.measuring_control_cycle = False

            # Set control_dt for the first time
            self.control_dt = self.physics_dt * self.physics_steps_per_control_step

            logger.info(
                f"Auto-detected physics_steps_per_control_step: {measured_steps}. "
                f"control_dt = {self.control_dt:.6f}s"
            )

    def refresh_tensors(self):
        """
        Refresh all tensor data from the simulation.

        This ensures that the tensor data is up-to-date with the current state
        of the physics simulation. This is especially important when using the GPU pipeline.

        Returns:
            Boolean indicating success
        """
        try:
            # These calls ensure the tensor data is updated from the physics simulation
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)
            return True
        except Exception as e:
            logger.error(f"Error refreshing tensors: {e}")
            logger.exception("Traceback:")
            return False
