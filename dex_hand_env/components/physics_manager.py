"""
Physics manager component for DexHand environment.

This module provides physics simulation management for the DexHand environment,
including physics stepping, synchronization, and auto-detection of physics steps.
"""

# Import IsaacGym first
from isaacgym import gymapi, gymtorch

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

    def __init__(self, gym, sim, device, use_gpu_pipeline=False):
        """
        Initialize the physics manager.

        Args:
            gym: The isaacgym gym instance
            sim: The isaacgym simulation instance
            device: PyTorch device
            use_gpu_pipeline: Whether GPU pipeline is enabled
        """
        self.gym = gym
        self.sim = sim
        self.device = device
        self.use_gpu_pipeline = use_gpu_pipeline

        # Physics tracking variables
        self.physics_step_count = 0
        self.last_control_step_count = 0
        self.physics_steps_per_control_step = 1
        self.auto_detected_physics_steps = False

        # Default physics timestep
        self.physics_dt = 0.01
        self.control_dt = self.physics_dt * self.physics_steps_per_control_step

    def set_dt(self, physics_dt):
        """
        Set the physics timestep.

        Args:
            physics_dt: The physics simulation timestep
        """
        self.physics_dt = physics_dt
        self.control_dt = self.physics_dt * self.physics_steps_per_control_step

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

            # Auto-detect physics_steps_per_control_step
            if not self.auto_detected_physics_steps:
                # Measure distance from last control step
                measured_steps = self.physics_step_count - self.last_control_step_count

                # If we've measured more steps than currently configured, update
                if measured_steps > self.physics_steps_per_control_step:
                    self.physics_steps_per_control_step = measured_steps
                    self.auto_detected_physics_steps = True

                    # Update control_dt to match the actual control frequency
                    self.control_dt = self.physics_dt * self.physics_steps_per_control_step

                    logger.info(f"Auto-detected physics_steps_per_control_step: {measured_steps}. "
                          f"This means {measured_steps} physics steps occur between each policy action.")

            return True
        except Exception as e:
            logger.error(f"Error in step_physics: {e}")
            logger.exception("Traceback:")
            return False

    def apply_tensor_states(self, gym, sim, env_ids, dof_state, root_state_tensor, hand_indices=None):
        """
        Apply tensor states to the physics simulation.

        Args:
            gym: The isaacgym gym instance
            sim: The isaacgym simulation instance
            env_ids: Tensor of environment IDs
            dof_state: Tensor of DOF states
            root_state_tensor: Tensor of root states
            hand_indices: Optional indices of hand actors for each environment

        Returns:
            Boolean indicating success
        """
        try:
            # For DOF state, we use env_ids directly
            env_ids_int32 = env_ids.to(torch.int32)

            # Apply DOF state
            self.gym.set_dof_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(dof_state),
                gymtorch.unwrap_tensor(env_ids_int32),
                len(env_ids)
            )

            # Apply root state - use the right approach based on what we have
            if hand_indices is not None:
                # Convert environment indices to actor indices
                # Create actor indices tensor with the same length as env_ids
                actor_indices = torch.zeros(len(env_ids), dtype=torch.int32, device=self.device)
                for i, env_id in enumerate(env_ids):
                    if env_id < len(hand_indices):
                        actor_indices[i] = hand_indices[env_id]
                    else:
                        logger.warning(f"Environment ID {env_id} out of range for hand_indices")

                # Need to extract the specific actor states from the tensor
                # The root_state_tensor has shape [num_envs, num_bodies, 13]
                # We need to create a tensor with shape [num_bodies, 13] containing just the actors we want to update
                actor_states = torch.zeros((len(env_ids), 13), device=self.device)

                # For each env_id, copy the root state for its corresponding actor
                for i, env_id in enumerate(env_ids):
                    if env_id < root_state_tensor.shape[0] and i < len(actor_indices):
                        actor_idx = actor_indices[i]
                        if actor_idx < root_state_tensor.shape[1]:
                            actor_states[i] = root_state_tensor[env_id, actor_idx]

                # Set actor root state tensor indexed
                self.gym.set_actor_root_state_tensor_indexed(
                    self.sim,
                    gymtorch.unwrap_tensor(actor_states),
                    gymtorch.unwrap_tensor(actor_indices),
                    len(env_ids)
                )
            else:
                # Just try to set the entire root state tensor
                try:
                    self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(root_state_tensor))
                except Exception as e:
                    logger.warning(f"Could not set actor root state tensor: {e}")
                    # Continue anyway, as DOF state is more important

            # Step physics to settle objects
            # This may require multiple physics steps for stability
            success = self.step_physics()
            return success
        except Exception as e:
            logger.critical(f"Error in physics_manager.apply_tensor_states: {e}")
            logger.exception("Traceback:")
            return False

    def mark_control_step(self):
        """
        Mark that a control step has occurred.

        This helps track physics vs control steps for auto-detection.
        """
        self.last_control_step_count = self.physics_step_count

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