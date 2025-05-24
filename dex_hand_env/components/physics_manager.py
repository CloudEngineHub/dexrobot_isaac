"""
Physics manager component for DexHand environment.

This module provides physics simulation management for the DexHand environment,
including physics stepping, synchronization, and auto-detection of physics steps.
"""

# Import IsaacGym first
from isaacgym import gymapi, gymtorch

# Then import PyTorch
import torch


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
    
    def step_physics(self):
        """
        Physics stepping wrapper function.
        
        This function handles physics stepping and auto-detection of how many
        physics steps are needed per control step for stable resets.
        It is used by both the main stepping function and reset_idx.
        
        Returns:
            Boolean indicating whether physics was successfully stepped
        """
        try:
            # Increment physics step counter
            self.physics_step_count += 1
            
            # Simulate physics
            self.gym.simulate(self.sim)
            
            # Fetch results
            # With GPU pipeline, only fetch on CPU (matching IsaacGymEnvs pattern)
            if self.device == 'cpu' or not self.use_gpu_pipeline:
                self.gym.fetch_results(self.sim, True)
            
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
                    
                    print(f"Auto-detected physics_steps_per_control_step: {measured_steps}. "
                          f"This means {measured_steps} physics steps occur between each policy action.")
            
            return True
        except Exception as e:
            print(f"Error in step_physics: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def apply_tensor_states(self, gym, sim, env_ids, dof_state, root_state_tensor):
        """
        Apply tensor states to the physics simulation.
        
        Args:
            gym: The isaacgym gym instance
            sim: The isaacgym simulation instance
            env_ids: Tensor of environment IDs
            dof_state: Tensor of DOF states
            root_state_tensor: Tensor of root states
            
        Returns:
            Boolean indicating success
        """
        try:
            # Apply DOF state
            self.gym.set_dof_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(dof_state),
                gymtorch.unwrap_tensor(env_ids.to(torch.int32)),
                len(env_ids)
            )
            
            # Apply root state
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(root_state_tensor),
                gymtorch.unwrap_tensor(env_ids.to(torch.int32)),
                len(env_ids)
            )
            
            # Step physics to settle objects
            # This may require multiple physics steps for stability
            return self.step_physics()
        except Exception as e:
            print(f"Error in apply_tensor_states: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def mark_control_step(self):
        """
        Mark that a control step has occurred.
        
        This helps track physics vs control steps for auto-detection.
        """
        self.last_control_step_count = self.physics_step_count