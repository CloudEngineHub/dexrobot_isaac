#!/usr/bin/env python
"""
DexHand test script with GPU pipeline enabled.

This script is based on the successful gpu_pipeline_debug.py approach,
but extended to include the functionality from dexhand_test.py.
"""

import os
import sys
import time
import argparse
import yaml
import traceback

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import IsaacGym first
from isaacgym import gymapi, gymtorch

# Then import PyTorch
import torch

def load_config(config_path=None):
    """Load config from YAML file or use default path."""
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "dex_hand_env/cfg/task/BaseTask.yaml"
        )
    
    print(f"Loading config from {config_path}")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        cfg_yaml = yaml.safe_load(f)
    
    # Convert to the expected structure
    cfg = {
        "env": cfg_yaml.get("env", {}),
        "sim": cfg_yaml.get("sim", {}),
        "task": cfg_yaml.get("task", {}),
        "physics_engine": "physx"
    }
    
    # Add required sim parameters that are not in YAML
    if "gravity" not in cfg["sim"]:
        cfg["sim"]["gravity"] = [0.0, 0.0, -9.81]
    
    if "up_axis" not in cfg["sim"]:
        cfg["sim"]["up_axis"] = "z"
    
    return cfg

def main():
    """Main function to test DexHand with GPU pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test DexHand with GPU pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML file")
    parser.add_argument("--no-gpu-pipeline", action="store_true", help="Disable GPU pipeline")
    parser.add_argument("--steps", type=int, default=300, help="Number of steps to run")
    parser.add_argument("--movement-speed", type=float, default=0.05, help="Speed of DOF movement")
    parser.add_argument("--sleep", type=float, default=0.01, help="Sleep time between steps")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--headless", action="store_true", help="Run without visualization")
    args = parser.parse_args()

    print("Starting DexHand GPU test script...")

    # Load configuration
    cfg = load_config(args.config)
    
    # Configure GPU pipeline
    cfg["sim"]["use_gpu_pipeline"] = not args.no_gpu_pipeline
    print(f"GPU pipeline enabled: {cfg['sim']['use_gpu_pipeline']}")
    
    # Ensure we only create one environment for stability
    cfg["env"]["numEnvs"] = 1
    
    # Set up the simulation parameters explicitly
    sim_params = gymapi.SimParams()
    
    # Set physics parameters from config
    sim_params.dt = cfg["sim"].get("dt", 0.01)
    sim_params.substeps = cfg["sim"].get("substeps", 2)
    sim_params.up_axis = gymapi.UP_AXIS_Z if cfg["sim"].get("up_axis", "z") == "z" else gymapi.UP_AXIS_Y
    sim_params.gravity = gymapi.Vec3(*cfg["sim"]["gravity"])
    
    # Set GPU pipeline
    sim_params.use_gpu_pipeline = cfg["sim"]["use_gpu_pipeline"]
    
    # Configure PhysX-specific parameters critical for GPU pipeline
    sim_params.physx.use_gpu = True
    sim_params.physx.solver_type = 1  # TGS
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 2
    sim_params.physx.contact_offset = 0.02
    sim_params.physx.rest_offset = 0.001
    
    # These parameters are critical for GPU pipeline stability
    if cfg["sim"]["use_gpu_pipeline"]:
        sim_params.physx.contact_collection = gymapi.ContactCollection.CC_LAST_SUBSTEP
        sim_params.physx.default_buffer_size_multiplier = 1.0
        sim_params.physx.max_gpu_contact_pairs = 1024 * 16
        sim_params.physx.always_use_articulations = True
    
    # Create the gym
    gym = gymapi.acquire_gym()
    
    # Create the simulation
    print("Creating simulation...")
    sim_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    sim_device_id = 0
    graphics_device_id = -1 if args.headless else 0
    
    try:
        sim = gym.create_sim(sim_device_id, graphics_device_id, gymapi.SIM_PHYSX, sim_params)
        if sim is None:
            print("*** Failed to create sim")
            return
        print("Simulation created successfully")
    except Exception as e:
        print(f"Error creating simulation: {e}")
        traceback.print_exc()
        return
    
    # Create a ground plane
    print("Creating ground plane...")
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)
    
    # Create a viewer if not headless
    viewer = None
    if not args.headless:
        print("Creating viewer...")
        try:
            # Create camera with better properties
            camera_props = gymapi.CameraProperties()
            camera_props.horizontal_fov = 75.0
            camera_props.width = 1280
            camera_props.height = 720
            
            viewer = gym.create_viewer(sim, camera_props)
            if viewer is None:
                print("*** Failed to create viewer")
                return
            print("Viewer created successfully")
            
            # Set initial camera position
            cam_pos = gymapi.Vec3(1.0, 0.5, 0.8)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
            gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
        except Exception as e:
            print(f"Error creating viewer: {e}")
            traceback.print_exc()
            viewer = None
    
    # Load the hand asset with minimal options
    print("Loading hand asset...")
    asset_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
    hand_asset_file = "dexrobot_mujoco/dexrobot_mujoco/models/dexhand021_right_simplified_floating.xml"
    
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    
    try:
        asset = gym.load_asset(sim, asset_root, hand_asset_file, asset_options)
        if asset is None:
            print("*** Failed to load asset")
            return
        print("Asset loaded successfully")
        
        # Print DOF information
        dof_count = gym.get_asset_dof_count(asset)
        print(f"Asset has {dof_count} DOFs")
    except Exception as e:
        print(f"Error loading asset: {e}")
        traceback.print_exc()
        return
    
    # Create an environment
    print("Creating environment...")
    env_spacing = 1.0
    env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
    env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
    
    try:
        env = gym.create_env(sim, env_lower, env_upper, 1)
        if env is None:
            print("*** Failed to create environment")
            return
        print("Environment created successfully")
    except Exception as e:
        print(f"Error creating environment: {e}")
        traceback.print_exc()
        return
    
    # Add the hand actor to the environment
    print("Adding hand actor to environment...")
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 0.5)
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    
    try:
        actor_handle = gym.create_actor(env, asset, pose, "hand", 0, 0)
        if actor_handle == -1:
            print("*** Failed to create actor")
            return
        print("Actor created successfully with handle", actor_handle)
    except Exception as e:
        print(f"Error creating actor: {e}")
        traceback.print_exc()
        return
    
    # Set DOF properties with high stiffness for stability
    print("Setting DOF properties...")
    try:
        dof_props = gym.get_actor_dof_properties(env, actor_handle)
        for i in range(dof_count):
            dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            if i < 3:  # base translation DOFs
                dof_props["stiffness"][i] = 5000.0
                dof_props["damping"][i] = 200.0
                if i == 2:  # vertical axis
                    dof_props["stiffness"][i] = 10000.0
                    dof_props["damping"][i] = 500.0
            elif i < 6:  # base rotation DOFs
                dof_props["stiffness"][i] = 1000.0
                dof_props["damping"][i] = 100.0
            else:  # finger DOFs
                dof_props["stiffness"][i] = 100.0
                dof_props["damping"][i] = 10.0
                
        gym.set_actor_dof_properties(env, actor_handle, dof_props)
        print("DOF properties set successfully")
    except Exception as e:
        print(f"Error setting DOF properties: {e}")
        traceback.print_exc()
    
    # Get DOF state tensor
    dof_state_tensor = gym.acquire_dof_state_tensor(sim)
    dof_state = gymtorch.wrap_tensor(dof_state_tensor)
    dof_state_shape = dof_state.shape
    print(f"DOF state tensor shape: {dof_state_shape}")
    
    # Reshape DOF state tensor
    dof_pos = dof_state.view(1, dof_count, 2)[..., 0]
    dof_vel = dof_state.view(1, dof_count, 2)[..., 1]
    
    # Get joint info
    num_base_dofs = 6
    
    # Initialize target positions
    targets = torch.zeros_like(dof_pos)
    
    # Prepare the simulation
    print("Preparing simulation...")
    try:
        gym.prepare_sim(sim)
        print("Simulation prepared successfully")
    except Exception as e:
        print(f"Error preparing simulation: {e}")
        traceback.print_exc()
        return
    
    # Define base joint names
    base_joint_names = ["ARTx", "ARTy", "ARTz", "ARRx", "ARRy", "ARRz"]
    
    # Run the simulation loop
    print(f"Running simulation for {args.steps} steps...")
    
    try:
        for i in range(args.steps):
            if args.debug and i % 10 == 0:
                print(f"Step {i+1}/{args.steps}")
            
            # Calculate which DOF to move this step
            dof_index = i % dof_count
            
            # Reset targets to current position
            targets[:] = dof_pos
            
            # Set target for current DOF
            targets[0, dof_index] = dof_pos[0, dof_index] + args.movement_speed
            
            # Get DOF name for debugging
            dof_name = f"DOF_{dof_index}"
            if dof_index < len(base_joint_names):
                dof_name = base_joint_names[dof_index]
            
            # Debug output
            if args.debug and i % 10 == 0:
                print(f"  Moving {dof_name} (index {dof_index}) by {args.movement_speed}")
                print(f"  Current position: {dof_pos[0, dof_index].item():.4f}")
                print(f"  Target position: {targets[0, dof_index].item():.4f}")
            
            # Set DOF position targets
            gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(targets))
            
            # Step the physics
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            
            # Update state tensors
            gym.refresh_dof_state_tensor(sim)
            
            # Render if we have a viewer
            if viewer is not None:
                gym.step_graphics(sim)
                gym.draw_viewer(viewer, sim, True)
                gym.sync_frame_time(sim)
            
            # Sleep between steps
            time.sleep(args.sleep)
    except Exception as e:
        print(f"Error in simulation loop: {e}")
        traceback.print_exc()
    
    # Cleanup
    print("Cleaning up...")
    if viewer is not None:
        try:
            gym.destroy_viewer(viewer)
            print("Viewer destroyed")
        except Exception as e:
            print(f"Error destroying viewer: {e}")
    
    try:
        gym.destroy_sim(sim)
        print("Simulation destroyed")
    except Exception as e:
        print(f"Error destroying simulation: {e}")
    
    print("Test completed successfully")

if __name__ == "__main__":
    main()