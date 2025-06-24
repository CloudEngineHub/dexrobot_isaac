"""Test script to verify parallel environment reset behavior."""

from dexhand_env.tasks.dexhand_base import DexHandBase
from dexhand_env.tasks.base_task import BaseTask
from loguru import logger
import torch


def main():
    # Create environment with 4 parallel environments
    cfg = {
        "task": {"name": "BaseTask"},
        "env": {
            "numEnvs": 4,
            "envSpacing": 2.0,
            "episodeLength": 50,
            "enableDebugVis": False,
            "aggregateMode": True,
            "clipObservations": 1000.0,
            "clipActions": 1.0,
            "controlFrequencyInv": 5,
            "policyControlsHandBase": True,  # Enable base control
            "policyControlsFingers": True,
            "maxBaseLinearVelocity": 0.3,
            "maxBaseAngularVelocity": 0.5,
            "maxFingerJointVelocity": 0.5,
            "controlMode": "position",
            "observationKeys": ["dof_pos", "dof_vel"],
            "contactForceBodies": [
                "r_f_link1_pad",
                "r_f_link2_pad",
                "r_f_link3_pad",
                "r_f_link4_pad",
                "r_f_link5_pad",
            ],
        },
        "sim": {
            "dt": 0.005,
            "substeps": 1,
            "up_axis": "z",
            "gravity": [0.0, 0.0, -9.81],
            "physx": {
                "num_threads": 4,
                "solver_type": 1,
                "use_gpu": True,
                "num_position_iterations": 8,
                "num_velocity_iterations": 0,
                "contact_offset": 0.001,
                "rest_offset": 0.0,
                "bounce_threshold_velocity": 0.5,
                "max_depenetration_velocity": 1000.0,
                "default_buffer_size_multiplier": 5.0,
                "max_gpu_contact_pairs": 8388608,
                "num_subscenes": 4,
                "contact_collection": 1,
            },
        },
    }

    # Create a dummy task instance
    task = BaseTask(None, None, "cuda:0", 4, cfg)

    # Create environment
    env = DexHandBase(
        cfg=cfg,
        task=task,
        rl_device="cuda:0",
        sim_device="cuda:0",
        graphics_device_id=-1,  # Headless
        headless=True,
        virtual_screen_capture=False,
        force_render=False,
    )

    logger.info(f"Created environment with {env.num_envs} parallel environments")

    # Set different actions for each environment to test parallel behavior
    actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)

    # Give each environment different finger actions (last 12 actions are fingers)
    for i in range(env.num_envs):
        # Set different base position targets for each environment
        if env.num_actions > 12:  # Has base control
            actions[i, 0] = 0.2 * i  # Different X position for each env
            actions[i, 1] = 0.1 * i  # Different Y position for each env
        # Set different finger targets
        actions[i, -12:] = 0.5 * (i + 1) / env.num_envs  # Different finger positions

    logger.info("Initial reset...")
    obs = env.reset()

    # Log initial DOF positions
    logger.info("Initial DOF positions after reset:")
    for i in range(env.num_envs):
        base_dofs = env.dof_pos[i, :6].cpu().numpy()
        logger.info(f"  Env {i} base DOFs: {base_dofs}")

    # Run for some steps
    logger.info("\nRunning simulation with different actions per environment...")
    for step in range(30):
        obs, rew, done, info = env.step(actions)

        if step % 10 == 0:
            logger.info(f"\nStep {step} - DOF positions:")
            for i in range(env.num_envs):
                base_dofs = env.dof_pos[i, :6].cpu().numpy()
                logger.info(f"  Env {i} base DOFs: {base_dofs}")

    # Now check reset behavior
    logger.info("\nForcing reset of environments 1 and 3...")
    reset_envs = torch.tensor([1, 3], device=env.device, dtype=torch.long)
    env.reset_idx(reset_envs)

    logger.info("DOF positions after selective reset:")
    for i in range(env.num_envs):
        base_dofs = env.dof_pos[i, :6].cpu().numpy()
        reset_status = "RESET" if i in [1, 3] else "NOT RESET"
        logger.info(f"  Env {i} base DOFs ({reset_status}): {base_dofs}")

    # Run a few more steps
    logger.info("\nRunning 10 more steps...")
    for step in range(10):
        obs, rew, done, info = env.step(actions)

    logger.info("\nFinal DOF positions:")
    for i in range(env.num_envs):
        base_dofs = env.dof_pos[i, :6].cpu().numpy()
        logger.info(f"  Env {i} base DOFs: {base_dofs}")

    logger.info("\nTest completed!")


if __name__ == "__main__":
    main()
