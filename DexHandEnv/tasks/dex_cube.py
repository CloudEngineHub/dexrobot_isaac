import os
import sys

import torch
import numpy as np
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt
from typing import Dict, Tuple

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_from_angle_axis

from DexHandEnv.tasks.base.vec_task import VecTask
from DexHandEnv.utils.torch_jit_utils import (
    quat_mul,
    tensor_clamp,
    to_torch,
    axisangle2quat,
)


class DexCube(VecTask):

    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
    ):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        logger.add(sys.stderr, level="DEBUG")
        logger.info(
            f"max_episode_length={self.max_episode_length}, dt={self.cfg['sim']['dt']}, total_time={self.max_episode_length * self.cfg['sim']['dt']}"
        )

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.dex_position_noise = self.cfg["env"]["dexPositionNoise"]
        self.dex_rotation_noise = self.cfg["env"]["dexRotationNoise"]
        self.dex_dof_noise = self.cfg["env"]["dexDofNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.control_type = self.cfg["env"]["controlType"]
        assert self.control_type in {
            "osc",
            "joint_tor",
            "joint_pos",
        }, "Invalid control type specified. Must be one of: {osc, joint_tor, joint_pos}"

        # dimensions
        self.cfg["env"]["numObservations"] = (
            19  # Pregrasp finger positions
            + 6  # Pregrasp hand pose
            + 25 * 2  # Current qpos and qvel
            + 1  # Episode time
            + 5  # Contact state
        )
        # actions include: delta EEF if OSC (6) or joint torques (7) + bool gripper (1)
        self.cfg["env"]["numActions"] = 25

        # Values to be filled in at runtime
        self.states = (
            {}
        )  # will be dict filled with relevant states to use for reward calculation
        self.handles = {}  # will be dict mapping names to relevant sim handles
        self.num_dofs = None  # Total number of DOFs per env
        self.actions = None  # Current actions to be deployed
        self._cube_state = None  # Current state of cubeA for the current env
        self._cube_id = None  # Actor ID corresponding to cubeA for a given env

        # Tensor placeholders
        self._root_state = None  # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._q = None  # Joint positions           (n_envs, n_dof)
        self._qd = None  # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = (
            None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        )
        self._contact_forces = None  # Contact forces in sim
        self._eef_state = None  # end effector state (at grasping point)
        # self._j_eef = None  # Jacobian for end effector
        # self._mm = None  # Mass matrix
        self._arm_control = None  # Tensor buffer for controlling arm
        self._hand_control = None  # Tensor buffer for controlling gripper
        self._pos_control = None  # Position actions
        self._dex_effort_limits = None  # Actuator effort limits for dex
        self._global_indices = (
            None  # Unique indices corresponding to all envs in flattened array
        )

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.reward_settings = {
            "height_reward_scale": self.cfg["env"]["heightRewardScale"],
            "distance_reward_scale": self.cfg["env"]["distanceRewardScale"],
            "action_penalty_scale": self.cfg["env"]["actionPenaltyScale"],
            "pregrasp_deviation_scale": self.cfg["env"]["pregraspPenaltyScale"],
            "qdot_penalty_scale": self.cfg["env"]["qdotPenaltyScale"],
            "early_termination_penalty": self.cfg["env"]["terminationPenaltyScale"],
            "success_reward": self.cfg["env"]["successRewardScale"],
        }

        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        # Load pregrasp pose
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        pregrasp_dataset_path = os.path.join(
            current_file_dir, cfg["env"]["pregraspDatasetPath"]
        )
        self.pregrasp_finger_qpos, self.pregrasp_hand_pose = self.load_pregrasp_dataset(
            pregrasp_dataset_path
        )
        self.pregrasp_indices = torch.zeros(
            self.num_envs, dtype=torch.int64, device=self.device
        )

        # dex defaults
        self.dex_default_dof_pos = to_torch([0.0] * 25, device=self.device)

        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        # Refresh tensors
        self._refresh()

        self.extras = {}

    def _update_tensor_views(self):
        self._q = self._dof_state.view(self.num_envs, self.num_dex_dofs, 2)[..., 0]
        self._qd = self._dof_state.view(self.num_envs, self.num_dex_dofs, 2)[..., 1]
        self._eef_state = self._rigid_body_state[:, self.ee_indices, :]
        self._dex_state = self._root_state[:, self.dex_handle, :]
        self._object_state = self._root_state[:, 2, :]

    def load_pregrasp_dataset(self, dataset_path):
        df = pd.read_csv(dataset_path)

        df = df.drop("tag", axis=1)

        # Extract finger joint positions
        finger_cols = [
            col for col in df.columns if col.startswith("r_f_") and col.endswith("_pos")
        ]
        pregrasp_finger_qpos = df[finger_cols].values

        # Extract hand pose
        pose_cols = [col for col in df.columns if col.startswith("hand")]
        pregrasp_hand_pose = df[pose_cols].values

        # Convert to torch tensors
        pregrasp_finger_qpos = torch.tensor(pregrasp_finger_qpos, dtype=torch.float32)
        pregrasp_hand_pose = torch.tensor(pregrasp_hand_pose, dtype=torch.float32)

        pregrasp_finger_qpos = pregrasp_finger_qpos.to(self.device)
        pregrasp_hand_pose = pregrasp_hand_pose.to(self.device)
        return pregrasp_finger_qpos, pregrasp_hand_pose

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self._create_ground_plane()
        self._create_envs(
            self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs))
        )

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../assets"
        )
        dex_asset_file = "mjcf/dex_hand_assets/"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                self.cfg["env"]["asset"].get("assetRoot", asset_root),
            )
            dex_asset_file = self.cfg["env"]["asset"].get(
                "assetFileNamedex", dex_asset_file
            )

        # Load dex asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = False
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = 3
        asset_options.use_mesh_materials = True
        dex_asset = self.gym.load_asset(
            self.sim, asset_root, dex_asset_file, asset_options
        )
        # fingertips = ["ee_link1", "ee_link2", "ee_link3", "ee_link4", "ee_link5"]
        fingertips = [
            "r_f_link1_3",
            "r_f_link2_4",
            "r_f_link3_4",
            "r_f_link4_4",
            "r_f_link5_4",
        ]
        self.fingertip_names = fingertips
        self.fingertip_handles = [
            self.gym.find_asset_rigid_body_index(dex_asset, name) for name in fingertips
        ]
        logger.info(f"fingertip_handles: {self.fingertip_handles}")

        # Create fingertip force sensors, if needed
        sensor_pose = gymapi.Transform()
        sensor_options = gymapi.ForceSensorProperties()
        sensor_options.enable_forward_dynamics_forces = False  # for example gravity
        sensor_options.enable_constraint_solver_forces = True  # for example contacts
        sensor_options.use_world_frame = (
            False  # report forces in world frame (easier to get vertical components)
        )
        for ft_handle in self.fingertip_handles:
            self.gym.create_asset_force_sensor(
                dex_asset, ft_handle, sensor_pose, sensor_options
            )
        # set dex dof properties
        self.num_dex_bodies = self.gym.get_asset_rigid_body_count(dex_asset)
        self.num_dex_dofs = self.gym.get_asset_dof_count(dex_asset)
        logger.info(f"num dex bodies: {self.num_dex_bodies}")
        logger.info(f"num dex dofs: {self.num_dex_dofs}")
        dex_dof_props = self.gym.get_asset_dof_properties(dex_asset)
        self.dex_dof_lower_limits = []
        self.dex_dof_upper_limits = []
        self._dex_effort_limits = []
        for i in range(self.num_dex_dofs):
            dex_dof_props["driveMode"][i] = 3  # gymapi.DOF_MODE_POS
            if self.physics_engine == gymapi.SIM_PHYSX:
                dex_dof_props["stiffness"][i] = 5000
                dex_dof_props["damping"][i] = 100
            else:
                dex_dof_props["stiffness"][i] = 7000.0
                dex_dof_props["damping"][i] = 50.0

            self.dex_dof_lower_limits.append(dex_dof_props["lower"][i])
            self.dex_dof_upper_limits.append(dex_dof_props["upper"][i])
            self._dex_effort_limits.append(dex_dof_props["effort"][i])
        logger.info(f"dex dof lower limits: {self.dex_dof_lower_limits}")
        logger.info(f"dex dof upper limits: {self.dex_dof_upper_limits}")
        logger.info(f"dex effort limits: {self._dex_effort_limits}")
        logger.info(f"dex dof props: {dex_dof_props}")

        self.dex_dof_lower_limits = to_torch(
            self.dex_dof_lower_limits, device=self.device
        )
        self.dex_dof_upper_limits = to_torch(
            self.dex_dof_upper_limits, device=self.device
        )
        self._dex_effort_limits = to_torch(self._dex_effort_limits, device=self.device)

        # Create table asset
        table_thickness = 0.004
        table_pos = [0.0, 0.0, 0.825]
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(
            self.sim, *[2, 2, table_thickness], table_opts
        )
        # Define start pose for table
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._table_surface_pos = np.array(table_pos) + np.array(
            [0, 0, table_thickness / 2]
        )
        self.reward_settings["table_height"] = self._table_surface_pos[2]

        # Create objects
        ## object1 - cube
        self.object_size = 0.06

        cmap = plt.get_cmap(f"tab20")
        colors = cmap.colors
        color = colors[0]
        self.object_color = gymapi.Vec3(color[0], color[1], color[2])

        cube_size = [self.object_size] * 3
        object_opts = gymapi.AssetOptions()
        object_opts.density = 5e2
        self.object_asset = self.gym.create_box(self.sim, *cube_size, object_opts)

        self.object_start_pose = gymapi.Transform()
        self.object_start_pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
        self.object_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for dex
        dex_start_pose = gymapi.Transform()
        dex_start_pose.p = gymapi.Vec3(0, 0.0, 1.2)
        dex_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Compute aggregate size
        num_dex_bodies = self.gym.get_asset_rigid_body_count(dex_asset)
        num_dex_shapes = self.gym.get_asset_rigid_shape_count(dex_asset)
        max_agg_bodies = num_dex_bodies + 1 + 1  # 1 for table
        max_agg_shapes = num_dex_shapes + 1 + 1  # 1 for table

        self.dexs = []
        self.envs = []
        self.dex_handle = 0
        self._object_id = 0

        # Create environments
        for env_idx in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: dex should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create dex
            ## Potentially randomize start pose
            if self.dex_position_noise > 0:
                rand_xy = self.dex_position_noise * (-1.0 + np.random.rand(2) * 2.0)
                dex_start_pose.p = gymapi.Vec3(
                    0.0,
                    0.0,
                    0.0,
                )
            if self.dex_rotation_noise > 0:
                rand_rot = torch.zeros(1, 3)
                rand_rot[:, -1] = self.dex_rotation_noise * (-1.0)
                new_quat = axisangle2quat(rand_rot).squeeze().numpy().tolist()
                dex_start_pose.r = gymapi.Quat(*new_quat)
            dex_self_collisions = 1  # 1 to disable
            dex_actor = self.gym.create_actor(
                env_ptr,
                dex_asset,
                dex_start_pose,
                "dex",
                env_idx,
                dex_self_collisions,  # collision group that actor will be part of. The actor will not collide with anything outside of the same collisionGroup
                0,  # bitwise filter for elements in the same collisionGroup to mask off collision
            )
            self.gym.set_actor_dof_properties(env_ptr, dex_actor, dex_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create table
            table_self_collisions = 0  # 1 to disable
            table_actor = self.gym.create_actor(
                env_ptr,
                table_asset,
                table_start_pose,
                "table",
                env_idx,
                table_self_collisions,
                0,
            )

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create objects
            object_self_collisions = 0  # 1 to disable
            self._object_id = self.gym.create_actor(
                env_ptr,
                self.object_asset,
                self.object_start_pose,
                f"box",
                env_idx,
                object_self_collisions,
                0,
            )
            # Set colors
            self.gym.set_rigid_body_color(
                env_ptr,
                self._object_id,
                0,  # index of rigid body to be set
                gymapi.MESH_VISUAL,
                self.object_color,
            )
            print(f"Created object with color: {self.object_color}")

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.dexs.append(dex_actor)

        # Setup init state buffer
        self._init_object_state = torch.zeros(self.num_envs, 13, device=self.device)

        # Setup data
        self.init_data()

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        self.handles = {}
        # 添加hand handle
        self.handles["hand"] = self.gym.find_actor_rigid_body_handle(
            env_ptr, self.dex_handle, "r_f_link1_1"
        )
        # 循环添加finger tip handles
        for i in range(1, 6):  # 从1到5
            tip_name = self.fingertip_names[i - 1]
            self.handles[f"finger_tip{i}"] = self.gym.find_actor_rigid_body_handle(
                env_ptr, self.dex_handle, tip_name
            )

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(
            self.num_envs, -1, 13
        )
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(
            self.num_envs, -1, 2
        )
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(
            self.num_envs, -1, 13
        )

        self.ee_indices = [self.handles[f"finger_tip{i+1}"] for i in range(5)]
        logger.info(f"EE indices: {self.ee_indices}")

        # NOTE: root_state: num_envs * (1+1+num_type_of_objects) * 13
        self._objects_state = self._root_state[:, self._object_id, :]
        self._dex_state = self._root_state[:, self.dex_handle, :]

        # Set up contact force
        _net_contact_forces_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.contact_force_tensor = gymtorch.wrap_tensor(
            _net_contact_forces_tensor
        ).view(self.num_envs, -1, 3)
        logger.info(
            f"Contact force tensor dimension: {self.contact_force_tensor.shape}"
        )

        # Set up torque and touch sensors
        _force_sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        self.vec_sensor_tensor = gymtorch.wrap_tensor(_force_sensor_tensor).view(
            self.num_envs, 5, 6
        )
        _dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(_dof_force_tensor).view(
            self.num_envs, self.num_dex_dofs
        )  # FIXME: useless now

        self._update_tensor_views()

        # Initialize actions
        self._pos_control = torch.zeros(
            (self.num_envs, self.num_dex_dofs), dtype=torch.float, device=self.device
        )

        # # Initialize control
        # self.arm_dof = 2
        # self._arm_control = self._pos_control[:, : self.arm_dof]
        # self._hand_control = self._pos_control[:, self.arm_dof :]

        # Initialize indices
        self._global_indices = torch.arange(
            self.num_envs * (1 + 1 + 1),
            dtype=torch.int32,
            device=self.device,
        ).view(
            self.num_envs, -1
        )  # order: dex, table, objects

    def _step_physics(self):
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self._refresh()

    def reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(
                start=0, end=self.num_envs, device=self.device, dtype=torch.long
            )

        # Reset object states
        self._reset_object_state(env_ids, self.init_object_state(env_ids))

        # Reset hand state
        self._reset_dex_dof_and_state(env_ids)

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def _reset_object_state(self, env_ids, nominal_object_state):
        num_resets = len(env_ids)

        # Randomize object properties
        scale_random = (
            torch.rand((num_resets,), device=self.device) * 0.1 + 0.9
        )  # Random scale from 0.9 to 1.0
        size_random = 0.06 * scale_random
        pos_random = (
            torch.rand((num_resets, 2), device=self.device) * 0.02 - 0.01
        )  # Random xy offset between -0.01 and 0.01
        rot_random = (
            torch.rand((num_resets,), device=self.device) * np.pi / 6 - np.pi / 12
        )  # Random rotation in ±15 degrees

        # Apply randomizations
        object_state = self._apply_randomizations(
            nominal_object_state, size_random, pos_random, rot_random
        )

        # Set object states and perform physics steps
        self._set_object_state_and_step(env_ids, object_state)

        # Apply scale and reset state
        self._apply_scale_and_reset_state(env_ids, object_state, scale_random)

    def _apply_randomizations(
        self, nominal_object_state, size_random, pos_random, rot_random
    ):
        object_state = nominal_object_state.clone()
        object_state[:, :2] += pos_random
        object_state[:, 2] = self._table_surface_pos[2] + size_random / 2

        z_axis = torch.tensor([[0.0, 0.0, 1.0]], device=self.device)
        rot_quat = quat_from_angle_axis(rot_random, z_axis)
        object_state[:, 3:7] = quat_mul(rot_quat, object_state[:, 3:7])

        return object_state

    def _set_object_state_and_step(self, env_ids, object_state):
        self._objects_state[env_ids, :] = object_state
        self._root_state[env_ids, 2, :] = object_state

        if len(env_ids) > 0:
            multi_env_ids_objects_int32 = self._global_indices[env_ids, 2:].flatten()
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self._root_state),
                gymtorch.unwrap_tensor(multi_env_ids_objects_int32),
                len(multi_env_ids_objects_int32),
            )

        self._step_physics()

    def _apply_scale_and_reset_state(self, env_ids, object_state, scale_random):
        object_handle = self._object_id
        for i, env_id in enumerate(env_ids):
            self.gym.set_actor_scale(self.envs[env_id], object_handle, scale_random[i])

        self._step_physics()

        self._objects_state[env_ids, :] = object_state
        self._root_state[env_ids, 2, :] = object_state
        if len(env_ids) > 0:
            multi_env_ids_objects_int32 = self._global_indices[env_ids, 2:].flatten()
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self._root_state),
                gymtorch.unwrap_tensor(multi_env_ids_objects_int32),
                len(multi_env_ids_objects_int32),
            )

        self._step_physics()

    def _reset_dex_dof_and_state(self, env_ids):
        num_resets = len(env_ids)

        # Select pregrasp poses
        pregrasp_indices = torch.randint(
            0, len(self.pregrasp_hand_pose), (num_resets,), device=self.device
        )
        self.pregrasp_indices[env_ids] = pregrasp_indices

        # Get selected pregrasp poses
        selected_hand_poses = self.pregrasp_hand_pose[pregrasp_indices]
        selected_finger_qpos = self.pregrasp_finger_qpos[pregrasp_indices]
        # Set hand pose and DOF states
        self._set_hand_pose_and_dof_states(
            env_ids, selected_hand_poses, selected_finger_qpos
        )

        # Deploy updates
        self._deploy_dex_updates(env_ids)

    def _set_hand_pose_and_dof_states(self, env_ids, hand_poses, finger_qpos):
        self._dex_state[env_ids, :] = 0
        self._dex_state[env_ids, 2] = 0.85
        self._dex_state[env_ids, 5] = np.sqrt(2) / 2
        self._dex_state[env_ids, 6] = np.sqrt(2) / 2

        self._root_state[env_ids, self.dex_handle, :] = self._dex_state[env_ids, :]

        dof_pos = torch.zeros((len(env_ids), self.num_dex_dofs), device=self.device)

        dof_pos[:, :6] = hand_poses
        dof_pos[:, 6:] = finger_qpos

        dof_pos = tensor_clamp(
            dof_pos,
            self.dex_dof_lower_limits.unsqueeze(0),
            self.dex_dof_upper_limits.unsqueeze(0),
        )

        self._q[env_ids, :] = dof_pos
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])
        self._pos_control[env_ids, :] = dof_pos

    def _deploy_dex_updates(self, env_ids):
        if len(env_ids) > 0:
            multi_env_ids_dex_int32 = self._global_indices[
                env_ids, self.dex_handle
            ].flatten()

            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self._root_state),
                gymtorch.unwrap_tensor(multi_env_ids_dex_int32),
                len(multi_env_ids_dex_int32),
            )

            self.gym.set_dof_position_target_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self._pos_control),
                gymtorch.unwrap_tensor(multi_env_ids_dex_int32),
                len(multi_env_ids_dex_int32),
            )

            self.gym.set_dof_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self._dof_state),
                gymtorch.unwrap_tensor(multi_env_ids_dex_int32),
                len(multi_env_ids_dex_int32),
            )

        self._step_physics()

    def _update_states(self):
        ee_contact_force = self.contact_force_tensor[:, self.ee_indices, :]
        ee_contact_force_norm = ee_contact_force.norm(dim=-1)
        ee_contact_state = ee_contact_force_norm > 0.01
        finger_to_object_dist = torch.stack(
            [
                torch.linalg.norm(
                    self._eef_state[:, i, :3] - self._object_state[:, :3], dim=-1
                )
                for i in range(5)
            ],
            dim=-1,
        )
        self.states.update(
            {
                # dex state
                "q": self._q[:, :],
                "dq": self._qd[:, :],
                "eef_pos": self._eef_state[:, :, :3].reshape(self.num_envs, -1),
                "eef_quat": self._eef_state[:, :, 3:7].reshape(self.num_envs, -1),
                "eef_vel": self._eef_state[:, :, 7:].reshape(self.num_envs, -1),
                # dex force
                "ee_contact_state": ee_contact_state,
                # object state
                "object_pos": self._object_state[:, :3],
                "object_quat": self._object_state[:, 3:7],
                # pregrasp pose (fixed in one episode)
                "pregrasp_finger_qpos": self.pregrasp_finger_qpos[
                    self.pregrasp_indices
                ],
                "pregrasp_hand_pose": self.pregrasp_hand_pose[self.pregrasp_indices],
                "finger_to_object_dist": finger_to_object_dist,
                # 5 physics steps per RL step
                "episode_time": self.progress_buf.unsqueeze(-1) * self.dt * 5,
            }
        )

    def _refresh(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        # self.gym.refresh_jacobian_tensors(self.sim)
        # self.gym.refresh_mass_matrix_tensors(self.sim)
        # self.gym.refresh_dof_force_tensor(self.sim)

        # Refresh states
        self._update_tensor_views()
        self._update_states()

    def compute_observations(self):
        self._refresh()
        obs = [
            "pregrasp_finger_qpos",
            "pregrasp_hand_pose",
            "q",
            "dq",
            "episode_time",
            "ee_contact_state",
        ]
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)

        return self.obs_buf

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:], reward_terms = compute_dex_reward(
            self.reset_buf,
            self.progress_buf,
            self.actions,
            self.states,
            self.reward_settings,
            self.max_episode_length,
            self.dt * 5,
        )

        for term, value in reward_terms.items():
            mean_value = value.mean().item()
            self.extras[f"rewards/{term}"] = mean_value

    def init_object_state(self, env_ids):
        # If env_ids is None, we reset all the envs
        if env_ids is None:
            env_ids = torch.arange(
                start=0, end=self.num_envs, device=self.device, dtype=torch.long
            )
        num_resets = len(env_ids)

        # Initialize buffer to hold sampled values
        object_state = torch.zeros(num_resets, 13, device=self.device)
        object_state[:, 6] = 1.0
        total_offset = 0

        object_height = self.object_size
        # Set x and y value
        total_offset += object_height
        object_state[:, 0] = 0.0
        object_state[:, 1] = 0.0
        # Set z value, which is fixed height
        object_state[:, 2] = self._table_surface_pos[2] + object_height / 2

        return object_state

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

        # Convert actions to qpos increments
        qpos_increments = self.actions * self.cfg["env"]["actionScale"]
        # Update target positions
        self._pos_control = self._q + qpos_increments
        # Clip to joint limits
        self._pos_control = torch.clamp(
            self._pos_control,
            self.dex_dof_lower_limits.unsqueeze(0),
            self.dex_dof_upper_limits.unsqueeze(0),
        )

        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self._pos_control)
        )

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward()
        self.debug_visualize()

    def debug_visualize(self):
        """
        Draw thumb finger tip in red if in contact with object
        """
        env_index = 0
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            rigid_body_list = np.array(self.fingertip_handles)
            # draw contact force, FIXME: cpu only for isaacgym
            # self.gym.draw_env_rigid_contacts(
            #     self.viewer, self.envs[env_index], gymapi.Vec3(1, 0, 0), 5, False
            # )
            # set finger colors
            for i in range(5):
                if self.states["ee_contact_state"][env_index, i]:
                    self.gym.set_rigid_body_color(
                        self.envs[env_index],
                        self.dexs[env_index],
                        rigid_body_list[i],  # index of rigid body to be set
                        gymapi.MESH_VISUAL,
                        gymapi.Vec3(1, 0, 0),  # red
                    )
                else:
                    self.gym.set_rigid_body_color(
                        self.envs[env_index],
                        self.dexs[env_index],
                        rigid_body_list[i],
                        gymapi.MESH_VISUAL,
                        gymapi.Vec3(0, 0, 0),
                    )

            # Grab relevant states to visualize
            eef_pos = self.states["eef_pos"].reshape(self.num_envs, -1, 3)
            eef_rot = self.states["eef_quat"].reshape(self.num_envs, -1, 4)

            self.gym.add_lines(
                self.viewer,
                self.envs[env_index],
                1,
                [0, 0, 0.01, 0.2, 0, 0.01],
                [0.85, 0.1, 0.1],
            )
            self.gym.add_lines(
                self.viewer,
                self.envs[env_index],
                1,
                [0, 0, 0.01, 0, 0.2, 0.01],
                [0.1, 0.85, 0.1],
            )
            self.gym.add_lines(
                self.viewer,
                self.envs[env_index],
                1,
                [0, 0, 0.01, 0, 0, 0.2],
                [0.1, 0.1, 0.85],
            )

    #####################################################################
    ###======================keyboard functions=======================###
    #####################################################################

    def subscribe_keyboard_event(self):
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_ENTER, "lock viewer to robot"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_P, "reset the environment"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_Q, "exit camera follow mode"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_R, "record video"
        )

    def check_keyboard_event(self, action, value):
        if action == "lock viewer to robot" and value > 0:
            self.lock_viewer_to_robot = (self.lock_viewer_to_robot + 1) % 3
        elif action == "reset the environment" and value > 0:
            self.reset_idx(torch.tensor([self.follow_robot_index], device=self.device))
        elif action == "exit camera follow mode" and value > 0:
            self.lock_viewer_to_robot = 0

    def viewer_follow(self):
        """
        Callback called before rendering the scene
        Default behaviour: Follow robot
        """
        if self.lock_viewer_to_robot == 0:
            return
        distance = 0
        if self.lock_viewer_to_robot == 1:
            distance = torch.tensor(
                [-1.4, 0, 0.6], device=self.device, requires_grad=False
            )
        elif self.lock_viewer_to_robot == 2:
            distance = torch.tensor(
                [0, -1, 0.8], device=self.device, requires_grad=False
            )
        pos = self._dex_state[self.follow_robot_index, 0:3] + distance
        lookat = self._dex_state[self.follow_robot_index, 0:3]
        cam_pos = gymapi.Vec3(pos[0], pos[1], pos[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_dex_reward(
    reset_buf: torch.Tensor,
    progress_buf: torch.Tensor,
    actions: torch.Tensor,
    states: Dict[str, torch.Tensor],
    reward_settings: Dict[str, float],
    max_episode_length: float,
    dt: float,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute the reward for the Dex environment.

    Args:
        reset_buf: Buffer indicating which environments need to be reset
        progress_buf: Buffer indicating the progress of each environment
        actions: The actions taken by the agent
        states: The states of the environment (Dict[str, Tensor])
        reward_settings: The reward settings (Dict[str, float])
        max_episode_length: The maximum length of an episode
        dt: The time step of the environment

    Returns:
        rewards: The computed rewards
        reset_buf: The updated reset buffer
        reward_terms: The individual reward terms (Dict[str, Tensor])
    """
    num_envs = reset_buf.size(0)
    device = reset_buf.device

    # Compute object height relative to table
    object_height = states["object_pos"][:, 2] - reward_settings["table_height"]

    # Main reward: clipped object height
    height_reward = reward_settings["height_reward_scale"] * torch.clamp(
        object_height, 0.0, 0.3
    )

    # Penalty for large actions
    actions_arm = actions[:, :6]
    actions_hand = actions[:, 6:]
    action_arm_penalty = torch.sum(actions_arm**2, dim=-1)
    action_hand_penalty = torch.sum(actions_hand**2, dim=-1)
    action_penalty = action_arm_penalty + 1e-2 * action_hand_penalty
    action_penalty = reward_settings["action_penalty_scale"] * action_penalty

    # Penalty for deviation from pregrasp gesture
    pregrasp_deviation = torch.sum(
        (states["q"][:, 6:] - states["pregrasp_finger_qpos"]) ** 2, dim=-1
    )
    pregrasp_penalty = reward_settings["pregrasp_deviation_scale"] * pregrasp_deviation

    # Distance reward
    finger_to_object_dist = states["finger_to_object_dist"]
    dist_reward = reward_settings["distance_reward_scale"] * (
        1 - torch.tanh(finger_to_object_dist)
    )
    dist_reward = dist_reward.mean(dim=-1)  # average across 5 fingers

    rewards = height_reward + dist_reward - action_penalty - pregrasp_penalty

    """
    **Boolean indexing** is a powerful array indexing technique that uses arrays of True/False to select specific elements.
    It enables you to use logical conditions to extract or modify data in arrays.
    e.g. rewards[early_termination]
    If agent died to early termination, then we need subtract the penalty.
    """

    reset_buf = torch.zeros(num_envs, device=device)

    # Terminate if max episode length is reached
    timeout = progress_buf >= max_episode_length
    reset_buf[timeout] = 1

    # Terminate if object is too low after 3 seconds
    time_condition = progress_buf * dt >= 3.0
    height_condition = object_height < 0.1
    early_termination = torch.logical_and(time_condition, height_condition)
    reset_buf[early_termination] = 1

    # Penalty for early termination
    termination_penalty = torch.zeros_like(rewards)
    termination_penalty[early_termination] = reward_settings[
        "early_termination_penalty"
    ]
    rewards[early_termination] -= reward_settings["early_termination_penalty"]

    # Reward for successfully completing an episode
    success_condition = torch.logical_and(timeout, object_height >= 0.1)
    success_reward = torch.zeros_like(rewards)
    success_reward[success_condition] = reward_settings["success_reward"]
    rewards[success_condition] += reward_settings["success_reward"]

    reward_terms = {
        "height_reward": height_reward,
        "dist_reward": dist_reward,
        "success_reward": success_reward,
        "action_penalty": -action_penalty,
        "pregrasp_penalty": -pregrasp_penalty,
        "termination_penalty": -termination_penalty,
    }

    return rewards, reset_buf, reward_terms
