# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os, sys, copy
import torch

from isaacgym import gymtorch
from isaacgym import gymapi

from DexHandEnv.utils.torch_jit_utils import (
    scale,
    unscale,
    quat_mul,
    quat_conjugate,
    quat_from_angle_axis,
    to_torch,
    get_axis_params,
    torch_rand_float,
    tensor_clamp,
)

from DexHandEnv.tasks.base.vec_task import VecTask

from icecream import ic


class DexReorient(VecTask):

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

        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]

        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = (
            10.0  # scale factor of velocity based observations
        )

        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.force_scale = self.cfg["env"].get("forceScale", 0.0)
        self.force_prob_range = self.cfg["env"].get("forceProbRange", [0.001, 0.1])
        self.force_decay = self.cfg["env"].get("forceDecay", 0.99)
        self.force_decay_interval = self.cfg["env"].get("forceDecayInterval", 0.08)

        self.dex_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.1)

        self.object_type = self.cfg["env"]["objectType"]
        assert self.object_type in ["cube_single_dof", "block", "egg", "pen"]

        self.ignore_z = self.object_type == "pen"

        self.asset_files_dict = {
            "cube_single_dof": "dexhand/mjcf/cube_single_dof.xml",
            "block": "dexhand/urdf/objects/cube_multicolor.urdf",
            "egg": "dexhand/mjcf/open_ai_assets/hand/egg.xml",
            "pen": "dexhand/mjcf/open_ai_assets/hand/pen.xml",
        }

        if "asset" in self.cfg["env"]:
            self.asset_files_dict["block"] = self.cfg["env"]["asset"].get(
                "assetFileNameBlock", self.asset_files_dict["block"]
            )
            self.asset_files_dict["egg"] = self.cfg["env"]["asset"].get(
                "assetFileNameEgg", self.asset_files_dict["egg"]
            )
            self.asset_files_dict["pen"] = self.cfg["env"]["asset"].get(
                "assetFileNamePen", self.asset_files_dict["pen"]
            )

        # can be "openai", "full_no_vel", "full", "full_state"
        self.obs_type = self.cfg["env"]["observationType"]

        if not (self.obs_type in ["openai", "full_no_vel", "full", "full_state"]):
            raise Exception(
                "Unknown type of observations!\nobservationType should be one of: [openai, full_no_vel, full, full_state]"
            )

        print("Obs type:", self.obs_type)

        self.num_obs_dict = {
            "openai": 42,
            "full_no_vel": 77,
            "full": 158,
            "full_state": 211,
        }

        self.up_axis = "z"

        self.fingertips = [
            "r_f_link1_3",
            "r_f_link2_4",
            "r_f_link3_4",
            "r_f_link4_4",
            "r_f_link5_4",
        ]
        self.num_fingertips = len(self.fingertips)

        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        num_states = 0
        if self.asymmetric_obs:
            num_states = 211

        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = num_states
        self.cfg["env"]["numActions"] = 21

        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        self.dt = self.sim_params.dt
        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(
                round(self.reset_time / (control_freq_inv * self.dt))
            )
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(10.0, 5.0, 1.0)
            cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
            sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
            self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(
                self.num_envs, self.num_fingertips * 6
            )

            dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
            self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(
                self.num_envs, self.num_dex_hand_dofs
            )

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.object_dof = 1
        self.dex_hand_default_dof_pos = torch.zeros(
            self.num_dex_hand_dofs, dtype=torch.float, device=self.device
        )
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dex_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[
            :, : self.num_dex_hand_dofs
        ]
        # ic(self.dof_state.view(self.num_envs, -1, 2).shape)
        self.dex_hand_dof_pos = self.dex_hand_dof_state[..., 0]
        self.dex_hand_dof_vel = self.dex_hand_dof_state[..., 1]

        self.object_dof_state = self.dof_state.view(self.num_envs, -1, 2)[
            :, self.num_dex_hand_dofs : (self.num_dex_hand_dofs + self.object_dof)
        ]
        self.object_dof_pos = self.object_dof_state[..., 0]
        self.object_dof_vel = self.object_dof_state[..., 1]

        self.goal_dof_state = self.dof_state.view(self.num_envs, -1, 2)[
            :, (self.num_dex_hand_dofs + self.object_dof) :
        ]
        self.goal_dof_pos = self.goal_dof_state[..., 0]
        self.goal_dof_vel = self.goal_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(
            self.num_envs, -1, 13
        )
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(
            -1, 13
        )

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device
        )
        self.cur_targets = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device
        )

        self.global_indices = torch.arange(
            self.num_envs * 3, dtype=torch.int32, device=self.device
        ).view(self.num_envs, -1)
        self.x_unit_tensor = to_torch(
            [1, 0, 0], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch(
            [0, 1, 0], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch(
            [0, 0, 1], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.consecutive_successes = torch.zeros(
            1, dtype=torch.float, device=self.device
        )

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)

        self.total_successes = 0
        self.total_resets = 0

        # object apply random forces parameters
        self.force_decay = to_torch(
            self.force_decay, dtype=torch.float, device=self.device
        )
        self.force_prob_range = to_torch(
            self.force_prob_range, dtype=torch.float, device=self.device
        )
        self.random_force_prob = torch.exp(
            (torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
            * torch.rand(self.num_envs, device=self.device)
            + torch.log(self.force_prob_range[1])
        )

        self.rb_forces = torch.zeros(
            (self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device
        )

        # TODO:
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
            self.num_envs, -1, 3
        )  # shape: num_envs, num_bodies, xyz axis

    def create_sim(self):
        self.dt = self.cfg["sim"]["dt"]
        self.up_axis_idx = 2 if self.up_axis == "z" else 1  # index of up axis: Y=1, Z=2

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

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        )
        dex_hand_asset_file = None

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                self.cfg["env"]["asset"].get("assetRoot", asset_root),
            )
            dex_hand_asset_file = self.cfg["env"]["asset"].get(
                "assetFileNamedex", dex_hand_asset_file
            )

        object_asset_file = self.asset_files_dict[self.object_type]

        # load dexhand asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        # asset_options.replace_cylinder_with_capsule = False

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        # Note - DOF mode is set in the MJCF file and loaded by Isaac Gym
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        dex_hand_asset = self.gym.load_asset(
            self.sim, asset_root, dex_hand_asset_file, asset_options
        )

        self.num_dex_hand_bodies = self.gym.get_asset_rigid_body_count(dex_hand_asset)
        self.num_dex_hand_shapes = self.gym.get_asset_rigid_shape_count(dex_hand_asset)
        self.num_dex_hand_dofs = self.gym.get_asset_dof_count(
            dex_hand_asset
        )  # 谜之错误 25
        self.num_dex_hand_actuators = self.gym.get_asset_actuator_count(dex_hand_asset)
        self.num_dex_hand_tendons = self.gym.get_asset_tendon_count(dex_hand_asset)

        # tendon set up
        limit_stiffness = 30
        t_damping = 0.1
        relevant_tendons = [
            "robot0:T_FFJ1c",
            "robot0:T_MFJ1c",
            "robot0:T_RFJ1c",
            "robot0:T_LFJ1c",
        ]
        tendon_props = self.gym.get_asset_tendon_properties(dex_hand_asset)

        for i in range(self.num_dex_hand_tendons):
            for rt in relevant_tendons:
                if self.gym.get_asset_tendon_name(dex_hand_asset, i) == rt:
                    tendon_props[i].limit_stiffness = limit_stiffness
                    tendon_props[i].damping = t_damping
        self.gym.set_asset_tendon_properties(dex_hand_asset, tendon_props)

        # actuated_dof_names = [
        #     self.gym.get_asset_actuator_joint_name(dex_hand_asset, i)
        #     for i in range(self.num_dex_hand_actuators)
        # ]

        actuated_dof_names = [
            "ARRx",
            "ARRy",
            "r_f_joint2_2",
            "r_f_joint2_3",
            "r_f_joint2_4",
            "r_f_joint3_2",
            "r_f_joint3_3",
            "r_f_joint3_4",
            "r_f_joint4_2",
            "r_f_joint4_3",
            "r_f_joint4_4",
            "r_f_joint5_2",
            "r_f_joint5_3",
            "r_f_joint5_4",
            "r_f_joint1_2",
            "r_f_joint1_3",
            "r_f_joint1_1",
            "r_f_joint2_1",
            "r_f_joint3_1",
            "r_f_joint4_1",
            "r_f_joint5_1",
        ]

        self.actuated_dof_indices = [
            self.gym.find_asset_dof_index(dex_hand_asset, name)
            for name in actuated_dof_names
        ]
        self.unactuated_dof_indices = [
            i
            for i in range(self.num_dex_hand_dofs)
            if i not in self.actuated_dof_indices
        ]

        # ic(self.actuated_dof_indices, actuated_dof_names)

        # get dex_hand dof properties, loaded by Isaac Gym from the MJCF file
        dex_hand_dof_props = self.gym.get_asset_dof_properties(dex_hand_asset)
        # set velocity limits for ARRx and ARRy
        dex_hand_dof_props[self.actuated_dof_indices][0]["velocity"] = 1.0
        dex_hand_dof_props[self.actuated_dof_indices][1]["velocity"] = 1.0
        # dex_hand_dof_props[self.unactuated_dof_indices][:]["velocity"] = 0.0

        # ic(dex_hand_dof_props)
        # ic(dex_hand_dof_props["lower"].shape)

        self.dex_hand_dof_lower_limits = []
        self.dex_hand_dof_upper_limits = []
        self.dex_hand_dof_default_pos = []
        self.dex_hand_dof_default_vel = []

        for i in range(self.num_dex_hand_dofs):
            self.dex_hand_dof_lower_limits.append(0.0)
            self.dex_hand_dof_upper_limits.append(0.1)

            self.dex_hand_dof_default_pos.append(0.0)
            self.dex_hand_dof_default_vel.append(0.0)

        self.actuated_dof_indices = to_torch(
            self.actuated_dof_indices, dtype=torch.long, device=self.device
        )
        self.dex_hand_dof_lower_limits = to_torch(
            self.dex_hand_dof_lower_limits, device=self.device
        )
        self.dex_hand_dof_upper_limits = to_torch(
            self.dex_hand_dof_upper_limits, device=self.device
        )
        self.dex_hand_dof_default_pos = to_torch(
            self.dex_hand_dof_default_pos, device=self.device
        )
        self.dex_hand_dof_default_vel = to_torch(
            self.dex_hand_dof_default_vel, device=self.device
        )

        print(f"lower={self.dex_hand_dof_lower_limits}")
        print(f"upper={self.dex_hand_dof_upper_limits}")

        self.fingertip_handles = [
            self.gym.find_asset_rigid_body_index(dex_hand_asset, name)
            for name in self.fingertips
        ]

        # create fingertip force sensors, if needed
        if self.obs_type == "full_state" or self.asymmetric_obs:
            sensor_pose = gymapi.Transform()
            for ft_handle in self.fingertip_handles:
                self.gym.create_asset_force_sensor(
                    dex_hand_asset, ft_handle, sensor_pose
                )

        # load manipulated object and goal assets
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.fix_base_link = False
        object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        object_asset_options.collapse_fixed_joints = True
        object_asset_options.disable_gravity = False
        object_asset_options.thickness = 0.001
        object_asset_options.angular_damping = 0.01
        object_asset_options.use_physx_armature = True
        object_asset = self.gym.load_asset(
            self.sim, asset_root, object_asset_file, object_asset_options
        )
        # info
        self.num_object_bodies = self.gym.get_asset_rigid_body_count(object_asset)
        self.num_object_shapes = self.gym.get_asset_rigid_shape_count(object_asset)
        self.num_object_dofs = self.gym.get_asset_dof_count(object_asset)
        self.num_object_actuators = self.gym.get_asset_actuator_count(object_asset)
        object_dof_props = self.gym.get_asset_dof_properties(object_asset)
        # ic(object_dof_props)
        # for i in range(self.num_object_dofs):
        #     ic(object_dof_props["lower"][i])
        #     ic(object_dof_props["upper"][i])

        goal_object_dof_props = copy.deepcopy(object_dof_props)
        goal_object_dof_props[0]["damping"] = 0

        object_asset_options.disable_gravity = True
        goal_asset = self.gym.load_asset(
            self.sim, asset_root, object_asset_file, object_asset_options
        )

        dex_hand_start_pose = gymapi.Transform()
        dex_hand_start_pose.p = gymapi.Vec3(*get_axis_params(0.5, self.up_axis_idx))
        dex_hand_start_pose.r = gymapi.Quat(0.0, 1, 0, 0)
        # dex_hand_start_pose.r = gymapi.Quat(-0.07, 0, 0.997, 0.0)

        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()
        object_start_pose.p.x = dex_hand_start_pose.p.x

        pose_dx, pose_dy, pose_dz = -0.22, -0, 0.10
        object_start_pose.p.x = dex_hand_start_pose.p.x + pose_dx
        object_start_pose.p.y = dex_hand_start_pose.p.y + pose_dy
        object_start_pose.p.z = dex_hand_start_pose.p.z + pose_dz
        object_start_pose.r = gymapi.Quat(0.0, -1, 0, 1.0)

        if self.object_type == "pen":
            object_start_pose.p.z = dex_hand_start_pose.p.z + 0.02

        self.goal_displacement = gymapi.Vec3(-0.2, -0.06, 0.12)
        self.goal_displacement_tensor = to_torch(
            [
                self.goal_displacement.x,
                self.goal_displacement.y,
                self.goal_displacement.z,
            ],
            device=self.device,
        )
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement

        goal_start_pose.p.z -= 0.04

        # compute aggregate size
        max_agg_bodies = self.num_dex_hand_bodies + 2 + 2
        max_agg_shapes = self.num_dex_hand_shapes + 2 + 2

        self.dex_hands = []
        self.envs = []

        self.object_init_state = []
        self.hand_start_states = []

        self.hand_indices = []
        self.fingertip_indices = []
        self.object_indices = []
        self.goal_object_indices = []

        self.fingertip_handles = [
            self.gym.find_asset_rigid_body_index(dex_hand_asset, name)
            for name in self.fingertips
        ]

        dex_hand_rb_count = self.gym.get_asset_rigid_body_count(dex_hand_asset)
        object_rb_count = self.gym.get_asset_rigid_body_count(object_asset)
        self.object_rb_handles = list(
            range(dex_hand_rb_count, dex_hand_rb_count + object_rb_count)
        )

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            self_collisions = 1  # 1 to disable
            dex_hand_actor = self.gym.create_actor(
                env_ptr,
                dex_hand_asset,
                dex_hand_start_pose,
                "hand",
                i,
                self_collisions,
                0,
            )
            self.hand_start_states.append(
                [
                    dex_hand_start_pose.p.x,
                    dex_hand_start_pose.p.y,
                    dex_hand_start_pose.p.z,
                    dex_hand_start_pose.r.x,
                    dex_hand_start_pose.r.y,
                    dex_hand_start_pose.r.z,
                    dex_hand_start_pose.r.w,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )
            self.gym.set_actor_dof_properties(
                env_ptr, dex_hand_actor, dex_hand_dof_props
            )
            hand_idx = self.gym.get_actor_index(
                env_ptr, dex_hand_actor, gymapi.DOMAIN_SIM
            )
            self.hand_indices.append(hand_idx)

            # enable DOF force sensors, if needed
            if self.obs_type == "full_state" or self.asymmetric_obs:
                self.gym.enable_actor_dof_force_sensors(env_ptr, dex_hand_actor)

            # add object
            object_handle = self.gym.create_actor(
                env_ptr, object_asset, object_start_pose, "object", i, 0, 0
            )
            self.object_init_state.append(
                [
                    object_start_pose.p.x,
                    object_start_pose.p.y,
                    object_start_pose.p.z,
                    object_start_pose.r.x,
                    object_start_pose.r.y,
                    object_start_pose.r.z,
                    object_start_pose.r.w,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )
            object_idx = self.gym.get_actor_index(
                env_ptr, object_handle, gymapi.DOMAIN_SIM
            )
            self.object_indices.append(object_idx)
            self.gym.set_actor_dof_properties(env_ptr, object_handle, object_dof_props)

            # add goal object
            goal_handle = self.gym.create_actor(
                env_ptr,
                goal_asset,
                goal_start_pose,
                "goal_object",
                i + self.num_envs,
                0,
                0,
            )
            goal_object_idx = self.gym.get_actor_index(
                env_ptr, goal_handle, gymapi.DOMAIN_SIM
            )
            self.goal_object_indices.append(goal_object_idx)
            self.gym.set_actor_dof_properties(
                env_ptr, goal_handle, goal_object_dof_props
            )

            if self.object_type != "block":
                self.gym.set_rigid_body_color(
                    env_ptr,
                    object_handle,
                    0,
                    gymapi.MESH_VISUAL,
                    gymapi.Vec3(0.6, 0.72, 0.98),
                )
                self.gym.set_rigid_body_color(
                    env_ptr,
                    goal_handle,
                    0,
                    gymapi.MESH_VISUAL,
                    gymapi.Vec3(0.6, 0.72, 0.98),
                )

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.dex_hands.append(dex_hand_actor)

        # we are not using new mass values after DR when calculating random forces applied to an object,
        # which should be ok as long as the randomization range is not too big
        object_rb_props = self.gym.get_actor_rigid_body_properties(
            env_ptr, object_handle
        )
        self.object_rb_masses = [prop.mass for prop in object_rb_props]

        self.object_init_state = to_torch(
            self.object_init_state, device=self.device, dtype=torch.float
        ).view(self.num_envs, 13)
        self.goal_states = self.object_init_state.clone()
        self.goal_states[:, self.up_axis_idx] -= 0.04
        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(
            self.hand_start_states, device=self.device
        ).view(self.num_envs, 13)

        self.fingertip_handles = to_torch(
            self.fingertip_handles, dtype=torch.long, device=self.device
        )
        self.object_rb_handles = to_torch(
            self.object_rb_handles, dtype=torch.long, device=self.device
        )
        self.object_rb_masses = to_torch(
            self.object_rb_masses, dtype=torch.float, device=self.device
        )

        self.hand_indices = to_torch(
            self.hand_indices, dtype=torch.long, device=self.device
        )
        self.object_indices = to_torch(
            self.object_indices, dtype=torch.long, device=self.device
        )
        self.goal_object_indices = to_torch(
            self.goal_object_indices, dtype=torch.long, device=self.device
        )

        # TODO: contact indices
        body_names = self.gym.get_asset_rigid_body_names(dex_hand_asset)
        # ic(body_names)
        penalized_contact_names = ["test_link2"]
        self.penalised_contact_indices = torch.zeros(
            len(penalized_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.dex_hands[0], penalized_contact_names[i]
            )
        # ic(self.penalised_contact_indices)

    def compute_reward(self, actions):
        (
            self.rew_buf[:],
            self.reset_buf[:],
            self.reset_goal_buf[:],
            self.progress_buf[:],
            self.successes[:],
            self.consecutive_successes[:],
            reward_dict,
        ) = compute_hand_reward(
            self.rew_buf,
            self.reset_buf,
            self.reset_goal_buf,
            self.progress_buf,
            self.successes,
            self.consecutive_successes,
            self.max_episode_length,
            self.object_pos,
            self.object_rot,
            self.goal_pos,
            self.goal_rot,
            self.dist_reward_scale,
            self.rot_reward_scale,
            self.rot_eps,
            self.actions,
            self.action_penalty_scale,
            self.success_tolerance,
            self.reach_goal_bonus,
            self.fall_dist,
            self.fall_penalty,
            self.max_consecutive_successes,
            self.av_factor,
            (self.object_type == "pen"),
            self.contact_forces[:, self.penalised_contact_indices, :],
            self.object_dof_vel,
        )

        self.extras["consecutive_successes"] = self.consecutive_successes.mean()
        for key, value in reward_dict.items():
            self.extras[key] = value

        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = (
                self.total_successes + (self.successes * self.reset_buf).sum()
            )

            # The direct average shows the overall result more quickly, but slightly undershoots long term
            # policy performance.
            print(
                "Direct average consecutive successes = {:.1f}".format(
                    direct_average_successes / (self.total_resets + self.num_envs)
                )
            )
            if self.total_resets > 0:
                print(
                    "Post-Reset average consecutive successes = {:.1f}".format(
                        self.total_successes / self.total_resets
                    )
                )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
            self.gym.refresh_force_sensor_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][
            :, :, 0:13
        ]
        self.fingertip_pos = self.rigid_body_states[:, self.fingertip_handles][
            :, :, 0:3
        ]

        if self.obs_type == "openai":
            self.compute_fingertip_observations(True)
        elif self.obs_type == "full_no_vel":
            self.compute_full_observations(True)
        elif self.obs_type == "full":
            self.compute_full_observations()
        elif self.obs_type == "full_state":
            self.compute_full_state()
        else:
            print("Unknown observations type!")

        if self.asymmetric_obs:
            self.compute_full_state(True)

    def compute_fingertip_observations(self, no_vel=False):
        if no_vel:
            # Per https://arxiv.org/pdf/1808.00177.pdf Table 2
            #   Fingertip positions
            #   Object Position, but not orientation
            #   Relative target orientation

            # 3*self.num_fingertips = 15
            self.obs_buf[:, 0:15] = self.fingertip_pos.reshape(self.num_envs, 15)
            self.obs_buf[:, 15:18] = self.object_pose[:, 0:3]
            self.obs_buf[:, 18:22] = quat_mul(
                self.object_rot, quat_conjugate(self.goal_rot)
            )

            self.obs_buf[:, 22:42] = self.actions
        else:
            # 13*self.num_fingertips = 65
            self.obs_buf[:, 0:65] = self.fingertip_state.reshape(self.num_envs, 65)
            self.obs_buf[:, 65:72] = self.object_pose
            self.obs_buf[:, 72:75] = self.object_linvel
            self.obs_buf[:, 75:78] = self.vel_obs_scale * self.object_angvel

            self.obs_buf[:, 78:85] = self.goal_pose
            self.obs_buf[:, 85:89] = quat_mul(
                self.object_rot, quat_conjugate(self.goal_rot)
            )

            self.obs_buf[:, 89:109] = self.actions

    def compute_full_observations(self, no_vel=False):
        if no_vel:
            self.obs_buf[:, 0 : self.num_dex_hand_dofs] = unscale(
                self.dex_hand_dof_pos,
                self.dex_hand_dof_lower_limits,
                self.dex_hand_dof_upper_limits,
            )

            self.obs_buf[:, 24:31] = self.object_pose
            self.obs_buf[:, 31:38] = self.goal_pose
            self.obs_buf[:, 38:42] = quat_mul(
                self.object_rot, quat_conjugate(self.goal_rot)
            )

            # 3*self.num_fingertips = 15
            self.obs_buf[:, 42:57] = self.fingertip_pos.reshape(self.num_envs, 15)

            self.obs_buf[:, 57:77] = self.actions
        else:  # hand pos + hand vel + object pose + object lin vel + object ang vel
            # + object joint angle + object joint vel
            # + goal joint angle + goal joint vel
            # + fingertip state + actions
            self.obs_buf[:, 0 : self.num_dex_hand_dofs] = unscale(
                self.dex_hand_dof_pos,
                self.dex_hand_dof_lower_limits,
                self.dex_hand_dof_upper_limits,
            )
            self.obs_buf[:, self.num_dex_hand_dofs : 2 * self.num_dex_hand_dofs] = (
                self.vel_obs_scale * self.dex_hand_dof_vel
            )

            self.obs_buf[:, 48:55] = self.object_pose
            self.obs_buf[:, 55:58] = self.object_linvel
            self.obs_buf[:, 58:61] = self.vel_obs_scale * self.object_angvel

            self.obs_buf[:, 61:62] = self.object_dof_pos
            self.obs_buf[:, 62:63] = self.object_dof_vel
            self.obs_buf[:, 63:64] = self.goal_dof_pos

            # 13*self.num_fingertips = 65
            self.obs_buf[:, 64:129] = self.fingertip_state.reshape(self.num_envs, 65)

            self.obs_buf[:, 129:150] = self.actions  # 19 + 2

    def compute_full_state(self, asymm_obs=False):
        if asymm_obs:
            self.states_buf[:, 0 : self.num_dex_hand_dofs] = unscale(
                self.dex_hand_dof_pos,
                self.dex_hand_dof_lower_limits,
                self.dex_hand_dof_upper_limits,
            )
            self.states_buf[:, self.num_dex_hand_dofs : 2 * self.num_dex_hand_dofs] = (
                self.vel_obs_scale * self.dex_hand_dof_vel
            )
            self.states_buf[
                :, 2 * self.num_dex_hand_dofs : 3 * self.num_dex_hand_dofs
            ] = (self.force_torque_obs_scale * self.dof_force_tensor)

            obj_obs_start = 3 * self.num_dex_hand_dofs  # 72
            self.states_buf[:, obj_obs_start : obj_obs_start + 7] = self.object_pose
            self.states_buf[:, obj_obs_start + 7 : obj_obs_start + 10] = (
                self.object_linvel
            )
            self.states_buf[:, obj_obs_start + 10 : obj_obs_start + 13] = (
                self.vel_obs_scale * self.object_angvel
            )

            goal_obs_start = obj_obs_start + 13  # 85
            self.states_buf[:, goal_obs_start : goal_obs_start + 7] = self.goal_pose
            self.states_buf[:, goal_obs_start + 7 : goal_obs_start + 11] = quat_mul(
                self.object_rot, quat_conjugate(self.goal_rot)
            )

            # fingertip observations, state(pose and vel) + force-torque sensors
            num_ft_states = 13 * self.num_fingertips  # 65
            num_ft_force_torques = 6 * self.num_fingertips  # 30

            fingertip_obs_start = goal_obs_start + 11  # 96
            self.states_buf[
                :, fingertip_obs_start : fingertip_obs_start + num_ft_states
            ] = self.fingertip_state.reshape(self.num_envs, num_ft_states)
            self.states_buf[
                :,
                fingertip_obs_start
                + num_ft_states : fingertip_obs_start
                + num_ft_states
                + num_ft_force_torques,
            ] = (
                self.force_torque_obs_scale * self.vec_sensor_tensor
            )

            # obs_end = 96 + 65 + 30 = 191
            # obs_total = obs_end + num_actions = 211
            obs_end = fingertip_obs_start + num_ft_states + num_ft_force_torques
            self.states_buf[:, obs_end : obs_end + self.num_actions] = self.actions
        else:
            self.obs_buf[:, 0 : self.num_dex_hand_dofs] = unscale(
                self.dex_hand_dof_pos,
                self.dex_hand_dof_lower_limits,
                self.dex_hand_dof_upper_limits,
            )
            self.obs_buf[:, self.num_dex_hand_dofs : 2 * self.num_dex_hand_dofs] = (
                self.vel_obs_scale * self.dex_hand_dof_vel
            )
            self.obs_buf[:, 2 * self.num_dex_hand_dofs : 3 * self.num_dex_hand_dofs] = (
                self.force_torque_obs_scale * self.dof_force_tensor
            )

            obj_obs_start = 3 * self.num_dex_hand_dofs  # 72
            self.obs_buf[:, obj_obs_start : obj_obs_start + 7] = self.object_pose
            self.obs_buf[:, obj_obs_start + 7 : obj_obs_start + 10] = self.object_linvel
            self.obs_buf[:, obj_obs_start + 10 : obj_obs_start + 13] = (
                self.vel_obs_scale * self.object_angvel
            )

            goal_obs_start = obj_obs_start + 13  # 85
            self.obs_buf[:, goal_obs_start : goal_obs_start + 7] = self.goal_pose
            self.obs_buf[:, goal_obs_start + 7 : goal_obs_start + 11] = quat_mul(
                self.object_rot, quat_conjugate(self.goal_rot)
            )

            # fingertip observations, state(pose and vel) + force-torque sensors
            num_ft_states = 13 * self.num_fingertips  # 65
            num_ft_force_torques = 6 * self.num_fingertips  # 30

            fingertip_obs_start = goal_obs_start + 11  # 96
            self.obs_buf[
                :, fingertip_obs_start : fingertip_obs_start + num_ft_states
            ] = self.fingertip_state.reshape(self.num_envs, num_ft_states)
            self.obs_buf[
                :,
                fingertip_obs_start
                + num_ft_states : fingertip_obs_start
                + num_ft_states
                + num_ft_force_torques,
            ] = (
                self.force_torque_obs_scale * self.vec_sensor_tensor
            )

            # obs_end = 96 + 65 + 30 = 191
            # obs_total = obs_end + num_actions = 211
            obs_end = fingertip_obs_start + num_ft_states + num_ft_force_torques
            self.obs_buf[:, obs_end : obs_end + self.num_actions] = self.actions

    def reset_target_pose(self, env_ids, apply_reset=False):
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)

        new_rot = randomize_rotation(
            rand_floats[:, 0],
            rand_floats[:, 1],
            self.x_unit_tensor[env_ids],
            self.y_unit_tensor[env_ids],
        )
        # set new rot to standard pose
        new_rot = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)

        self.goal_states[env_ids, 0:3] = self.goal_init_state[env_ids, 0:3]
        # self.goal_states[env_ids, 3:7] = new_rot
        self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = (
            self.goal_states[env_ids, 0:3] + self.goal_displacement_tensor
        )
        self.root_state_tensor[self.goal_object_indices[env_ids], 3:7] = (
            self.goal_states[env_ids, 3:7]
        )
        self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = (
            torch.zeros_like(
                self.root_state_tensor[self.goal_object_indices[env_ids], 7:13]
            )
        )

        if apply_reset:
            goal_object_indices = self.goal_object_indices[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_state_tensor),
                gymtorch.unwrap_tensor(goal_object_indices),
                len(env_ids),
            )
        self.reset_goal_buf[env_ids] = 0

    def reset_idx(self, env_ids, goal_env_ids):
        # randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # generate random values
        rand_floats = torch_rand_float(
            -1.0,
            1.0,
            (len(env_ids), self.num_dex_hand_dofs * 2 + 5),
            device=self.device,
        )
        rand_floats[:, : self.num_dex_hand_dofs] = 1.0

        # randomize start object poses
        self.reset_target_pose(env_ids)

        # reset rigid body forces
        self.rb_forces[env_ids, :, :] = 0.0

        # reset object
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[
            env_ids
        ].clone()
        self.root_state_tensor[self.object_indices[env_ids], 0:2] = (
            self.object_init_state[env_ids, 0:2]
            + self.reset_position_noise * rand_floats[:, 0:2]
        )
        self.root_state_tensor[self.object_indices[env_ids], self.up_axis_idx] = (
            self.object_init_state[env_ids, self.up_axis_idx]
            + self.reset_position_noise * rand_floats[:, self.up_axis_idx]
        )

        new_object_rot = randomize_rotation(
            rand_floats[:, 3],
            rand_floats[:, 4],
            self.x_unit_tensor[env_ids],
            self.y_unit_tensor[env_ids],
        )
        if self.object_type == "pen":
            rand_angle_y = torch.tensor(0.3)
            new_object_rot = randomize_rotation_pen(
                rand_floats[:, 3],
                rand_floats[:, 4],
                rand_angle_y,
                self.x_unit_tensor[env_ids],
                self.y_unit_tensor[env_ids],
                self.z_unit_tensor[env_ids],
            )

        # self.root_state_tensor[self.object_indices[env_ids], 3:7] = new_object_rot
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(
            self.root_state_tensor[self.object_indices[env_ids], 7:13]
        )

        object_indices = torch.unique(
            torch.cat(
                [
                    self.object_indices[env_ids],
                    self.goal_object_indices[env_ids],
                    self.goal_object_indices[goal_env_ids],
                ]
            ).to(torch.int32)
        )
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(object_indices),
            len(object_indices),
        )

        # reset random force probabilities
        self.random_force_prob[env_ids] = torch.exp(
            (torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
            * torch.rand(len(env_ids), device=self.device)
            + torch.log(self.force_prob_range[1])
        )

        # reset dex hand
        delta_max = self.dex_hand_dof_upper_limits - self.dex_hand_dof_default_pos
        delta_min = self.dex_hand_dof_lower_limits - self.dex_hand_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * (
            rand_floats[:, 5 : 5 + self.num_dex_hand_dofs] + 1
        )

        pos = self.dex_hand_default_dof_pos + self.reset_dof_pos_noise * rand_delta
        # import ipdb; ipdb.set_trace()
        self.dex_hand_dof_pos[env_ids, :] = pos
        self.dex_hand_dof_vel[env_ids, :] = (
            self.dex_hand_dof_default_vel
            + self.reset_dof_vel_noise
            * rand_floats[
                :, 5 + self.num_dex_hand_dofs : 5 + self.num_dex_hand_dofs * 2
            ]
        )
        self.dex_hand_dof_pos[:, self.unactuated_dof_indices] = 0.0
        self.dex_hand_dof_vel[:, self.unactuated_dof_indices] = 0.0
        self.prev_targets[env_ids, : self.num_dex_hand_dofs] = pos
        self.cur_targets[env_ids, : self.num_dex_hand_dofs] = pos

        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.prev_targets),
            gymtorch.unwrap_tensor(hand_indices),
            len(env_ids),
        )

        hand_indices = torch.unique(
            torch.cat(
                [
                    self.hand_indices[env_ids],
                    self.object_indices[env_ids],
                    self.goal_object_indices[goal_env_ids],
                ]
            ).to(torch.int32)
        )
        self.reset_target_pose(env_ids)
        self.object_dof_pos[env_ids] = torch_rand_float(
            -torch.pi, torch.pi, (len(env_ids), 1), device=self.device
        )
        # self.object_dof_vel[env_ids] = 10.0
        self.goal_dof_vel[env_ids] = 3.0  # FIXME: for cool
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(hand_indices),
            len(hand_indices),
        )

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

    def pre_physics_step(self, actions):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # if only goals need reset, then call set API
        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids, apply_reset=True)
        # if goals need reset in addition to other envs, call set API in reset_idx()
        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)

        if len(env_ids) > 0:
            self.reset_idx(env_ids, goal_env_ids)

        self.actions = actions.clone().to(self.device)
        if self.random_action:
            self.actions = 2 * torch.rand_like(self.actions) - 1
        if self.use_relative_control:
            targets = (
                self.prev_targets[:, self.actuated_dof_indices]
                + self.dex_hand_dof_speed_scale * self.dt * self.actions
            )
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(
                targets,
                self.dex_hand_dof_lower_limits[self.actuated_dof_indices],
                self.dex_hand_dof_upper_limits[self.actuated_dof_indices],
            )
        else:
            self.cur_targets[:, self.unactuated_dof_indices] = 0.0
            self.cur_targets[:, self.actuated_dof_indices] = scale(
                self.actions,
                self.dex_hand_dof_lower_limits[self.actuated_dof_indices],
                self.dex_hand_dof_upper_limits[self.actuated_dof_indices],
            )
            self.cur_targets[:, self.actuated_dof_indices] = (
                self.act_moving_average * self.cur_targets[:, self.actuated_dof_indices]
                + (1.0 - self.act_moving_average)
                * self.prev_targets[:, self.actuated_dof_indices]
            )
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(
                self.cur_targets[:, self.actuated_dof_indices],
                self.dex_hand_dof_lower_limits[self.actuated_dof_indices],
                self.dex_hand_dof_upper_limits[self.actuated_dof_indices],
            )
            # ic(self.cur_targets[0, self.actuated_dof_indices], self.actions[0, :])

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[
            :, self.actuated_dof_indices
        ]
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.cur_targets)
        )

        if self.force_scale > 0.0:
            self.rb_forces *= torch.pow(
                self.force_decay, self.dt / self.force_decay_interval
            )

            # apply new forces
            force_indices = (
                torch.rand(self.num_envs, device=self.device) < self.random_force_prob
            ).nonzero()
            self.rb_forces[force_indices, self.object_rb_handles, :] = (
                torch.randn(
                    self.rb_forces[force_indices, self.object_rb_handles, :].shape,
                    device=self.device,
                )
                * self.object_rb_masses
                * self.force_scale
            )

            self.gym.apply_rigid_body_force_tensors(
                self.sim,
                gymtorch.unwrap_tensor(self.rb_forces),
                None,
                gymapi.LOCAL_SPACE,
            )

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.compute_observations()
        # import ipdb

        # ipdb.set_trace()
        self.compute_reward(self.actions)

        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                targetx = (
                    (
                        self.goal_pos[i]
                        + quat_apply(
                            self.goal_rot[i],
                            to_torch([1, 0, 0], device=self.device) * 0.2,
                        )
                    )
                    .cpu()
                    .numpy()
                )
                targety = (
                    (
                        self.goal_pos[i]
                        + quat_apply(
                            self.goal_rot[i],
                            to_torch([0, 1, 0], device=self.device) * 0.2,
                        )
                    )
                    .cpu()
                    .numpy()
                )
                targetz = (
                    (
                        self.goal_pos[i]
                        + quat_apply(
                            self.goal_rot[i],
                            to_torch([0, 0, 1], device=self.device) * 0.2,
                        )
                    )
                    .cpu()
                    .numpy()
                )

                p0 = (
                    self.goal_pos[i].cpu().numpy()
                    + self.goal_displacement_tensor.cpu().numpy()
                )
                self.gym.add_lines(
                    self.viewer,
                    self.envs[i],
                    1,
                    [p0[0], p0[1], p0[2], targetx[0], targetx[1], targetx[2]],
                    [0.85, 0.1, 0.1],
                )
                self.gym.add_lines(
                    self.viewer,
                    self.envs[i],
                    1,
                    [p0[0], p0[1], p0[2], targety[0], targety[1], targety[2]],
                    [0.1, 0.85, 0.1],
                )
                self.gym.add_lines(
                    self.viewer,
                    self.envs[i],
                    1,
                    [p0[0], p0[1], p0[2], targetz[0], targetz[1], targetz[2]],
                    [0.1, 0.1, 0.85],
                )

                objectx = (
                    (
                        self.object_pos[i]
                        + quat_apply(
                            self.object_rot[i],
                            to_torch([1, 0, 0], device=self.device) * 0.2,
                        )
                    )
                    .cpu()
                    .numpy()
                )
                objecty = (
                    (
                        self.object_pos[i]
                        + quat_apply(
                            self.object_rot[i],
                            to_torch([0, 1, 0], device=self.device) * 0.2,
                        )
                    )
                    .cpu()
                    .numpy()
                )
                objectz = (
                    (
                        self.object_pos[i]
                        + quat_apply(
                            self.object_rot[i],
                            to_torch([0, 0, 1], device=self.device) * 0.2,
                        )
                    )
                    .cpu()
                    .numpy()
                )

                p0 = self.object_pos[i].cpu().numpy()
                self.gym.add_lines(
                    self.viewer,
                    self.envs[i],
                    1,
                    [p0[0], p0[1], p0[2], objectx[0], objectx[1], objectx[2]],
                    [0.85, 0.1, 0.1],
                )
                self.gym.add_lines(
                    self.viewer,
                    self.envs[i],
                    1,
                    [p0[0], p0[1], p0[2], objecty[0], objecty[1], objecty[2]],
                    [0.1, 0.85, 0.1],
                )
                self.gym.add_lines(
                    self.viewer,
                    self.envs[i],
                    1,
                    [p0[0], p0[1], p0[2], objectz[0], objectz[1], objectz[2]],
                    [0.1, 0.1, 0.85],
                )

    def subscribe_keyboard_event(self):
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_ENTER, "lock viewer to robot"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_UP, "lock viewer to last robot"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_DOWN, "lock viewer to next robot"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_P, "reset the environment"
        )

    def check_keyboard_event(self, action, value):
        if action == "lock viewer to robot" and value > 0:
            self.lock_viewer_to_robot = (self.lock_viewer_to_robot + 1) % 3
        elif action == "lock viewer to next robot" and value > 0:
            if self.follow_robot_index >= 0:
                self.follow_robot_index += 1
                if self.follow_robot_index >= self.num_envs:
                    self.follow_robot_index = 0
        elif action == "lock viewer to last robot" and value > 0:
            if self.follow_robot_index >= 0:
                self.follow_robot_index -= 1
                if self.follow_robot_index < 0:
                    self.follow_robot_index = self.num_envs - 1
        elif action == "reset the environment" and value > 0:
            self.reset_idx([self.follow_robot_index], [self.follow_robot_index])

    def zero_actions(self) -> torch.Tensor:
        """Returns a buffer with zero actions.

        Returns:
            A buffer of zero torch actions
        """
        actions = torch.ones(
            [self.num_envs, self.num_actions],
            dtype=torch.float32,
            device=self.rl_device,
        )
        actions[:, :2] = 0.0

        return actions

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
        pos = self.hand_start_states[self.follow_robot_index, 0:3] + distance
        lookat = self.hand_start_states[self.follow_robot_index, 0:3]
        cam_pos = gymapi.Vec3(pos[0], pos[1], pos[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_hand_reward(
    rew_buf,
    reset_buf,
    reset_goal_buf,
    progress_buf,
    successes,
    consecutive_successes,
    max_episode_length: float,
    object_pos,
    object_rot,
    target_pos,
    target_rot,
    dist_reward_scale: float,
    rot_reward_scale: float,
    rot_eps: float,
    actions,
    action_penalty_scale: float,
    success_tolerance: float,
    reach_goal_bonus: float,
    fall_dist: float,
    fall_penalty: float,
    max_consecutive_successes: int,
    av_factor: float,
    ignore_z_rot: bool,
    contact_forces: torch.Tensor,
    object_dof_vel: torch.Tensor,
):
    # Distance from the hand to the object
    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)

    if ignore_z_rot:
        success_tolerance = 2.0 * success_tolerance

    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = 2.0 * torch.asin(
        torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0)
    )

    dist_rew = goal_dist * dist_reward_scale
    rot_rew = 1.0 / (torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    action_penalty = action_penalty_scale * torch.sum(actions**2, dim=-1)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    # reward = dist_rew + rot_rew + action_penalty

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(
        torch.abs(rot_dist) <= success_tolerance,
        torch.ones_like(reset_goal_buf),
        reset_goal_buf,
    )
    # successes = successes + goal_resets

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    # reward_reach_goal_bonus = torch.where(goal_resets == 1, reach_goal_bonus, torch.zeros_like(reward))
    # reward += reward_reach_goal_bonus

    # Fall penalty: distance to the goal is larger than a threshold
    reward = torch.zeros_like(dist_rew)
    reward_fall_dist = torch.where(
        goal_dist >= fall_dist, fall_penalty, torch.zeros_like(reward)
    )

    # Check env termination conditions, including maximum success number
    resets = torch.where(goal_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)
    if max_consecutive_successes > 0:
        # Reset progress buffer on goal envs if max_consecutive_successes > 0
        progress_buf = torch.where(
            torch.abs(rot_dist) <= success_tolerance,
            torch.zeros_like(progress_buf),
            progress_buf,
        )
        resets = torch.where(
            successes >= max_consecutive_successes, torch.ones_like(resets), resets
        )
    resets = torch.where(
        progress_buf >= max_episode_length - 1, torch.ones_like(resets), resets
    )

    # Apply penalty for not reaching the goal
    reward_consecutive = torch.zeros_like(reward)
    if max_consecutive_successes > 0:
        reward_consecutive = torch.where(
            progress_buf >= max_episode_length - 1,
            0.5 * fall_penalty,
            torch.zeros_like(reward),
        )
        # reward += reward_consecutive

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets
        + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    # reward object dof vel
    reward_object_dof_vel = 0.3 * object_dof_vel
    # if torch.jit.is_scripting():
    #     print(reward.shape, reward_fall_dist.shape, reward_object_dof_vel.shape)
    # else:
    #     print(reward.shape, reward_fall_dist.shape, reward_object_dof_vel.shape)
    reward = reward_fall_dist + reward_object_dof_vel.squeeze()
    reward_dict = {}
    # reward_dict['rew_dist_rew'] = dist_rew.mean()
    # reward_dict['rew_rot_rew'] = rot_rew.mean()
    # reward_dict['rew_action_penalty'] = (action_penalty * action_penalty_scale).mean()
    # reward_dict['red_reach_goal_bonus'] = reward_reach_goal_bonus.mean()
    # reward_dict['rew_fall_dist'] = reward_fall_dist.mean()
    # reward_dict['rew_consecutive'] = reward_consecutive.mean()
    reward_dict["rew_fall_dist"] = reward_fall_dist.mean()
    reward_dict["rew_object_dof_vel"] = reward_object_dof_vel.mean()

    return (
        reward,
        resets,
        goal_resets,
        progress_buf,
        successes,
        cons_successes,
        reward_dict,
    )


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
        quat_from_angle_axis(rand1 * np.pi, y_unit_tensor),
    )


@torch.jit.script
def randomize_rotation_pen(
    rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor
):
    rot = quat_mul(
        quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
        quat_from_angle_axis(rand0 * np.pi, z_unit_tensor),
    )
    return rot
