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

import math
import numpy as np
import os
import torch
import xml.etree.ElementTree as ET

from isaacgymenvs.utils.torch_jit_utils import *
from .base.vec_task import VecTask

from isaacgym import gymutil, gymtorch, gymapi


class DroneHoops(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        # Observations:
        # 0:13 - root state
        self.cfg["env"]["numObservations"] = 13

        # Actions:
        self.cfg["env"]["numActions"] = 4

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        dofs_per_env = 6
        
        # Drone has 5 bodies: 1 root, 4 rotors and the marker
        self.bodies_per_env = 10

        self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        #self.dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        actors_per_env = 2
        vec_root_tensor = gymtorch.wrap_tensor(self.root_tensor).view(self.num_envs, actors_per_env, 13)
        #vec_dof_tensor = gymtorch.wrap_tensor(self.dof_state_tensor).view(self.num_envs, dofs_per_env, 2)

        self.root_states = vec_root_tensor[:, 0, :]
        self.root_positions = self.root_states[:, 0:3]
        self.target_root_positions = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.target_root_positions[:, 2] = 1
        self.root_quats = self.root_states[:, 3:7]
        self.root_linvels = self.root_states[:, 7:10]
        self.root_angvels = self.root_states[:, 10:13]

        self.marker_states = vec_root_tensor[:, 1, :]
        self.marker_positions = self.marker_states[:, 0:3]
        # self.hole_states = vec_root_tensor[:, 2, :]
        # self.hole_positions = self.hole_states[:, 0:3]

        # self.dof_states = vec_dof_tensor
        # self.dof_positions = vec_dof_tensor[..., 0]
        # self.dof_velocities = vec_dof_tensor[..., 1]

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        self.initial_root_states = self.root_states.clone()
        self.initial_marker_states = self.marker_states.clone()
        #self.initial_hole_states = self.hole_states.clone()
        #self.initial_dof_states = self.dof_states.clone()

        self.thrust_lower_limit = 0
        self.thrust_upper_limit = 4000
        self.thrust_velocity_scale = 4000
        self.thrust_lateral_component = 0.2

        # control tensors
        self.thrusts = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device, requires_grad=False)
        self.forces = torch.zeros((self.num_envs, self.bodies_per_env, 3), dtype=torch.float32, device=self.device, requires_grad=False)

        self.all_actor_indices = torch.arange(self.num_envs * 2, dtype=torch.int32, device=self.device).reshape((self.num_envs, 2))
        
        self.vec_collision = gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim))
        print("Collision: ", self.vec_collision.shape)
        self.collision = self.vec_collision.view(self.num_envs, 10, 3)

        if self.viewer:
            # cam_pos = gymapi.Vec3(2.25, 2.25, 3.0)
            # cam_target = gymapi.Vec3(3.5, 4.0, 1.9)
            cam_pos = gymapi.Vec3(-3.0, -3.0, 1.8)
            cam_target = gymapi.Vec3(2.2, 2.0, 1.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

            # need rigid body states for visualizing thrusts
            self.rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
            self.rb_states = gymtorch.wrap_tensor(self.rb_state_tensor).view(self.num_envs, self.bodies_per_env, 13)
            self.rb_positions = self.rb_states[..., 0:3]
            self.rb_quats = self.rb_states[..., 3:7]

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z

        # Mars gravity
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self.dt = self.sim_params.dt
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))


    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_file = "urdf/Drone.urdf"
        hoop_asset_file = "urdf/Hoop_sq.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.angular_damping = 0.0
        asset_options.max_angular_velocity = 100.0
        asset_options.max_linear_velocity = 100.0
        asset_options.slices_per_cylinder = 40
        asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        #asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        #asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        
        #marker_asset = self.gym.create_sphere(self.sim, 0.1, asset_options)
        marker_asset = self.gym.load_asset(self.sim, asset_root, hoop_asset_file, asset_options)
        
        radius_inner = 0.25
        #hole_shape = self.gym.create_sphere(self.sim, radius_inner, asset_options)

        default_pose = gymapi.Transform()
        default_pose.p.z = 1.0

        self.envs = []
        self.actor_handles = []
        for i in range(self.num_envs):
            # create env instance
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            actor_handle = self.gym.create_actor(env, asset, default_pose, "drone", i, 0, 1)

            # dof_props = self.gym.get_actor_dof_properties(env, actor_handle)
            # dof_props['stiffness'].fill(0)
            # dof_props['damping'].fill(0)
            # self.gym.set_actor_dof_properties(env, actor_handle, dof_props)

            marker_handle = self.gym.create_actor(env, marker_asset, default_pose, "hoop", i, 0, 1)
            self.gym.set_rigid_body_color(env, marker_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 0, 0))
            
            #hole_handle = self.gym.create_actor(env, hole_shape, default_pose, "hoop", 0, 0, 1)
            #self.gym.set_rigid_body_color(env, hole_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0, 0, 0))
            
            
            self.body_drone_props = self.gym.get_actor_rigid_body_properties(env, actor_handle)
            self.actor_handles.append(actor_handle)
            self.envs.append(env)
            
        self.drone_mass = 0
        for prop in self.body_drone_props:
            self.drone_mass += prop.mass
        print("Total drone mass: ", self.drone_mass)

        if self.debug_viz:
            # need env offsets for the rotors
            self.rotor_env_offsets = torch.zeros((self.num_envs, 2, 3), device=self.device)
            for i in range(self.num_envs):
                env_origin = self.gym.get_env_origin(self.envs[i])
                self.rotor_env_offsets[i, ..., 0] = env_origin.x
                self.rotor_env_offsets[i, ..., 1] = env_origin.y
                self.rotor_env_offsets[i, ..., 2] = env_origin.z

    


    def set_targets(self, env_ids):
        num_sets = len(env_ids)
        # set target position in the env wall center XZ
        # self.target_root_positions[env_ids, 0] = torch.zeros(num_sets, device=self.device)
        # self.target_root_positions[env_ids, 1] = torch.zeros(num_sets, device=self.device) - self.cfg["env"]['envSpacing']
        # self.target_root_positions[env_ids, 2] = torch.zeros(num_sets, device=self.device) + 1
        
        # # set target position in a random position in the center of the env
        self.target_root_positions[env_ids, 0] = torch_rand_float(-1, 1, (num_sets, 1), self.device).flatten()
        self.target_root_positions[env_ids, 1] = torch_rand_float(0, 0.5, (num_sets, 1), self.device).flatten() - self.cfg["env"]['envSpacing']
        self.target_root_positions[env_ids, 2] = torch_rand_float(1, 1.5, (num_sets, 1), self.device).flatten()
        
        # # set target position in a random position based on the drone position
        #self.target_root_positions[env_ids, 0] = self.root_states[env_ids, 0] + torch_rand_float(-1.5, 1.5, (num_sets, 1), self.device).flatten()
        #self.target_root_positions[env_ids, 1] = self.root_states[env_ids, 1] + torch_rand_float(-1.5, 1.5, (num_sets, 1), self.device).flatten()
        #self.target_root_positions[env_ids, 2] = self.root_states[env_ids, 2] + torch_rand_float(-0.2, 1.5, (num_sets, 1), self.device).flatten()
        
        # add 0.2m to the target position in the Z axis
        
        self.marker_positions[env_ids, 0] = self.target_root_positions[env_ids, 0]
        self.marker_positions[env_ids, 1] = self.target_root_positions[env_ids, 1] + 0.3
        self.marker_positions[env_ids, 2] = self.target_root_positions[env_ids, 2]
        #self.hole_positions[env_ids] = self.target_root_positions[env_ids]
        # copter "position" is at the bottom of the legs, so shift the target up so it visually aligns better
        #self.marker_positions[env_ids, 2] += 0.0
        #self.hole_positions[env_ids, 2] += 0.0
        
        actor_indices = self.all_actor_indices[env_ids, 1].flatten()

        return actor_indices

    def reset_idx(self, env_ids):

        # set rotor speeds
        # self.dof_velocities[:, 1] = -50
        # self.dof_velocities[:, 3] = 50

        num_resets = len(env_ids)

        target_actor_indices = self.set_targets(env_ids)

        actor_indices = self.all_actor_indices[env_ids, 0].flatten()

        self.root_states[env_ids] = self.initial_root_states[env_ids]
        self.root_states[env_ids, 0] += torch_rand_float(-1.5, 1.5, (num_resets, 1), self.device).flatten()
        self.root_states[env_ids, 1] += torch_rand_float(-1.5, 1.5, (num_resets, 1), self.device).flatten()
        self.root_states[env_ids, 2] += torch_rand_float(-0.2, 1.5, (num_resets, 1), self.device).flatten()
        # self.root_states[env_ids, 0] = 2 + self.target_root_positions[env_ids, 0] + torch_rand_float(-1.5, 1.5, (num_resets, 1), self.device).flatten()
        # self.root_states[env_ids, 1] = 2 + self.target_root_positions[env_ids, 1] + torch_rand_float(-1.5, 1.5, (num_resets, 1), self.device).flatten()
        # self.root_states[env_ids, 2] = self.target_root_positions[env_ids, 2] + torch_rand_float(-0.2, 1.5, (num_resets, 1), self.device).flatten()

        #self.gym.set_dof_state_tensor_indexed(self.sim, self.dof_state_tensor, gymtorch.unwrap_tensor(actor_indices), num_resets)

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        return torch.unique(torch.cat([target_actor_indices, actor_indices]))

    def pre_physics_step(self, _actions):
        # resets
        set_target_ids = (self.progress_buf % 500 == 0).nonzero(as_tuple=False).squeeze(-1)

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        actor_indices = torch.tensor([], device=self.device, dtype=torch.int32)
        if len(reset_env_ids) > 0:
            #print("Resetting envs: ", reset_env_ids)
            actor_indices = self.reset_idx(reset_env_ids)
            #print("root states: ", self.root_states[reset_env_ids])
            #print("root positions: ", self.root_positions[reset_env_ids])
            
        
        target_actor_indices = torch.tensor([], device=self.device, dtype=torch.int32)
        if len(set_target_ids) > 0:
            target_actor_indices = self.set_targets(set_target_ids)

        reset_indices = torch.unique(torch.cat([target_actor_indices, actor_indices]))
        if len(reset_indices) > 0:
            self.gym.set_actor_root_state_tensor_indexed(self.sim, self.root_tensor, gymtorch.unwrap_tensor(reset_indices), len(reset_indices))

        actions = _actions.to(self.device)
        #print(actions)
        #thrust_action_speed_scale = 4000
        thrust_prop_0 = torch.clamp(actions[:, 0] * self.thrust_velocity_scale, self.thrust_lower_limit, self.thrust_upper_limit)
        thrust_prop_1 = torch.clamp(actions[:, 1] * self.thrust_velocity_scale, self.thrust_lower_limit, self.thrust_upper_limit)
        thrust_prop_2 = torch.clamp(actions[:, 2] * self.thrust_velocity_scale, self.thrust_lower_limit, self.thrust_upper_limit)
        thrust_prop_3 = torch.clamp(actions[:, 3] * self.thrust_velocity_scale, self.thrust_lower_limit, self.thrust_upper_limit)
        
        #print(self.dt * thrust_prop_0, thrust_prop_1, thrust_prop_2, thrust_prop_3)

        #force_constant = 0.25*self.drone_mass*9.82*torch.ones((self.num_envs, 1), dtype=torch.float32, device=self.device_id, requires_grad=False)
        
        
        self.forces[:, 1, 2] = self.dt * thrust_prop_0
        self.forces[:, 2, 2] = self.dt * thrust_prop_1
        self.forces[:, 3, 2] = self.dt * thrust_prop_2
        self.forces[:, 4, 2] = self.dt * thrust_prop_3

        # clear actions for reset envs
        self.thrusts[reset_env_ids] = 0.0
        self.forces[reset_env_ids] = 0.0

        # apply actions
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.forces), None, gymapi.LOCAL_SPACE)

    def post_physics_step(self):

        self.progress_buf += 1

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        self.compute_observations()
        self.compute_contacts()
        self.compute_reward()

        # debug viz
        if self.viewer and self.debug_viz:
            # compute start and end positions for visualizing thrust lines
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            rotor_indices = torch.LongTensor([2, 4, 6, 8])
            quats = self.rb_quats[:, rotor_indices]
            dirs = -quat_axis(quats.view(self.num_envs * 4, 4), 2).view(self.num_envs, 4, 3)
            starts = self.rb_positions[:, rotor_indices] + self.rotor_env_offsets
            ends = starts + 0.1 * self.thrusts.view(self.num_envs, 4, 1) * dirs

            # submit debug line geometry
            verts = torch.stack([starts, ends], dim=2).cpu().numpy()
            colors = np.zeros((self.num_envs * 4, 3), dtype=np.float32)
            colors[..., 0] = 1.0
            self.gym.clear_lines(self.viewer)
            self.gym.add_lines(self.viewer, None, self.num_envs * 4, verts, colors)

    def compute_observations(self):
        self.obs_buf[..., 0:3] = (self.target_root_positions - self.root_positions) / 3
        self.obs_buf[..., 3:7] = self.root_quats
        self.obs_buf[..., 7:10] = self.root_linvels / 2
        self.obs_buf[..., 10:13] = self.root_angvels / math.pi
        return self.obs_buf

    def compute_contacts(self):
        # compute when a drone has crashed with the hoop
        
        #acquire_net_contact_force_tensor(self: Gym, arg0: Sim)→ Tensor
        # Retrieves buffer for net contract forces. The buffer has shape (num_rigid_bodies, 3). Each contact force state contains one value for each X, Y, Z axis.
        #refresh_net_contact_force_tensor(self: Gym, arg0: Sim)
        
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # Convert the boolean tensor to a float tensor
        #self.collision_indicator = self.collision_indicator.float()

        # Add an extra dimension to match the desired shape
        #self.collision_indicator = self.collision_indicator.unsqueeze(-1)

        #print("Collision indicator: ", self.collision_indicator.shape)
        #print("Collision: ", self.collision_indicator)
        # # print if collision is detected:
        # if self.collision.nonzero(as_tuple=False).squeeze(-1).shape[0] != 0:
        #     print("Collision: ", self.collision.nonzero(as_tuple=False).squeeze(-1).shape, self.collision[self.collision.nonzero(as_tuple=False).squeeze(-1)])
        # #print(self.collision)
        # # print the number of env that has a collition is detected and the force of the collision
        # #print("Collision: ", self.collision.nonzero(as_tuple=False).squeeze(-1).shape, self.collision[self.collision.nonzero(as_tuple=False).squeeze(-1)])
        
        
    
    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_drone_reward(
            self.root_positions,
            self.target_root_positions,
            self.root_quats,
            self.root_linvels,
            self.root_angvels,
            self.reset_buf, self.progress_buf, self.max_episode_length, self.collision
        )


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_drone_reward(root_positions, target_root_positions, root_quats, root_linvels, root_angvels, reset_buf, progress_buf, max_episode_length, collision):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor) -> Tuple[Tensor, Tensor]

    
    # distance to the normal of the target position
    #target_dist_norm = torch.sqrt(torch.square(target_root_positions[..., 0:2] - root_positions[..., 0:2]).sum(-1))
    target_root_positions_xz = target_root_positions[:, [0,2]]
    root_positions_xz = root_positions[:, [0,2]]
                          
    target_dist_norm = torch.sqrt(torch.square(target_root_positions_xz - root_positions_xz).sum(-1))
    # distance to the plane of the target position
    target_dist_plane =  root_positions[..., 1] - target_root_positions[..., 1]
    
    # entrance to the hoop is 0.3m in front of the hoop
    # target_root_positions_exit = target_root_positions.clone()
    # target_root_positions_exit[..., 1] -= 0.3
    
    
    # target_dist = torch.sqrt(torch.square(target_root_positions - root_positions).sum(-1))
    # target_dist_exit = torch.sqrt(torch.square(target_root_positions_exit - root_positions).sum(-1))
    
    # pos_reward = torch.where(target_dist > 0.05, 2.0 / (1.0 + target_dist * target_dist), 3.0 / (1.0 + target_dist_exit * target_dist_exit))
    # #################
    
    # # distance to target position    
    # target_dist = torch.sqrt(torch.square(target_root_positions - root_positions).sum(-1))
    # zy_target_dist = torch.sqrt(torch.square(target_root_positions[..., 1:2] - root_positions[..., 1:2]).sum(-1))
    
    # exp_target_dist = 0.35-0.8*torch.exp(-13.22*torch.square(target_root_positions[..., 0] - root_positions[..., 0]))
    
    # #print(target_dist)
    # pos_reward = 1.0 / (1.0 + target_dist * target_dist) + 1.0 / (1.0 + zy_target_dist * zy_target_dist) + 1.0 / (1.0 + exp_target_dist * exp_target_dist)
       
    
    # target_dist = torch.sqrt(torch.square(target_root_positions - root_positions).sum(-1))
    # print(target_dist)
    # pos_reward = 3.0 / (1.0 + target_dist * target_dist) 
    
      
    #target_root_positions[..., 1] -= 0.4
    
    # # distance to target position  
    target_dist = torch.sqrt(torch.square(target_root_positions - root_positions).sum(-1))
    ## print(target_dist)
    pos_reward = 2.0 / (1.0 + target_dist * target_dist) 
    
    ##  f = max[1-(dp/dmax), 0] where dp is the distance to the target plane means distance in x axis and dmax is a threshold distance
    #f = torch.max(1 - (target_root_positions[..., 0] - root_positions[..., 0]) / 0.5, torch.zeros_like(target_root_positions[..., 0]))
    
    #pos_reward += 
    
    
    
    # uprightness
    ups = quat_axis(root_quats, 2)
    tiltage = torch.abs(1 - ups[..., 2])
    up_reward = 1.0 / (1.0 + tiltage * tiltage)

    # spinning
    spinnage = torch.abs(root_angvels[..., 2])
    spinnage_reward = 1.0 / (1.0 + spinnage * spinnage)

    # combined reward
    # uprigness and spinning only matter when close to the target
    reward = pos_reward + pos_reward * (up_reward + spinnage_reward)
    
    # time penalty
    #time_penalty = 3.0 * progress_buf / max_episode_length
    #reward -= time_penalty
    

    # resets due to misbehavior
    ones = torch.ones_like(reset_buf)
    die = torch.zeros_like(reset_buf)
    die = torch.where(target_dist > 8.0, ones, die)
    die = torch.where(root_positions[..., 2] < 0.5, ones, die)
    die = torch.where(ups[..., 2] < 0, ones, die)
    
    # query contact state to see if we've crashed
    #
    
    #collision_indicator = collision.abs().sum(dim=-1).sum(dim=-1)
    #die = torch.where(collision_indicator > 0.0, ones, die)
    #print("Collision: ", target_dist)
    
    collision_indicator = collision.abs().sum(dim=-1).sum(dim=-1)
    collision_reward = torch.where(collision_indicator != 0.0, -10.0, 0.0)
    reward += collision_reward
    
    hoop = torch.where(target_dist < 0.5, ones, die)
    
    #reach target
    #Reach target
    reach_goal = 15000.0
    
    condition = torch.logical_and(target_dist_plane < 0.05, target_dist_norm < 0.40)

    die = torch.where(condition, ones, die)
    reward = torch.where(condition, reward + reach_goal, reward)
    
    # condition out of hoop
    condition = torch.logical_and(target_dist_plane < 0.0 , target_dist_norm >= 0.4)
    die = torch.where(condition, ones, die)
    

    # resets due to episode length
    reset = torch.where(progress_buf >= max_episode_length - 1, ones, die)

    return reward, reset
