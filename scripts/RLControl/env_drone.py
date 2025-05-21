from time import sleep
from typing import Any
import gymnasium as gym
import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from genesis.engine.entities import DroneEntity
#import gen_obstacles
import numpy as np
from tianshou.env import BaseVectorEnv
from tianshou.env.utils import gym_new_venv_step_type

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class DroneEnv(gym.Env):
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg,seed, show_viewer=False,device="cuda",n_render_env=None,log_level=40):
        print("In Drone ENV")
        self.device = torch.device(device)

  
        self.env_num=num_envs

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        
        self.dt = 0.01  # run in 100hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.terminated=[False]*num_envs
        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float64)
        self.seed(seed)
        
        #                torch.clip(self.rel_pos * self.obs_scales["rel_pos"], -1, 1),
        #        self.base_quat,
        #        torch.clip(self.base_lin_vel * self.obs_scales["lin_vel"], -1, 1),
        #        torch.clip(self.base_ang_vel * self.obs_scales["ang_vel"], -1, 1),
        #         self.last_actions,
        
        # --- Observation Space ---
        # Define reasonable bounds for observation space components
        low_obs = np.array(
            [-5]*3 + # pos_error
            [-1]*4 + # quat
            [-10]*3 + # lin_vel
            [-np.pi*4]*3 + # ang_vel (approx 4 rotations/sec)
            [-1.0]*4
        )
        high_obs = np.array(
            [5]*3 + # pos_error
            [1]*4 + # quat
            [10]*3 + # lin_vel
            [np.pi*4]*3 + # ang_vel
            [1.0]*4 #last action
        )
        self.observation_space = gym.spaces.Box(low=low_obs, high=high_obs, dtype=np.float64)


        if not gs._initialized:
            gs.init(backend=gs.gpu,debug=False,precision="32",logging_level=log_level)  
        
        self.scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt= 0.01 , substeps=1),
        viewer_options=gs.options.ViewerOptions(
            max_FPS=env_cfg["max_visualize_FPS"],
            camera_pos=(3.0, 0.0, 3.0),
            camera_lookat=(0.0, 0.0, 1.0),
            camera_fov=40,
        ),
        vis_options=gs.options.VisOptions(rendered_envs_idx=n_render_env),
        rigid_options=gs.options.RigidOptions(
            dt=self.dt ,
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
            enable_joint_limit=True,
        ),
        show_viewer=show_viewer,
        )
        # add plane
        self.scene.add_entity(gs.morphs.Plane())

        # add target
        if self.env_cfg["visualize_target"]:
            self.target = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="meshes/sphere.obj",
                    scale=0.05,
                    fixed=True,
                    collision=False,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(
                        color=(1.0, 0.5, 0.5),
                    ),
                ),
            )
        else:
            self.target = None

        # add camera
        if self.env_cfg["visualize_camera"]:
            self.cam = self.scene.add_camera(
                res=(640, 480),
                pos=(3.5, 0.0, 2.5),
                lookat=(0, 0, 0.5),
                fov=30,
                GUI=True,
            )

        # add drone
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.drone:DroneEntity = self.scene.add_entity(gs.morphs.Drone(file="urdf/drones/cf2x.urdf"))
        print("Drone Added")
        # build scene
        self.scene.build(n_envs=num_envs)
        super().__init__()
        print("Scene Builted")
        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # initialize buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.crash_condition= torch.tensor([False]*self.num_envs,device=self.device)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)

        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_base_pos = torch.zeros_like(self.base_pos)
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        self.extras = dict()  # extra information for logging
        self.reset()
        self.waiting_id: list[int] = []
        self.is_async=False
            # all environments are ready in the beginning
        self.ready_id = list(range(self.num_envs))
        self.is_closed = False

    def seed(self,seed):
        self.generator=torch.manual_seed(seed)
        np.random.seed(seed)
    def __len__(self):
        return self.num_envs
    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["pos_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["pos_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["pos_z_range"], (len(envs_idx),), self.device)
        if self.target is not None:
            self.target.set_pos(self.commands[envs_idx], zero_velocity=True, envs_idx=envs_idx)

    def _at_target(self):
        at_target = (
            (torch.norm(self.rel_pos, dim=1) < self.env_cfg["at_target_threshold"]).nonzero(as_tuple=False).flatten()
        )
        return at_target
    def render(self):
        pass
    def step(self, actions,ready_envs):
        actions=torch.from_numpy(actions).to(device=self.device).float()
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions
        mask=np.array([x in ready_envs for x in range(self.num_envs)])
        not_ready_envs=np.array([x for x in range(self.num_envs) if x not in ready_envs])
        #exec_actions[~self.crash_condition]=self.actions
        exec_actions[mask]=self.actions
        
        self.actions=exec_actions
        # 14468 is hover rpm
        self.drone.set_propellels_rpm((1 + exec_actions * 0.8) * 14468.429183500699)

        ### Save state for not operated agents
        temp_pos=self.drone.get_pos()
        temp_orientation=self.drone.get_quat()
        
        self.scene.step()
        
        ### Correct state for not operated agents
        if len(not_ready_envs)>0:
            #print(not_ready_envs)
            self.drone.set_pos(temp_pos[not_ready_envs],not_ready_envs)
            self.drone.set_quat(temp_orientation[not_ready_envs],not_ready_envs)


       
        # update buffers
        self.episode_length_buf[ready_envs] += 1
        self.last_base_pos[:] = self.base_pos[:]
        self.base_pos[:] = self.drone.get_pos()
        self.rel_pos = self.commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos
        self.base_quat[:] = self.drone.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.drone.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.drone.get_ang(), inv_base_quat)

        # resample commands
        envs_idx = self._at_target()
        self._resample_commands(envs_idx)

        # check termination and reset
        self.crash_condition = (
            (torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"])
            | (torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"])
            | (torch.abs(self.rel_pos[:, 0]) > self.env_cfg["termination_if_x_greater_than"])
            | (torch.abs(self.rel_pos[:, 1]) > self.env_cfg["termination_if_y_greater_than"])
            | (torch.abs(self.rel_pos[:, 2]) > self.env_cfg["termination_if_z_greater_than"])
            | (self.base_pos[:, 2] < self.env_cfg["termination_if_close_to_ground"])
        )#| self.crash_condition
        #self.terminated = (self.episode_length_buf > self.max_episode_length) | self.crash_condition
        #self.terminated = self.crash_condition
        self.reset_buf = (self.episode_length_buf > self.max_episode_length) | self.crash_condition

        time_out_idx = (self.episode_length_buf > self.max_episode_length)
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0
        self.terminated=self.crash_condition.cpu().numpy()
        self.truncated=time_out_idx.cpu().numpy()
        #self.base_pos[self.crash_condition] = torch.rand((len(torch.where(self.crash_condition)), 3), device=self.device, dtype=gs.tc_float)#self.base_init_pos
        #self.base_quat[self.crash_condition] = self.base_init_quat.reshape(1, -1)
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())
        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations
        self.obs_buf = torch.cat(
            [
                torch.clip(self.rel_pos * self.obs_scales["rel_pos"], -1, 1),
                self.base_quat,
                torch.clip(self.base_lin_vel * self.obs_scales["lin_vel"], -1, 1),
                torch.clip(self.base_ang_vel * self.obs_scales["ang_vel"], -1, 1),
                self.last_actions,
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]

        return self.obs_buf[mask], self.rew_buf[mask].cpu().numpy(),self.terminated[mask],self.truncated[mask],self._get_info_redux(ready_envs)
        obs_next_RO, rew_R, terminated_R, truncated_R, info_R
    
    def _get_info_redux(self,env_idx=None):
        if env_idx is None:
            return self._get_info()
        elif len(env_idx)==1:
            return self._get_info(env_idx)
        elif len(env_idx)==0:
            return np.array()
        else:
            info_redux=list()
            for ind in env_idx:
                info_redux.append( {
                        "distance": self.rel_pos[ind].cuda(),
                        "position": self.base_pos[ind].cuda(),
                        "orientation": self.base_euler[ind].cuda()
                    })
            return np.array(info_redux)
            
    
    def _get_obs(self):
        return self.obs_buf
    
    def _get_info(self,env_idx=None):
        
        if env_idx is not None:
            if True:
                return np.array([{
                    "distance": self.rel_pos[env_idx][0].cuda(),
                    "position": self.base_pos[env_idx][0].cuda(),
                    "orientation": self.base_euler[env_idx][0].cuda(),
                }])

        return {
            "distance": self.rel_pos.cuda(),
            "position": self.base_pos.cuda(),
            "orientation": self.base_euler.cuda(),
        }

    

    def get_privileged_observations(self):
        return None

    def reset_idx(self, env_id):
        if len(env_id) == 0:
            return

        # reset base
        self.base_pos[env_id] = self.base_init_pos#torch.rand((len(env_id), 3), device=self.device, dtype=gs.tc_float)#self.base_init_pos
        self.last_base_pos[env_id] = self.base_init_pos#self.base_pos[env_id].clone()
        self.rel_pos = self.commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos
        self.base_quat[env_id] = self.base_init_quat.reshape(1, -1)
        self.drone.set_pos(self.base_pos[env_id], zero_velocity=True, envs_idx=env_id)
        self.drone.set_quat(self.base_quat[env_id], zero_velocity=True, envs_idx=env_id)
        self.base_lin_vel[env_id] = 0
        self.base_ang_vel[env_id] = 0
        self.drone.zero_all_dofs_velocity(env_id)

        self.obs_buf = torch.cat(
        [
            torch.clip(self.rel_pos * self.obs_scales["rel_pos"], -1, 1),
            self.base_quat,
            torch.clip(self.base_lin_vel * self.obs_scales["lin_vel"], -1, 1),
            torch.clip(self.base_ang_vel * self.obs_scales["ang_vel"], -1, 1),
            self.last_actions,
        ],
        axis=-1,
        )


        # reset buffers
        self.last_actions[env_id] = 0.0
        self.episode_length_buf[env_id] = 0
        self.reset_buf[env_id] = True
        #self.crash_condition[env_id]=False
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_id]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][env_id] = 0.0

        self._resample_commands(env_id)

    def reset(self,env_id=None):
        if env_id is not None:
        
            self.reset_buf[env_id]=True
            self.reset_idx(env_id)
        else: 
            self.crash_condition[:]=False
            self.reset_buf[:] = True
            self.reset_idx(torch.arange(self.num_envs, device=self.device))
        if env_id is not None:
            return self.obs_buf[env_id], self._get_info_redux(env_id)
        else: 
            return self.obs_buf, self._get_info()




    # ------------ reward functions----------------
    def _reward_target(self):
        target_rew = torch.sum(torch.square(self.last_rel_pos), dim=1) - torch.sum(torch.square(self.rel_pos), dim=1)
        nans=any(target_rew.isnan())
        if nans:
            print(target_rew.isnan())
            target_rew[target_rew.isnan()]=0
        return target_rew

    def _reward_smooth(self):
        smooth_rew = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        nans=any(smooth_rew.isnan())
        if nans:
            print(smooth_rew.isnan())
            smooth_rew[smooth_rew.isnan()]=0
        return smooth_rew

    def _reward_yaw(self):
        yaw = self.base_euler[:, 2]
        yaw = torch.where(yaw > 180, yaw - 360, yaw) / 180 * 3.14159  # use rad for yaw_reward
        yaw_rew = torch.exp(self.reward_cfg["yaw_lambda"] * torch.abs(yaw))
        nans=any(yaw_rew.isnan())
        if nans:
            print(yaw_rew.isnan())
            yaw_rew[yaw_rew.isnan()]=0
        return yaw_rew

    def _reward_angular(self):
        angular_rew = torch.norm(self.base_ang_vel / 3.14159, dim=1)
        nans=any(angular_rew.isnan())
        if nans:
            print(angular_rew.isnan())
            angular_rew[angular_rew.isnan()]=0
        return angular_rew

    def _reward_crash(self):
        crash_rew = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        crash_rew[self.crash_condition] = 1
        nans=any(crash_rew.isnan())
        if nans:
            print(crash_rew.isnan())
            crash_rew[crash_rew.isnan()]=0
        return crash_rew
    

