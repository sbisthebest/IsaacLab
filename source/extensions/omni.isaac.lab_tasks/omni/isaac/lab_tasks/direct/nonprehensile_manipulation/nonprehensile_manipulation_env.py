"""
    Non-prehensile Manipulation environment.
"""

import torch
import math

from collections.abc import Sequence

from gymnasium.spaces import MultiDiscrete
import numpy as np

import omni.isaac.lab.sim as sim_utils

from omni.isaac.lab.sim.spawners.materials import RigidBodyMaterialCfg
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg, PhysxCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

@configclass
class NonPrehensileManipulationEnvCfg(DirectRLEnvCfg):

    # task config
    decimation = 2
    episode_length_s = 30.0
    action_bins = 11
    # action_space = MultiDiscrete(np.array([action_bins, action_bins]))
    action_space = [{action_bins}, {action_bins}]
    # action_space = 2
    observation_space = 23
    state_space = 0
    action_scale = 5

    # simulation config
    sim = SimulationCfg(
        dt=1.0/120.0,
        render_interval=2,
        gravity=(0.0, 0.0, -9.81),
        use_fabric=True, # False to update physics parameters in real-time

        physx = PhysxCfg(
            solver_type=1,
            enable_stabilization=True,
        )
    )

    # ground config
    ground_cfg = GroundPlaneCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Grid/default_environment.usd",
        color=(0., 0., 0.),
        physics_material=RigidBodyMaterialCfg(
            static_friction=0.3,
            dynamic_friction=0.3,
            restitution=0.0,
            friction_combine_mode='average',
            restitution_combine_mode='max',
        )
    )

    # box config
    box_height, box_width, box_length = 0.05, 0.1, 0.15

    box_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Box",
        spawn=sim_utils.CuboidCfg(
            size=(box_width, box_length, box_height),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.4, 0.3), metallic=0.1),
            physics_material=RigidBodyMaterialCfg(
                static_friction=0.3,
                dynamic_friction=0.3,
                restitution=0.0,
                friction_combine_mode='average',
                restitution_combine_mode='max',
            )
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0., 0., box_height/2),
            rot=(1., 0., 0., 0.),
            lin_vel=(0., 0., 0.),
            ang_vel=(0., 0., 0.)
        ),
        collision_group=0,  # -1 for global collision, 0 for local collision
        debug_vis=False     # Enable debug visualization for the asset
    )

    # robot config
    robot_height, robot_radius = 0.01, 0.02

    robot_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.CylinderCfg(
            radius=robot_radius,
            height=robot_height,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.6, 0.8), metallic=1.0),
            physics_material=RigidBodyMaterialCfg(
                static_friction=0.3,
                dynamic_friction=0.3,
                restitution=0.0,
                friction_combine_mode='average',
                restitution_combine_mode='max',
            )
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, robot_height/2),
            rot=(1.0, 0., 0., 0.),
            lin_vel=(0., 0., 0.),
            ang_vel=(0., 0., 0.)
        ),
        collision_group=0,
        debug_vis=False
    )

    # scene config
    scene = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=4.0
    )



class NonPrehensileManipulationEnv(DirectRLEnv):
    cfg : NonPrehensileManipulationEnvCfg

    def __init__(self, cfg : NonPrehensileManipulationEnvCfg, render_mode : None | str = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.action_bins = self.cfg.action_bins
        self.env_origins = self.scene.env_origins
        self.action_scale = self.cfg.action_scale

        self.targets = torch.zeros((self.num_envs, 3)).to(self.device)

        self.box_geometry = torch.Tensor([self.cfg.box_width,
                                          self.cfg.box_length]).repeat(self.num_envs, 1).to(self.device)

        self.min_box_robot_dist = self.cfg.robot_radius + math.sqrt(self.cfg.box_width**2 + self.cfg.box_length**2)

        # Initialize some variables
        self.prev_dist_box_to_target = torch.zeros((self.num_envs, 1)).to(self.device)
        self.prev_dist_robot_to_box = torch.zeros((self.num_envs, 1)).to(self.device)
        self.dist_box_to_target = torch.zeros((self.num_envs, 1)).to(self.device)
        self.dist_robot_to_box = torch.zeros((self.num_envs, 1)).to(self.device)
        self.prev_box_2d_lin_acc = torch.zeros((self.num_envs, 2)).to(self.device)
        self.box_2d_lin_acc = torch.zeros((self.num_envs, 2)).to(self.device)
        self.prev_robot_2d_lin_acc = torch.zeros((self.num_envs, 2)).to(self.device)
        self.robot_2d_lin_acc = torch.zeros((self.num_envs, 2)).to(self.device)

    def _setup_scene(self):
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=self.cfg.ground_cfg)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        # add box to scene
        self.box = RigidObject(self.cfg.box_cfg)
        self.scene.rigid_objects["box"] = self.box

        # add robot to scene
        self.robot = RigidObject(self.cfg.robot_cfg)
        self.scene.rigid_objects["robot"] = self.robot

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        # breakpoint()
        self.actions = (2 * self.actions / (self.action_bins - 1)) - 1
        self.forces = torch.hstack((self.action_scale * self.actions, torch.zeros((self.num_envs, 1)).to(self.device))).reshape(self.num_envs, 1, 3)
        self.torques = torch.zeros_like(self.forces)

    def _apply_action(self) -> None:
        self.robot.set_external_force_and_torque(forces=self.forces,
                                                 torques=self.torques,)
        self.robot.write_data_to_sim()

    def _compute_intermediate_values(self):

        self.box_pos = self.box.data.root_pos_w - self.env_origins
        self.box_lin_vel_w = self.box.data.root_lin_vel_w
        self.box_ang_vel_w = self.box.data.root_ang_vel_w
        self.box_quat = self.box.data.root_quat_w
        self.box_lin_acc_w = self.box.data.body_lin_acc_w
        self.box_ang_acc_w = self.box.data.body_ang_acc_w

        self.robot_pos = self.robot.data.root_pos_w - self.env_origins
        self.robot_lin_vel_w = self.robot.data.root_lin_vel_w
        self.robot_ang_vel_w = self.robot.data.root_ang_vel_w
        self.robot_quat = self.robot.data.root_quat_w
        self.robot_lin_acc_w = self.robot.data.body_lin_acc_w
        self.robot_ang_acc_w = self.robot.data.body_ang_acc_w

        self.prev_dist_box_to_target[:] = self.dist_box_to_target
        self.prev_dist_robot_to_box[:] = self.dist_robot_to_box
        self.prev_box_2d_lin_acc[:] = self.box_2d_lin_acc
        self.prev_robot_2d_lin_acc[:] = self.robot_2d_lin_acc
    
        (
            self.box_ori,
            self.box_2d_pos,
            self.box_2d_lin_vel,
            self.box_2d_lin_acc,
            self.box_2d_ang_vel,
            self.box_2d_ang_acc,
            self.robot_ori,
            self.robot_2d_pos,
            self.robot_2d_lin_vel,
            self.robot_2d_lin_acc,
            self.robot_2d_ang_vel,
            self.robot_2d_ang_acc,
            self.robot_speed,
            self.dist_box_to_target,
            self.ang_box_to_target,
        ) = compute_intermediate_values(
            self.box_pos,
            self.box_quat,
            self.box_lin_vel_w,
            self.box_lin_acc_w,
            self.box_ang_vel_w,
            self.box_ang_acc_w,
            self.robot_pos,
            self.robot_quat,
            self.robot_lin_vel_w,
            self.robot_lin_acc_w,
            self.robot_ang_vel_w,
            self.robot_ang_acc_w,
            self.targets,
        )
        
    def _get_observations(self) -> dict:

        obs = torch.cat(
            (
                # self.normvec_box_to_target,
                # self.normvec_robot_to_box,
                # self.dist_box_to_target,
                # self.dist_robot_to_box,
                self.box_geometry,
                self.box_ori,
                self.box_2d_pos,
                self.box_2d_lin_vel,
                self.box_2d_lin_acc,
                self.box_2d_ang_vel,
                self.box_2d_ang_acc,
                # self.robot_speed,
                # self.robot_ori,
                self.robot_2d_pos,
                self.robot_2d_lin_vel,
                self.robot_2d_lin_acc,
                self.robot_2d_ang_vel,
                self.robot_2d_ang_acc,
                self.targets,
            ),
            dim=-1,
        )

        observations = {"policy": obs}

        return observations

    def _get_rewards(self) -> torch.Tensor:

        total_reward = compute_rewards(self.dist_box_to_target,
                                       self.ang_box_to_target,
                                       self.robot_speed,
                                       self.robot_2d_pos,
                                       self.cfg.scene.env_spacing / 2,
                                       self.episode_length_buf,
                                       self.max_episode_length,
                                       self.num_envs,
                                       self.device,
                                       )

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        self._compute_intermediate_values()

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = (self.box_pos[:, 2] > self.cfg.box_height * 0.6) | \
                     (self.robot_pos[:, 2] > self.cfg.robot_height * 0.6) | \
                     (self.dist_box_to_target[:, 0] < 1e-2) | \
                     (torch.max(torch.abs(self.robot_pos[:, 0:2])) > self.cfg.scene.env_spacing / 2)

        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.box._ALL_INDICES

        self.robot.reset(env_ids)
        self.box.reset(env_ids)
        super()._reset_idx(env_ids)

        self.targets = torch.hstack((self.cfg.scene.env_spacing/2 * torch.rand((self.num_envs, 2)) - self.cfg.scene.env_spacing/4,
                                     2 * torch.pi * torch.rand((self.num_envs, 1)) - torch.pi)).to(self.device)

        self._box_pos = torch.hstack((self.cfg.scene.env_spacing/2*torch.rand((self.num_envs, 2)) - self.cfg.scene.env_spacing/4,
                                      self.cfg.box_height/2*torch.ones((self.num_envs, 1))))
        self._robot_pos = torch.hstack((self.cfg.scene.env_spacing/2*torch.rand((self.num_envs, 2)) - self.cfg.scene.env_spacing/4,
                                        self.cfg.robot_height/2*torch.ones((self.num_envs, 1))))

        self._box_ori = 2 * torch.pi * torch.rand((self.num_envs, 1)) - torch.pi
        self._box_quat = torch.hstack((torch.cos(self._box_ori/2),
                                       torch.zeros((self.num_envs, 2)),
                                       torch.sin(self._box_ori/2)))
        self._robot_quat = torch.tensor([1, 0, 0, 0], dtype=torch.float32).repeat(self.num_envs, 1)

        self._box_lin_vel = torch.zeros((self.num_envs, 3))
        self._box_ang_vel = torch.zeros((self.num_envs, 3))
        self._robot_lin_vel = torch.zeros((self.num_envs, 3))
        self._robot_ang_vel = torch.zeros((self.num_envs, 3))

        self.box_root_state = torch.cat(
            (
                self._box_pos + self.env_origins[:, 0:3].cpu(),
                self._box_quat,
                self._box_lin_vel,
                self._box_ang_vel,
            ),
            dim=-1,
        ).to(self.device)

        self._robot_root_state = torch.cat(
            (
                self._robot_pos + self.env_origins[:, 0:3].cpu(),
                self._robot_quat,
                self._robot_lin_vel,
                self._robot_ang_vel,
            ),
            dim=-1,
        ).to(self.device)

        self.box.write_root_state_to_sim(self.box_root_state[env_ids], env_ids)
        self.robot.write_root_state_to_sim(self._robot_root_state[env_ids], env_ids)

        self._compute_intermediate_values()

@torch.jit.script
def compute_rewards(
    dist_box_to_target : torch.Tensor,
    ang_box_to_target : torch.Tensor,
    robot_speed : torch.Tensor,
    robot_2d_pos : torch.Tensor,
    boundary : float,
    episode_length_buf : torch.Tensor,
    max_episode_length : float,
    num_envs : int,
    device : str,
):
    dist_box_to_target = dist_box_to_target.squeeze()
    ang_box_to_target = ang_box_to_target.squeeze()
    robot_speed = robot_speed.squeeze()

    dist_error_reward = 1 - dist_box_to_target / boundary
    ang_error_reward = 1 - ang_box_to_target / torch.pi
    vel_penalty = torch.where(robot_speed > 0.1, 0.01-robot_speed**2, torch.zeros_like(robot_speed))
    vel_penalty = torch.where(vel_penalty < -1., -torch.ones_like(vel_penalty), vel_penalty)

    total_reward = 0.1 * dist_error_reward + 0.04 * ang_error_reward + 0.001 * vel_penalty

    total_reward = torch.where(dist_box_to_target < 1e-2, 100 * torch.ones_like(total_reward), total_reward)
    total_reward = torch.where(torch.max(torch.abs(robot_2d_pos)) > boundary, -50 * torch.ones_like(total_reward), total_reward)
    total_reward = torch.where(episode_length_buf >= max_episode_length-1, -50*torch.ones_like(total_reward), total_reward)

    # if torch.any(dist_box_to_target < 1e-2):
    #     print("Success Episode!")

    return total_reward


@torch.jit.script
def compute_intermediate_values(
    box_pos : torch.Tensor,
    box_quat : torch.Tensor,
    box_lin_vel : torch.Tensor,
    box_lin_acc : torch.Tensor,
    box_ang_vel : torch.Tensor,
    box_ang_acc : torch.Tensor,
    # prev_box_2d_lin_acc : torch.Tensor,
    robot_pos : torch.Tensor,
    robot_quat : torch.Tensor,
    robot_lin_vel : torch.Tensor,
    robot_lin_acc : torch.Tensor,
    robot_ang_vel : torch.Tensor,
    robot_ang_acc : torch.Tensor,
    # prev_robot_2d_lin_acc : torch.Tensor,
    targets : torch.Tensor,
    # dt : float,
):
    # Split target into pos and angle
    targets_pos = targets[:, 0:2]
    targets_theta = targets[:, 2:3]

    # Transform 3d position vector into 2d vector
    box_2d_pos = box_pos[:, 0:2]
    box_2d_lin_vel = box_lin_vel[:, 0:2]
    box_2d_ang_vel = box_ang_vel[:, 2:3]
    box_2d_lin_acc = box_lin_acc[:, 0, 0:2]
    box_2d_ang_acc = box_ang_acc[:, 0, 2:3]
    robot_2d_pos = robot_pos[:, 0:2]
    robot_2d_lin_vel = robot_lin_vel[:, 0:2]
    robot_2d_ang_vel = robot_ang_vel[:, 2:3]
    robot_2d_lin_acc = robot_lin_acc[:, 0, 0:2]
    robot_2d_ang_acc = robot_ang_acc[:, 0, 2:3]

    # Calculate box & robot angle respect to the world frame
    box_theta = 2*torch.atan2(box_quat[:, 3:4], box_quat[:, 0:1])
    robot_theta = 2*torch.atan2(robot_quat[:, 3:4], robot_quat[:, 0:1])

    # Calcualte box & robot orientation information
    box_ori = torch.hstack((torch.cos(box_theta),
                            torch.sin(box_theta)))
    robot_ori = torch.hstack((torch.cos(robot_theta),
                              torch.sin(robot_theta)))

    # Calculate distance b/w robot-box-target
    dist_box_to_target = torch.norm(targets_pos - box_2d_pos, dim=-1, keepdim=True)
    # dist_robot_to_box = torch.norm(box_2d_pos - robot_2d_pos, dim=-1, keepdim=True)

    # Calculate direction b/w robot-box-target
    # normvec_box_to_target = torch.where(dist_box_to_target > 1e-5, (targets - box_2d_pos) / torch.norm(targets - box_2d_pos), torch.zeros_like(targets))
    # normvec_robot_to_box = torch.where(dist_robot_to_box > 1e-5, (box_2d_pos - robot_2d_pos) / torch.norm(box_2d_pos - robot_2d_pos), torch.zeros_like(targets))

    # Calculate the robot speed
    robot_speed = torch.norm(robot_lin_vel, dim=-1, keepdim=True)

    # Calculate angle error b/w box-target
    ang_box_to_target = torch.abs(targets_theta - robot_theta)

    # Calculate box 2d kinematics
    # box_jerk = torch.norm(prev_box_2d_lin_acc - box_2d_lin_acc, dim=-1)/dt

    # Calculate robot 2d kinematics
    # robot_jerk = torch.norm(prev_robot_2d_lin_acc - robot_2d_lin_acc, dim=-1)/dt

    return (
        box_ori,
        box_2d_pos,
        box_2d_lin_vel,
        box_2d_lin_acc,
        box_2d_ang_vel,
        box_2d_ang_acc,
        robot_ori,
        robot_2d_pos,
        robot_2d_lin_vel,
        robot_2d_lin_acc,
        robot_2d_ang_vel,
        robot_2d_ang_acc,
        robot_speed,
        dist_box_to_target,
        ang_box_to_target,
    )
