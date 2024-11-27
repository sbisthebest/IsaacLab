"""
    Non-prehensile Manipulation environment.
"""

import torch
import math

from collections.abc import Sequence

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
    box_height, box_width, box_length = 0.05, 0.2, 0.4

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
    robot_height, robot_radius = 0.02, 0.05

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

    # task config
    decimation = 2
    episode_length_s = 10.0
    action_space = 3
    observation_space = 23
    state_space = 0
    action_scale = 5
    termination_dist = 10.

class NonPrehensileManipulationEnv(DirectRLEnv):
    cfg : NonPrehensileManipulationEnvCfg

    def __init__(self, cfg : NonPrehensileManipulationEnvCfg, render_mode : None | str = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        self.env_origins = self.scene.env_origins
        self.action_scale = self.cfg.action_scale

        self.targets = torch.zeros((self.num_envs, 3)).to(self.device)
        self.targets += self.env_origins

        self.box_geometry = torch.Tensor([self.cfg.box_width,
                                          self.cfg.box_length]).repeat(self.num_envs, 1).to(self.device)

        self.min_box_robot_dist = self.cfg.robot_radius + math.sqrt(self.cfg.box_width**2 + self.cfg.box_length**2)

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
        self.forces = self.action_scale * self.actions[:, 0:3].reshape(self.num_envs, 1, 3)
        self.torques = torch.zeros_like(self.forces)

    def _apply_action(self) -> None:
        self.robot.set_external_force_and_torque(forces=self.forces,
                                                 torques=self.torques,)
        
    def _compute_intermeidate_values(self):
        self.box_pos = self.box.data.root_pos_w
        self.box_lin_vel_w = self.box.data.root_lin_vel_w
        self.box_ang_vel_w = self.box.data.root_ang_vel_w
        self.box_quat = self.box.data.root_quat_w
        
        self.robot_pos = self.robot.data.root_pos_w
        self.robot_lin_vel_w = self.robot.data.root_lin_vel_w
        self.robot_ang_vel_w = self.robot.data.root_ang_vel_w
        self.robot_quat = self.robot.data.root_quat_w
        
        (
            self.box_ori,
            self.robot_ori,
            self.dist_box_to_target,
            self.dist_robot_to_box,
            self.normvec_box_to_target,
            self.normvec_robot_to_box
        ) = compute_intermediate_values(
            self.box_pos,
            self.box_quat,
            self.robot_pos,
            self.robot_quat,
            self.targets,
        )

        
    def _get_observations(self) -> dict:
        
        obs = torch.cat(
            (
                self.normvec_box_to_target,
                self.normvec_robot_to_box,
                self.dist_box_to_target,
                self.dist_robot_to_box,
                self.box_geometry,
                self.box_lin_vel_w,
                self.box_ang_vel_w,
                self.box_ori,
                self.robot_lin_vel_w,
                self.actions,
            ),
            dim=-1,
        )

        observations = {"policy": obs}

        return observations
    
    def _get_rewards(self) -> torch.Tensor:
        
        total_reward = compute_rewards(self.dist_box_to_target,
                                       self.dist_robot_to_box,
                                       self.box_lin_vel_w,
                                       self.box_ang_vel_w,
                                       self.box_ori,
                                       )

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        self._compute_intermeidate_values()

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = self.dist_robot_to_box.squeeze() > self.cfg.termination_dist

        return terminated, time_out
    
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.box._ALL_INDICES 
        
        super()._reset_idx(env_ids)

        self._box_pos = torch.hstack((self.cfg.scene.env_spacing/2*torch.rand((self.num_envs, 2)) - self.cfg.scene.env_spacing/4, 
                                      self.cfg.box_height/2*torch.ones((self.num_envs, 1))))
        self._robot_pos = torch.hstack((self.cfg.scene.env_spacing/2*torch.rand((self.num_envs, 2)) - self.cfg.scene.env_spacing/4,
                                        self.cfg.robot_height/2*torch.ones((self.num_envs, 1))))

        # Handle robot-box collision
        self._box_robot_dist = torch.abs(self._box_pos - self._robot_pos)
        self._robot_pos = torch.where(self._box_robot_dist < self.min_box_robot_dist,
                                      self._robot_pos + torch.hstack((2*self.min_box_robot_dist*torch.ones(self.num_envs, 2), torch.zeros(self.num_envs, 1))),
                                      self._robot_pos)
        
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

        self._compute_intermeidate_values()

@torch.jit.script
def compute_rewards(
    dist_box_to_target : torch.Tensor,
    dist_robot_to_box : torch.Tensor,
    box_lin_vel : torch.Tensor,
    box_ang_vel : torch.Tensor,
    box_ori : torch.Tensor,
):

    pos_error_reward = 1./(1. + dist_box_to_target.squeeze())
    ori_error_reward = 1./(1. + torch.norm(box_ori, dim=-1))

    # guidance = 1./(1. + dist_robot_to_box.squeeze())

    total_reward = pos_error_reward * ori_error_reward
    
    return total_reward

@torch.jit.script
def compute_intermediate_values(
    box_pos : torch.Tensor,
    box_quat : torch.Tensor,
    robot_pos : torch.Tensor,
    robot_quat : torch.Tensor,
    target : torch.Tensor,
):
    box_ori = 2*torch.atan2(box_quat[:, 3:4], box_quat[:, 0:1])
    robot_ori = 2*torch.atan2(robot_quat[:, 3:4], robot_quat[:, 0:1])

    dist_box_to_target = torch.norm(target - box_pos, dim=-1, keepdim=True)
    dist_robot_to_box = torch.norm(box_pos - robot_pos, dim=-1, keepdim=True)

    normvec_box_to_target = torch.where(dist_box_to_target > 1e-3, (target - box_pos) / torch.norm(target - box_pos), torch.zeros_like(target))
    normvec_robot_to_box = torch.where(dist_robot_to_box > 1e-3, (box_pos - robot_pos) / torch.norm(box_pos - robot_pos), torch.zeros_like(target))

    return (
        box_ori,
        robot_ori,
        dist_box_to_target,
        dist_robot_to_box,
        normvec_box_to_target,
        normvec_robot_to_box,
    )