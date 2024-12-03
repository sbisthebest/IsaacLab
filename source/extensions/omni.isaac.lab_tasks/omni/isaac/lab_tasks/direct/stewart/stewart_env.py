import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.sim.spawners.materials import RigidBodyMaterialCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg, PhysxCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab_assets import ISAACLAB_ASSETS_DATA_DIR
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import quat_apply, quat_inv

@configclass
class StewartManipulationEnvCfg(DirectRLEnvCfg):
    # task config
    episode_length_s = 10.
    decimation = 2
    action_space = 3
    observation_space = 9
    state_space = 0

    # simulation config
    sim = SimulationCfg(
        dt = 1.0/120.0,
        render_interval = decimation,
        gravity = (0.0, 0.0, -9.81),
        use_fabric = True,
        disable_contact_processing = False,

        physx = PhysxCfg(
            solver_type = 1,
            enable_stabilization = True,
        )
    )

    # ground config
    ground_cfg = GroundPlaneCfg(
        usd_path = f"{ISAAC_NUCLEUS_DIR}/Environments/Grid/default_environment.usd",
        color = (0., 0., 0.),
        physics_material = RigidBodyMaterialCfg(
            static_friction = 1.0,
            dynamic_friction = 1.0,
            restitution = 0.0,
            friction_combine_mode = 'average',
            restitution_combine_mode = 'min',
        )
    )

    # robot config
    robot_cfg = ArticulationCfg(
        prim_path = "/World/envs/env_.*/Robot",
        spawn = sim_utils.UsdFileCfg(
            usd_path = f"{ISAACLAB_ASSETS_DATA_DIR}/Personal/Stewart.usd",
            activate_contact_sensors = False,
            rigid_props = sim_utils.RigidBodyPropertiesCfg(
                disable_gravity = False,
            ),
            articulation_props = sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count = 6,
                solver_velocity_iteration_count = 3
            ),
        ),
        init_state = ArticulationCfg.InitialStateCfg(
            joint_pos = {
                "upper_leg_joint00" : 0.707,
                "upper_leg_joint01" : 0.,
                "upper_leg_joint10" : 0.707,
                "upper_leg_joint11" : 0.,
                "upper_leg_joint20" : 0.707,
                "upper_leg_joint21" : 0.,
                "lower_leg_joint0" : 1.507,
                "lower_leg_joint1" : 1.507,
                "lower_leg_joint2" : 1.507,
                "foot_joint0" : 0.707,
                "foot_joint1" : 0.707,
                "foot_joint2" : 0.707,
            },
            pos = (0.0, 0.0, 0.3),
            rot = (1.0, 0.0, 0.0, 0.0),
        ),
        actuators = {
            "joint": ImplicitActuatorCfg(
                joint_names_expr=["foot_joint[0-9]"],
                effort_limit=5.0,
                velocity_limit=10.0,
                stiffness = 1.0,
                damping = 0.0,
                friction = 0.1,
            ),
        },
    )

    # ball config
    ball_cfg = RigidObjectCfg(
        prim_path = "/World/envs/env_.*/Ball",
        spawn = sim_utils.SphereCfg(
            radius = 0.02,
            rigid_props = sim_utils.RigidBodyPropertiesCfg(),
            mass_props = sim_utils.MassPropertiesCfg(mass = 0.04),
            collision_props = sim_utils.CollisionPropertiesCfg(),
            visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.1),
            physics_material = RigidBodyMaterialCfg(
                static_friction = 0.3,
                dynamic_friction = 0.3,
                restitution = 0.0,
                friction_combine_mode = 'average',
                restitution_combine_mode = 'min',
            )
        ),
        init_state = RigidObjectCfg.InitialStateCfg(
            pos = (0., 0., 0.5),
            rot = (1., 0., 0., 0.),
            lin_vel = (0., 0., 0.),
            ang_vel = (0., 0., 0.)
        ),
        collision_group = 0,
        debug_vis = False
    )

    # scene config
    scene = InteractiveSceneCfg(
        num_envs = 1024,
        env_spacing = 1.0,
    )

    action_scale = 1.

    # robot spec
    plane_length = 0.18
    plane_height = 0.02
    upper_leg_length = 0.1
    lower_leg_lenth = 0.1
    foot_height = 0.005

class StewartManipulationEnv(DirectRLEnv):
    cfg : StewartManipulationEnvCfg

    def __init__(self, cfg : StewartManipulationEnvCfg, render_mode : str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.env_origins = self.scene.env_origins

        self.ball_targets = torch.zeros((self.num_envs, 2)).to(self.sim.device)
        self.ball_position = self.ball.data.body_pos_w[:, 0, :] - self.env_origins
        self.ball_velocity = self.ball.data.body_lin_vel_w[:, 0, :]
        self.plane_position = self.robot.data.body_pos_w[:, 0, :] - self.env_origins
        self.plane_quaternion = self.robot.data.body_quat_w[:, 0, :]

        self.rel_pos = torch.zeros((self.num_envs, 2)).to(self.sim.device)
        self.rel_vel = torch.zeros((self.num_envs, 2)).to(self.sim.device)
        self.ball_to_targets = torch.zeros((self.num_envs, 2)).to(self.sim.device)
        self.robot_dof_targets = torch.zeros((self.num_envs, 3)).to(self.sim.device)
        self.latent_height = torch.zeros((self.num_envs, 1)).to(self.sim.device)

        self.succeed_env_ids = torch.zeros((self.num_envs), dtype=torch.bool).to(self.sim.device)
        self.terminated_env_ids = torch.zeros((self.num_envs), dtype=torch.bool).to(self.sim.device)

        self.success_env_count = 0
        self.failed_env_count = 0

    def _setup_scene(self):
        # Add robot to scene
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        # Add ball to scene
        self.ball = RigidObject(self.cfg.ball_cfg)
        self.scene.rigid_objects["ball"] = self.ball

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=self.cfg.ground_cfg)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions):
        self.actions = actions.clone()
        self.robot_dof_targets[:] = self.actions * self.cfg.action_scale

    def _apply_action(self):
        # breakpoint()
        self.robot.set_joint_position_target(self.robot_dof_targets, joint_ids=[9, 10, 11])

    def _get_observations(self):
        # breakpoint()
        obs = torch.cat(
            (
                self.ball_to_targets,
                # self.rel_pos,
                self.rel_vel,
                self.ball_targets,
                self.robot_dof_targets,
            ),
            dim=-1,
        )

        observations = {"policy" : obs}

        return observations

    def _get_rewards(self) -> torch.Tensor:

        self.ball_position = self.ball.data.body_pos_w[:, 0, :] - self.env_origins
        self.ball_velocity = self.ball.data.body_lin_vel_w[:, 0, :]
        self.plane_position = self.robot.data.body_pos_w[:, 0, :] - self.env_origins
        self.plane_quaternion = self.robot.data.body_quat_w[:, 0, :]

        (
            self.ball_to_targets[:],
            self.rel_pos[:],
            self.rel_vel[:],
            self.latent_height[:],
            self.succeed_env_ids[:],
            self.terminated_env_ids[:],
        )=compute_intermediate_values(
            self.ball_targets,
            self.ball_position,
            self.ball_velocity,
            self.plane_position,
            self.plane_quaternion,
            self.robot_dof_targets,
            self.cfg.plane_length,
            self.cfg.lower_leg_lenth,
        )

        self.success_env_count += torch.sum(self.succeed_env_ids)
        self.failed_env_count += torch.sum(self.terminated_env_ids)

        if torch.sum(self.succeed_env_ids) > 0:
            print(f"success ratio : {self.success_env_count / (self.success_env_count + self.failed_env_count) * 100}%")

        total_reward = compute_rewards(self.ball_to_targets,
                                       self.rel_vel,
                                       self.succeed_env_ids,
                                       self.terminated_env_ids,
                                       self.episode_length_buf,
                                       self.max_episode_length,
                                       )

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # failed = (torch.max(torch.abs(self.rel_pos), dim=-1)[0] >= self.cfg.plane_length/2)
        # succeed = (torch.norm(self.ball_to_targets, dim=-1) < 2e-2) & (torch.norm(self.rel_vel, dim=-1) < 1e-3)

        terminated = self.terminated_env_ids

        return terminated, time_out

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.ball._ALL_INDICES

        super()._reset_idx(env_ids)

        init_joint_pos, init_stewart_pos, init_ball_pos = self._reset_stewart(env_ids)

        init_stewart_pos[:, 0:3] += self.env_origins[env_ids]
        init_stewart_vel = torch.zeros((len(env_ids), 6), device=self.sim.device)
        init_joint_vel = torch.zeros_like(init_joint_pos)

        self.robot.set_joint_position_target(init_joint_pos, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(init_joint_pos, init_joint_vel, env_ids=env_ids)
        self.robot.write_root_pose_to_sim(init_stewart_pos, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(init_stewart_vel, env_ids=env_ids)

        # init_ball_pos = self.ball.data.default_root_state
        init_ball_pos[:, 0:3] += self.env_origins[env_ids]
        init_ball_vel = torch.zeros((len(env_ids), 6), device=self.sim.device)

        self.ball.write_root_pose_to_sim(init_ball_pos, env_ids=env_ids)
        self.ball.write_root_velocity_to_sim(init_ball_vel, env_ids=env_ids)

        self.ball_targets[env_ids, :] = self._reset_target(env_ids)

    def _reset_stewart(self, env_ids):
        upper_joint = torch.rand((len(env_ids), 1)).to(self.sim.device)
        inter_joint = torch.zeros((len(env_ids), 1)).to(self.sim.device)
        lower_joint = 2*upper_joint
        foot_joint = upper_joint

        init_joint_pos = torch.hstack((upper_joint.repeat(1, 3),
                                       inter_joint.repeat(1, 3),
                                       lower_joint.repeat(1, 3),
                                       foot_joint.repeat(1, 3)))

        plane_height = self.cfg.plane_height/2 \
                     + self.cfg.upper_leg_length * torch.cos(upper_joint) \
                     + self.cfg.lower_leg_lenth * torch.cos(lower_joint) \
                     + self.cfg.foot_height/2

        init_stewart_pos = torch.hstack((torch.zeros(len(env_ids), 2, device=self.sim.device),
                                         plane_height,
                                         torch.ones(len(env_ids), 1, device=self.sim.device),
                                         torch.zeros(len(env_ids), 3, device=self.sim.device)))

        init_ball_pos = torch.hstack((0.8 * self.cfg.plane_length * torch.rand((len(env_ids), 2), device=self.sim.device) - 0.4 * self.cfg.plane_length,
                                      plane_height + 0.1,
                                      torch.ones(len(env_ids), 1, device=self.sim.device),
                                      torch.zeros(len(env_ids), 3, device=self.sim.device)))

        return init_joint_pos, init_stewart_pos, init_ball_pos

    def _reset_target(self, env_ids):

        new_target = 0.8 * self.cfg.plane_length * torch.rand((len(env_ids), 2)).to(self.sim.device) - 0.4 * self.cfg.plane_length
        # new_target = torch.zeros((len(env_ids), 2)).to(self.sim.device)

        return new_target

@torch.jit.script
def compute_intermediate_values(
    ball_targets : torch.Tensor,
    ball_pos : torch.Tensor,
    ball_vel : torch.Tensor,
    plane_pos : torch.Tensor,
    plane_quat : torch.Tensor,
    joint_angle : torch.Tensor,
    plane_length : float,
    lower_leg_length : float,
):
    _z_axis = torch.zeros_like(plane_pos)
    _z_axis[..., 2] = 1

    plane_norm_vector = quat_apply(plane_quat, _z_axis)

    # Calculate relative position
    plane_to_ball = ball_pos - plane_pos
    ball_projected = plane_to_ball - torch.diag(torch.inner(plane_to_ball, plane_norm_vector)).reshape(-1, 1) * plane_to_ball / torch.norm(plane_to_ball, dim=-1, keepdim=True)
    _rel_pos = quat_apply(quat_inv(plane_quat), ball_projected)
    rel_pos = _rel_pos[..., 0:2]
    ball_to_targets = ball_targets - rel_pos

    # Calculate relative velocity
    ball_vel_projected = ball_vel - torch.diag(torch.inner(ball_vel, plane_norm_vector)).reshape(-1, 1) * ball_vel / torch.norm(ball_vel, dim=-1, keepdim=True)
    _rel_vel = quat_apply(quat_inv(plane_quat), ball_vel_projected)
    rel_vel = _rel_vel[..., 0:2]

    # Calculate latent height
    latent_height = lower_leg_length * torch.sum(torch.cos(joint_angle), dim=-1, keepdim=True)/3

    # Calculate succeed enviornments
    succeed_env_ids = (torch.norm(ball_to_targets, dim=-1) < 1e-2) & (torch.norm(rel_vel, dim=-1) < 1e-3)

    # Calculate terminated environments
    terminated_env_ids = torch.max(torch.abs(rel_pos), dim=-1)[0] >= plane_length/2

    return (
        ball_to_targets,
        rel_pos,
        rel_vel,
        latent_height,
        succeed_env_ids,
        terminated_env_ids,
    )

@torch.jit.script
def compute_rewards(
    ball_to_targets : torch.Tensor,
    rel_vel : torch.Tensor,
    succeed_env_ids : torch.Tensor,
    terminated_env_ids : torch.Tensor,
    episode_length_buf : torch.Tensor,
    max_episode_length : float,
):

    pos_reward = 1./(1. + 100 * torch.norm(ball_to_targets, dim=-1))
    vel_reward = 1./(1. + 100 * torch.norm(rel_vel, dim=-1))

    # alive_reward = torch.ones_like(pos_reward)

    total_reward = pos_reward + vel_reward

    total_reward = torch.where(succeed_env_ids, 10 * torch.ones_like(pos_reward), total_reward)

    total_reward = torch.where(terminated_env_ids, -5 * torch.ones_like(total_reward), total_reward)
    # total_reward = torch.where(episode_length_buf >= max_episode_length - 1, -1 * torch.ones_like(total_reward), total_reward)

    return total_reward
