from __future__ import annotations

import math
import torch
from collections.abc import Sequence

from omni.isaac.lab_assets.cart_double_pendulum import CART_DOUBLE_PENDULUM_CFG

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sim.spawners.materials import RigidBodyMaterialCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.envs import DirectMARLEnv, DirectMARLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg, PhysxCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import sample_uniform
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

@configclass
class MACollaborativeManipulationEnvCfg(DirectMARLEnvCfg):
    
    # env
    decimation = 2
    episode_length_s = 20.0
    possible_agents = ["robot1", "robot2", "robot3", "robot4", "robot5", "robot6"]
    num_actions = {"robot1" : 3, "robot2" : 3, "robot3" : 3, "robot4" : 3, "robot5" : 3, "robot6" : 3}
    num_observations = {"robot1" : 36, "robot2" : 36, "robot3" : 36, "robot4" : 36, "robot5" : 36, "robot6" : 36}
    num_states = 36
    
    # simulation config
    sim: SimulationCfg = SimulationCfg(
        dt=1.0/120.0,
        render_interval=2,
        gravity=(0., 0., -9.81),
        use_fabric=True,
        
        physx = PhysxCfg(
            solver_type=1,
            enable_stabilization=True,
            gpu_found_lost_pairs_capacity=2**24,
            gpu_total_aggregate_pairs_capacity=2**24,
        )
    )
    
    # ground config
    ground_cfg: GroundPlaneCfg = GroundPlaneCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Grid/default_environment.usd",
        color=(0., 0., 0.),
        physics_material=RigidBodyMaterialCfg(
            static_friction=0.3,
            dynamic_friction=0.3,
            restitution=0.0,
            friction_combine_mode="average",
            restitution_combine_mode="max",
        )
    )
    
    # rod config
    rod_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 1.0, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.6),
            physics_material=RigidBodyMaterialCfg(
                static_friction=0.3,
                dynamic_friction=0.3,
                restitution=0.0,
                friction_combine_mode="average",
                restitution_combine_mode="max",
            )
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0., 0., 0.05),
            rot=(1., 0., 0., 0.),
            lin_vel=(0., 0., 0.),
            ang_vel=(0., 0., 0.)
        ),
        collision_group=0, # -1 for global collision, 0 for local collision
        debug_vis=False    # Enable debug visualization for the asset
    )

    # scene config
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=10.0
    )

class MACollaborativeManipulationEnv(DirectMARLEnv):
    cfg: MACollaborativeManipulationEnvCfg
    
    def __init__(self, cfg: MACollaborativeManipulationEnvCfg, render_mode: None | str = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Initialize state variable
        self.x = torch.zeros((self.num_envs, 6, 1)).to(self.device)
        
        # Initialize batch action
        self.batch_action_size = 3 * self.num_agents
        self.batch_action = torch.zeros(self.num_envs, self.batch_action_size, 1).to(self.device)
        
        # Save initial state
        # self.x = torch.zeros((self.num_envs, 6, 1)).to(self.device)
        self.init_x = self.x.clone()
        
        # Define state-space equation
        self.mu = 0.1
        self.M = 1
        self.I = 1
        
        self.r = ((1., 1.),
                  (1., -1.),
                  (-1., 1.),
                  (-1., -1.),
                  (0., 1.),
                  (1., 0.))
        
        self.r_obs = {"robot1" : torch.Tensor([1, 1]).repeat(self.num_envs, 1).to(self.device),
                      "robot2" : torch.Tensor([1, -1]).repeat(self.num_envs, 1).to(self.device),
                      "robot3" : torch.Tensor([-1, 1]).repeat(self.num_envs, 1).to(self.device),
                      "robot4" : torch.Tensor([-1, -1]).repeat(self.num_envs, 1).to(self.device),
                      "robot5" : torch.Tensor([0, 1]).repeat(self.num_envs, 1).to(self.device),
                      "robot6" : torch.Tensor([1, 0]).repeat(self.num_envs, 1).to(self.device)}
        
        self.r_obs_tensor = torch.Tensor([1, 1, 1, -1, -1, 1, -1, -1, 0, 1, 1, 0]).repeat(self.num_envs, 1).to(self.device)
        
        self.A = torch.Tensor([[0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 0, 1],
                               [0, 0, 0, -self.mu/self.M, 0, 0],
                               [0, 0, 0, 0, -self.mu/self.M, 0],
                               [0, 0, 0, 0, 0, -self.mu/self.M]]) \
                               .repeat(self.num_envs, 1, 1).to(self.device)
        
        self.B = torch.zeros((self.num_envs,
                              6,
                              self.batch_action_size)).to(self.device)
        
        self.T = torch.Tensor([[0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0],
                               [-self.mu/self.M, 0, 0, 0, 1],
                               [0, -self.mu/self.M, 0, -1, 0],
                               [0, 0, -self.mu/self.M, 0, 0]]) \
                               .repeat(self.num_envs, 1, 1).to(self.device)
        
        for env_id in range(self.num_envs):
            for robot_id in range(self.num_agents):
                self.B[env_id, :, 3*robot_id:3*robot_id+3] = \
                    torch.Tensor([[0, 0, 0],
                                  [0, 0, 0],
                                  [0, 0, 0],
                                  [1/self.M, 0, 0],
                                  [0, 1/self.M, 0],
                                  [-self.r[robot_id][1]/self.I, self.r[robot_id][0]/self.I, 1/self.I]]).to(self.device)

    def _setup_scene(self):
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=self.cfg.ground_cfg)
        
        # clone, filter and relicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        
        # add rod to scene
        self.rod = RigidObject(self.cfg.rod_cfg)
        self.scene.rigid_objects["rod"] = self.rod
        
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.5, 0.5, 0.5))
        light_cfg.func("/World/Light", light_cfg)
        
    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        
        # TO DO: modify actions into batch action
        self.prev_actions = self.actions
        self.actions = actions
        self.batch_action = torch.cat(list(self.actions.values()), dim=-1).reshape(self.num_envs, self.batch_action_size, 1)
    
    def _apply_action(self) -> None:
        
        self.f = calc_f(self.x)

        # Calculate state-space equation
        Ax = self.A @ self.x
        Bu = self.B @ self.batch_action
        
        k_1 = Ax + Bu + self.T @ self.f
        k_2 = Ax + self.physics_dt/2 * self.A @ k_1 + Bu + self.T @ calc_f(self.x + self.physics_dt/2 * k_1)
        k_3 = Ax + self.physics_dt/2 * self.A @ k_2 + Bu + self.T @ calc_f(self.x + self.physics_dt/2 * k_2)
        k_4 = Ax + self.physics_dt * self.A @ k_3 + Bu + self.T @ calc_f(self.x + self.physics_dt * k_3)
        
        self.x_dot = Ax + Bu + self.T @ self.f

        # Update state variable
        self.x += self.physics_dt/6 * (k_1 + 2*k_2 + 2*k_3 + k_4)
        
        # Clip theta(-pi ~ pi)
        self.x[:, 2, 0] = torch.where(self.x[:, 2, 0] >= torch.pi, self.x[:, 2, 0] - 2 * torch.pi, self.x[:, 2, 0])
        self.x[:, 2, 0] = torch.where(self.x[:, 2, 0] < -torch.pi, self.x[:, 2, 0] + 2 * torch.pi, self.x[:, 2, 0])

        self.rod_root_state = torch.cat([self.x[:, 0:2, 0] + self.scene.env_origins[:, 0:2],
                                     0.5 * torch.ones((self.num_envs, 1)).to(self.device),
                                     torch.cos(self.x[:, 2:3, 0]/2).to(self.device),
                                     torch.zeros((self.num_envs, 2)).to(self.device),
                                     torch.sin(self.x[:, 2:3, 0]/2),
                                     self.x[:, 3:5, 0],
                                     torch.zeros((self.num_envs, 3)).to(self.device),
                                     self.x[:, 5:6, 0]], dim=-1)

        self.rod.write_root_state_to_sim(self.rod_root_state)
        
    def _get_states(self) -> torch.Tensor:
        
        state = torch.cat(
            (
                self.x.squeeze(dim=-1),
                self.batch_action.squeeze(dim=-1),
                self.r_obs_tensor,
            ),
            dim=-1,
        )
        
        return state

    def _get_observations(self) -> dict[str, torch.Tensor]:
        
        observations = {
            "robot1" : torch.cat(
                (
                    self.x.reshape(self.num_envs, -1),
                    self.batch_action.reshape(self.num_envs, -1),
                    self.r_obs_tensor,
                    # self.actions["robot1"],
                    # self.r_obs["robot1"],
                ),
                dim=-1,
            ),
            "robot2" : torch.cat(
                (
                    self.x.reshape(self.num_envs, -1),
                    self.batch_action.reshape(self.num_envs, -1),
                    self.r_obs_tensor,
                    # self.actions["robot2"],
                    # self.r_obs["robot2"],
                ),
                dim=-1,
            ),
            "robot3" : torch.cat(
                (
                    self.x.reshape(self.num_envs, -1),
                    self.batch_action.reshape(self.num_envs, -1),
                    self.r_obs_tensor,
                    # self.actions["robot3"],
                    # self.r_obs["robot3"],
                ),
                dim=-1,
            ),
            "robot4" : torch.cat(
                (
                    self.x.reshape(self.num_envs, -1),
                    self.batch_action.reshape(self.num_envs, -1),
                    self.r_obs_tensor,
                    # self.actions["robot4"],
                    # self.r_obs["robot4"],
                ),
                dim=-1,
            ),
            "robot5" : torch.cat(
                (
                    self.x.reshape(self.num_envs, -1),
                    self.batch_action.reshape(self.num_envs, -1),
                    self.r_obs_tensor,
                    # self.actions["robot5"],
                    # self.r_obs["robot5"],
                ),
                dim=-1,
            ),
            "robot6" : torch.cat(
                (
                    self.x.reshape(self.num_envs, -1),
                    self.batch_action.reshape(self.num_envs, -1),
                    self.r_obs_tensor,
                    # self.actions["robot6"],
                    # self.r_obs["robot6"],
                ),
                dim=-1,
            ),
        }
        
        return observations
    
    def _get_rewards(self) -> dict[str, torch.Tensor]:
        total_reward = compute_rewards(
            self.x[:, :, 0],
            self.x_dot[:, :, 0],
            self.prev_actions,
            self.actions,
        )
        
        # print(total_reward["robot1"][0])
        
        return total_reward
    
    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        
        env_time_out = self.episode_length_buf >= self.max_episode_length - 1
        env_terminated = torch.zeros_like(env_time_out)
        
        terminated = {agent : env_terminated for agent in self.cfg.possible_agents}
        time_outs = {agent : env_time_out for agent in self.cfg.possible_agents}
        
        return terminated, time_outs
    
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.rod._ALL_INDICES
        
        super()._reset_idx(env_ids)
        
        _reset_x = torch.cat([10. * torch.rand((self.num_envs, 2)).to(self.device) - 5.,
                              2 * torch.pi * torch.rand((self.num_envs, 1)).to(self.device) - torch.pi,
                              torch.zeros((self.num_envs, 3)).to(self.device)], dim=-1)
        
        self.x[env_ids, :, 0] = _reset_x.clone()
        
        self.rod_root_state = torch.cat([self.x[:, 0:2, 0] + self.scene.env_origins[:, 0:2],
                                     0.5 * torch.ones((self.num_envs, 1)).to(self.device),
                                     torch.cos(self.x[:, 2:3, 0]/2).to(self.device),
                                     torch.zeros((self.num_envs, 2)).to(self.device),
                                     torch.sin(self.x[:, 2:3, 0]/2),
                                     self.x[:, 3:5, 0],
                                     torch.zeros((self.num_envs, 3)).to(self.device),
                                     self.x[:, 5:6, 0]], dim=-1)

        self.rod.write_root_state_to_sim(self.rod_root_state)
        
@torch.jit.script
def calc_f(x: torch.Tensor):
    '''
        Calculate current nonlienar function of the systems
        
        Args:
            x(Tensor(num_envs, num_state, 1)) : current state of the system
    '''
    
    f = torch.stack([torch.where(x[:, 3:4, 0] >= 0, torch.ones_like(x[:, 3:4, 0]), -torch.ones_like(x[:, 3:4, 0])),
                     torch.where(x[:, 4:5, 0] >= 0, torch.ones_like(x[:, 4:5, 0]), -torch.ones_like(x[:, 4:5, 0])),
                     torch.where(x[:, 5:6, 0] >= 0, torch.ones_like(x[:, 5:6, 0]), -torch.ones_like(x[:, 5:6, 0])),
                     x[:, 3:4, 0] * x[:, 5:6, 0],
                     x[:, 4:5, 0] * x[:, 5:6, 0]], dim=1)

    return f

@torch.jit.script
def compute_rewards(
    x : torch.Tensor,
    x_dot : torch.Tensor,
    prev_action : dict[str, torch.Tensor],
    action : dict[str, torch.Tensor],
):
    '''
        Compute reward for each environments
        
        Args:
            x(Tensor(num_envs, 6)) : state variable of each environments
            x_dot(Tensor(num_envs, 6)) : time derivative of state variable

        Return:
            total reward(Tensor(num_envs))
    '''
    
    # error_norm = torch.norm(x[:, 0:3], dim=-1)
    state_norm = torch.norm(x[:, 0:6], dim=-1)
    
    state_reward = torch.where(state_norm < 0.61803, 1. - torch.pow(state_norm, 2), 1./(1. + state_norm))
    # bonus_reward = torch.where(error_norm < 0.01, 0.1 * torch.ones_like(state_reward), torch.zeros_like(state_reward))
    
    # max_action_change = torch.max(torch.abs(prev_action - action), dim=-1)[0]

    # action_reward = torch.where(max_action_change < 0.1, torch.zeros_like(state_reward), -max_action_change * torch.ones_like(state_reward))

    env_reward = state_reward
    
    total_reward = {
        "robot1" : env_reward,
        "robot2" : env_reward,
        "robot3" : env_reward,
        "robot4" : env_reward,
        "robot5" : env_reward,
        "robot6" : env_reward,
    }

    return total_reward