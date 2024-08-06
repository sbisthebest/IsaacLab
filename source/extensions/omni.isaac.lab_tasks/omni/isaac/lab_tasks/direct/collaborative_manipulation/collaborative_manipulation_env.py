"""
    Collaborative Manipulation environment.
"""

import torch

from collections.abc import Sequence

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sim.spawners.materials import RigidBodyMaterialCfg
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
from omni.isaac.lab.envs import DirectRLEnvCfg, DirectRLEnv
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg, PhysxCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

@configclass
class CollaborativeManipultionEnvCfg(DirectRLEnvCfg):

    # simulation config
    sim: SimulationCfg = SimulationCfg(
        dt=1.0/120.0,
        render_interval=2,
        use_gpu_pipeline=True,
        gravity=(0.0, 0.0, -9.81),
        use_fabric=True, # False to update physics parameters in real-time

        physx = PhysxCfg(
            solver_type=1,
            use_gpu=True,
            enable_stabilization=True
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
            pos=(0., 0., 0.),
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
        env_spacing=2.0
    )

    # task config
    '''
        Actions : (# of agents) * (x-y force + torque)\

        Observation : x-y coordinate + phi + linear velocity + angular velocity
    '''
    decimation = 2
    episode_length_s = 10.0
    num_agents = 4
    num_actions = 3 * num_agents
    num_observations = 13

class CollaborativeManipulationEnv(DirectRLEnv):
    cfg: CollaborativeManipultionEnvCfg

    def __init__(self, cfg: CollaborativeManipultionEnvCfg, render_mode: None | str = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.rod_pos = self.rod.data.root_pos_w
        self.rod_quat = self.rod.data.root_quat_w
        self.rod_lin_vel_w = self.rod.data.root_lin_vel_w
        self.rod_ang_vel_w = self.rod.data.root_ang_vel_w
        self.rod_lin_vel_b = self.rod.data.root_lin_vel_b
        self.rod_ang_vel_b = self.rod.data.root_ang_vel_b

        self.env_origins = self.scene.env_origins

        self._x = self.rod_pos[:, 0:1] - self.env_origins[:, 0:1]
        self._y = self.rod_pos[:, 1:2] - self.env_origins[:, 1:2]
        self._vx = self.rod_lin_vel_w[:, 0:1]
        self._vy = self.rod_lin_vel_w[:, 1:2]
        self._w = self.rod_ang_vel_w[:, 2:3]
        self._phi = 2*torch.atan2(self.rod_quat[:, 3:4], self.rod_quat[:, 0:1])

        # Initialize state variable
        self.x = torch.cat([self._x,
                            self._y,
                            self._phi,
                            self._vx,
                            self._vy,
                            self._w], dim=-1).clone().unsqueeze(-1)

        # Define state-space equation
        self.mu = 0.1
        self.M = 1
        self.I = 1

        self.r = ((1., 1.),
                  (1., -1.),
                  (-1., 1.),
                  (-1., -1.))
        
        self.robot_num = self.cfg.num_agents

        self.A = torch.Tensor([[0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 0, 1],
                               [0, 0, 0, -self.mu/self.M, 0, 0],
                               [0, 0, 0, 0, -self.mu/self.M, 0],
                               [0, 0, 0, 0, 0, -self.mu/self.M]]) \
                               .repeat(self.num_envs, 1, 1).to(self.device)
        
        self.B = torch.zeros((self.num_envs,
                              6,
                              self.num_actions)).to(self.device)
        
        self.T = torch.Tensor([[0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0],
                               [-self.mu/self.M, 0, 0, 0, 1],
                               [0, -self.mu/self.M, 0, -1, 0],
                               [0, 0, -self.mu/self.M, 0, 0]]) \
                               .repeat(self.num_envs, 1, 1).to(self.device)
        
        for env_id in range(self.num_envs):
            for robot_id in range(self.robot_num):
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

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        # add rod to scene
        self.rod = RigidObject(self.cfg.rod_cfg)
        self.scene.rigid_objects["rod"] = self.rod

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.5, 0.5, 0.5))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # self.actions = actions.clone()
        self.actions = torch.Tensor([0.1, 0, 0, 0.1, 0, 0, 0.1, 0, 0, 0.1, 0, 0]).to(self.device).repeat(self.num_envs, 1).unsqueeze(-1)
    
    def _apply_action(self) -> None:
        self._u = self.actions.reshape(self.num_envs, self.num_actions, 1)

        self.f = torch.cat([U(self.x[:, 3:4, 0]),
                            U(self.x[:, 4:5, 0]),
                            U(self.x[:, 5:6, 0]),
                            self.x[:, 3:4, 0] * self.x[:, 5:6, 0],
                            self.x[:, 4:5, 0] * self.x[:, 5:6, 0]
                            ], dim=-1).unsqueeze(-1)

        self._x_dot = self.A @ self.x + self.T @ self.f + self.B @ self._u

        self.x += self._x_dot * self.physics_dt
        
        print(self.rod.data.root_state_w[0, :])
        print("x:", self.x[0, :, 0])
        print("f", self.f[0, :, 0])
        print("u", self._u[0, :, 0])
        print("\n\n\n")

        self.rod_root_state = torch.cat([self.x[:, 0:2, 0] + self.scene.env_origins[:, 0:2],
                                     torch.zeros((self.num_envs, 1)).to(self.device),
                                     torch.cos(self.x[:, 2:3, 0]/2).to(self.device),
                                     torch.zeros((self.num_envs, 2)).to(self.device),
                                     torch.sin(self.x[:, 2:3, 0]/2),
                                     self.x[:, 3:5, 0],
                                     torch.zeros((self.num_envs, 3)).to(self.device),
                                     self.x[:, 5:6, 0]], dim=-1)

        self.rod.write_root_state_to_sim(self.rod_root_state)

    def _get_observations(self) -> dict:
         obs = torch.cat(
             (
                 self.rod_pos,
                 self.rod_quat,
                 self.rod_lin_vel_w,
                 self.rod_ang_vel_w,
             ),
             dim=-1
         )
         
         observations = {"policy": obs}
         return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.rod_pos,
            self.rod_quat,
            self.rod_lin_vel_w,
            self.rod_ang_vel_w,
            self.reset_terminated,
        )

        return total_reward
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        reset_time_out = self.episode_length_buf >= self.max_episode_length - 1
        reset_terminated = torch.zeros_like(reset_time_out)

        return reset_terminated, reset_time_out
    
    def _reset_idx(self, env_ids: Sequence[int] | None):
        pass

@torch.jit.script
def U(x : torch.Tensor):
    '''
        Returns +1 for positive value, -1 for negative value
    '''

    return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

@torch.jit.script
def compute_rewards(
    rod_pos : torch.Tensor,
    rod_quat : torch.Tensor,
    rod_lin_vel_w : torch.Tensor,
    rod_ang_vel_w : torch.Tensor,
    reset_terminated : torch.Tensor
):
    '''
        Compute reward for each environments
        
        Args:
            rod_pos(Tensor(num_envs, 3)) : position of the rod in world frame
            rod_qaut(Tensor(num_envs, 4)) : quaternion of the rod in world frame
            rod_lin_vel_w(Tensor(num_envs, 3)) : linear velocity of the rod in world frame
            rod_ang_vel_w(Tensor(num_envs, 3)) : angular velocity of the rod in world frame

        Return:
            total reward(Tensor(num_envs))
    '''
    
    total_reward = torch.zeros_like(reset_terminated)

    return total_reward