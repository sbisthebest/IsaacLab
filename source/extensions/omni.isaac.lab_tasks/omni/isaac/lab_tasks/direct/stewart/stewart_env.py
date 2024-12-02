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

@configclass
class StewartManipulationEnvCfg(DirectRLEnvCfg):
    # task config
    episode_length_s = 10.
    decimation = 2
    action_space = 0
    observation_space = 0
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
            static_friction = 0.3,
            dynamic_friction = 0.3,
            restitution = 0.0,
            friction_combine_mode = 'average',
            restitution_combine_mode = 'max',
        )
    )
    
    # robot config
    robot = ArticulationCfg(
        prim_path = "/World/envs/env_.*/Robot",
        spawn = sim_utils.UsdFileCfg(
            usd_path = f"{ISAACLAB_ASSETS_DATA_DIR}/Personal/Stewart.usd",
            activate_contact_sensors = False,
            rigid_props = sim_utils.RigidBodyPropertiesCfg(
                disable_gravity = False,
            ),
            articulation_props = sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count = 12, solver_velocity_iteration_count = 1
            ),
        ),
        init_state = ArticulationCfg.InitialStateCfg(
            joint_pos = {
                # TODO : joint init
            },
            pos = (1.0, 0.0, 0.0),
            rot = (0.0, 0.0, 0.0, 1.0),
        ),
        actuators = {
            # TODO : joint spec
        },
    )
    
    # ball config
    ball = RigidObjectCfg(
        prim_path = "/World/envs/env_.*/Ball",
        spawn = sim_utils.SphereCfg(
            radius = 0.02,
            rigid_props = sim_utils.RigidBodyPropertiesCfg(),
            mass_props = sim_utils.MassPropertiesCfg(mass = 1.0),
            collision_props = sim_utils.CollisionPropertiesCfg(),
            visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metalic = 0.1),
            physics_material = RigidBodyMaterialCfg(
                static_friction = 0.3,
                dynamic_friction = 0.3,
                restitution = 0.0,
                friction_combine_mode = 'average',
                restitution_combine_mode = 'max',
            )
        ),
        init_state = RigidObjectCfg.InitialStateCfg(
            pos = (0., 0., 10.),
            rot = (1., 0., 0., 0.),
            lin_vel = (0., 0., 0.),
            ang_vel = (0., 0., 0.)
        ),
        collision_group = 0,
        debug_vis = False
    )
    
class StewartManipulationEnv(DirectRLEnv):
    cfg : StewartManipulationEnvCfg
    
    