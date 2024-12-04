import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.sim.spawners.materials import RigidBodyMaterialCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import TiledCamera, TiledCameraCfg, save_images_to_file
from omni.isaac.lab.sim import SimulationCfg, PhysxCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab_assets import ISAACLAB_ASSETS_DATA_DIR
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils import configclass

@configclass
class StewartManipulationCameraEnvCfg(DirectRLEnvCfg):

    # simulation config
    sim = SimulationCfg(
        dt = 1.0/120.0,
        render_interval = decimation,
        gravity = (0., 0., -9.81),
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
            friction_combine_mode = "average",
            restitution_combine_mode = "max",
        )
    )

    # camera config
    tiled_camera = TiledCameraCfg(
        prim_path = "/World/envs/env_.*/Camera",
        offset = TiledCameraCfg.OffsetCfg(pos=(0., 0., 0.4), rot=(0.707, 0.707, 0., 0.), convention="world"),
        data_types = ["rgb"],
        spawn = sim_utils.PinholeCameraCfg(),
        width=400,
        height=296,
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
                restitution_combine_mode = 'max',
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
        num_envs = 4096,
        env_spacing = 1.0,
    )

    # task config
    episode_length_s = 10.
    decimation = 2
    action_space = 3
    observation_space = [tiled_camera.height, tiled_camera.width, 3]
    state_space = 0

@configclass
class StewartManipulationCameraEnv(DirectRLEnv):

    cfg : StewartManipulationCameraEnvCfg

    def __init__(self, cfg : StewartManipulationCameraEnvCfg, render_mode : str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)


