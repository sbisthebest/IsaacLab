"""
    Stewart Manipulation environment
"""

import gymnasium as gym

from . import agents
from .stewart_env import StewartManipulationEnvCfg
from .stewart_camera_env import StewartManipulationCameraEnv


gym.register(
    id="stewart_manipulation",
    entry_point=f"{__name__}.stewart_env:StewartManipulationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point" : f"{__name__}.stewart_env:StewartManipulationEnvCfg",
        "rl_games_cfg_entry_point" : f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point" : f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point" : f"{agents.__name__}:sb3_ppo_cfg.yaml",
    }
)

gym.register(
    id="stewart_camera_manipulation",
    entry_point=f"{__name__}.stewart_camera_env:StewartManipulationCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point" : f"{__name__}.stewart_camera_env:StewartManipulationCameraEnvCfg",
        "rl_games_cfg_entry_point" : f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
        "skrl_cfg_entry_point" : f"{agents.__name__}:skrl_camera_ppo_cfg.yaml",
        "sb3_cfg_entry_point" : f"{agents.__name__}:sb3_camera_ppo_cfg.yaml",
    }
)