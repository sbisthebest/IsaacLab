"""
    Multi-Agent Collaborative Manipulation environment.
"""

import gymnasium as gym

from . import agents
from .MA_collaborative_manipulation_env import MACollaborativeManipulationEnv, MACollaborativeManipulationEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="MA_collaborative_manipulation",
    entry_point="omni.isaac.lab_tasks.direct.MA_collaborative_manipulation:MACollaborativeManipulationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MACollaborativeManipulationEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)
