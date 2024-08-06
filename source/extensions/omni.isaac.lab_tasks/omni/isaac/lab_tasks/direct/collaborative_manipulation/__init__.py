"""
    Collaborative Manipulation environment
"""

import gymnasium as gym

from . import agents
from .collaborative_manipulation_env import CollaborativeManipulationEnv, CollaborativeManipultionEnvCfg


gym.register(
    id="collaborative_manipulation",
    entry_point="omni.isaac.lab_tasks.direct.collaborative_manipulation:CollaborativeManipulationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point" : CollaborativeManipultionEnvCfg,
        "rl_games_cfg_entry_point" : f"{agents.__name__}:rl_games_ppo_cfg.yaml"
    }
)