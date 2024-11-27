"""
    Non-prehensile Manipulation environment
"""

import gymnasium as gym

from . import agents
from .nonprehensile_manipulation_env import NonPrehensileManipulationEnvCfg


gym.register(
    id="nonprehensile_manipulation",
    entry_point=f"{__name__}.nonprehensile_manipulation_env:NonPrehensileManipulationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point" : f"{__name__}.nonprehensile_manipulation_env:NonPrehensileManipulationEnvCfg",
        "rl_games_cfg_entry_point" : f"{agents.__name__}:rl_games_ppo_cfg.yaml"
    }
)