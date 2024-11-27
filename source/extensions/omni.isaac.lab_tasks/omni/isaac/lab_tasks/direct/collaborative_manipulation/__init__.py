"""
    Collaborative Manipulation environment
"""

import gymnasium as gym

from . import agents

gym.register(
    id="collaborative_manipulation",
    entry_point=f"{__name__}.collaborative_manipulation_env:CollaborativeManipulationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point" : f"{__name__}.collaborative_manipulation_env:CollaborativeManipulationEnvCfg",
        "rl_games_cfg_entry_point" : f"{agents.__name__}:rl_games_ppo_cfg.yaml"
    }
)