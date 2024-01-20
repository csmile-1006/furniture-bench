from typing import Dict

import flax.linen as nn
import gym
import numpy as np
from tqdm import trange


def evaluate(agent: nn.Module, env: gym.Env, num_episodes: int, temperature: float = 0.00) -> Dict[str, float]:
    stats = {"return": [], "length": [], "success": [], "spl": []}
    ep, total_step = 0, 0
    pbar = trange(num_episodes, desc="evaluation", ncols=0)
    observation, done = env.reset(), np.zeros((env._num_envs), dtype=bool)
    while ep < num_episodes:
        action = agent.sample_actions(observation, temperature=temperature)
        observation, _, done, info = env.step(action)
        total_step += min(env._num_envs, num_episodes)
        for env_idx in range(min(env._num_envs, num_episodes)):
            if done[env_idx].item() is True:
                new_ob = env.reset_env(env_idx)
                for key in observation:
                    observation[key][env_idx] = new_ob[key]
                done[env_idx] = False
                for k in stats.keys():
                    stats[k].append(info[f"episode_{env_idx}"][k])
                ep += 1
                pbar.update(1)
        pbar.set_postfix({"total_step": f"{total_step}", "num_success": f"{np.sum(stats['success'], dtype=np.int32)}"})

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats
