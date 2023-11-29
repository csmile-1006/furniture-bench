from typing import Dict

import flax.linen as nn
import gym
import numpy as np
from tqdm import tqdm, trange


def evaluate(agent: nn.Module, env: gym.Env, num_episodes: int, temperature: float = 0.00) -> Dict[str, float]:
    stats = {"return": [], "length": []}

    for ep in trange(num_episodes, desc="evaluation"):
        pbar = tqdm(total=env.furniture.max_env_steps, leave=False, desc=f"episode {ep + 1}")
        observation, done = env.reset(), False
        while not done:
            action = agent.eval_actions(observation)
            observation, _, done, info = env.step(action)
            pbar.update(1)

        for k in stats.keys():
            stats[k].append(info["episode"][k])

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats
