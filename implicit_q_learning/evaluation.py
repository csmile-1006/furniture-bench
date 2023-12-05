from typing import Dict

import flax.linen as nn
import gym
import numpy as np
from tqdm import trange


def evaluate(agent: nn.Module, env: gym.Env, num_episodes: int, temperature: float = 0.00) -> Dict[str, float]:
    stats = {"return": [], "length": [], "success": []}
    ep, total_step = 0, 0
    pbar = trange(num_episodes, desc="evaluation")
    observation, done = env.reset(), np.zeros((env._num_envs), dtype=bool)
    while ep < num_episodes:
        # for ep in trange(num_episodes, desc='evaluation'):
        # pbar = tqdm(total=env.furniture.max_env_steps, leave=False, desc=f"episode {ep + 1}")
        # observation, done = env.reset(), False
        # for env_idx in range(len(env._num_envs)):
        # while not done:
        action = agent.sample_actions(observation, temperature=temperature)
        observation, _, done, info = env.step(action)
        total_step += min(env._num_envs, num_episodes)
        for env_idx in range(min(env._num_envs, num_episodes)):
            if done[env_idx].item() is True:
                print(f"total_step {total_step} / env_idx: {env_idx} / done: {done[env_idx]}")
                new_ob = env.reset_env(env_idx)
                for key in observation:
                    observation[key][env_idx] = new_ob[key]
                done[env_idx] = False
                for k in stats.keys():
                    stats[k].append(info[f"episode_{env_idx}"][k])
                ep += 1
                pbar.update(1)
        # if total_step % (env.max_env_steps * env.env.num_envs) == 0:
        #     observation = env.reset()
        pbar.set_description(f"total_step: {total_step}")

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats
