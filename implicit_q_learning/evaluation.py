from typing import Dict

import pickle
import flax.linen as nn
import gym
import numpy as np
from tqdm import trange
from pathlib import Path


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


def evaluate_with_save(
    agent: nn.Module, env: gym.Env, num_episodes: int, temperature: float = 0.00, save_dir: Path = None
) -> Dict[str, float]:
    stats = {"return": [], "length": [], "success": [], "spl": []}
    episodes = {env_idx: {key: [] for key in ["observations", "actions"]} for env_idx in range(env._num_envs)}
    ep, total_step = 0, 0
    pbar = trange(num_episodes, desc="evaluation", ncols=0)
    observation, done = env.reset(), np.zeros((env._num_envs), dtype=bool)
    for env_idx in range(min(env._num_envs, num_episodes)):
        episodes[env_idx]["observations"].append(
            {key: observation[key][env_idx, -1] for key in ["color_image1", "color_image2"]}
        )
    while ep < num_episodes:
        action = agent.sample_actions(observation, temperature=temperature)
        observation, _, done, info = env.step(action)
        total_step += min(env._num_envs, num_episodes)
        for env_idx in range(min(env._num_envs, num_episodes)):
            episodes[env_idx]["observations"].append(
                {key: observation[key][env_idx, -1] for key in ["color_image1", "color_image2"]}
            )
            episodes[env_idx]["actions"].append(action[env_idx])
            if done[env_idx].item() is True:
                tp = "success" if info[f"episode_{env_idx}"]["success"] else "failure"
                pickle.dump(episodes[env_idx], (save_dir / f"episode_{ep}_{tp}.pkl").open("wb"))
                new_ob = env.reset_env(env_idx)
                episodes[env_idx] = {key: [] for key in ["observations", "actions"]}
                episodes[env_idx]["observations"].append(
                    {key: observation[key][env_idx, -1] for key in ["color_image1", "color_image2"]}
                )
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
