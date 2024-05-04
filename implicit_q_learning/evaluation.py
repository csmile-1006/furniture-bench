from typing import Dict, Any, Union

import pickle
import flax.linen as nn
import gym
import numpy as np
from tqdm import trange
from pathlib import Path

from furniture_bench.envs.policy_envs.furniture_sim_image_with_feature import FurnitureSimImageWithFeature
from furniture_bench.envs.furniture_sim_env import FurnitureSimEnv
from wrappers.video_recorder_wrapper import VideoRecorder, VideoRecorderList


def recursive_furniture_instance(env):
    try:
        return recursive_furniture_instance(env.env)
    except:
        pass

    if isinstance(env, FurnitureSimImageWithFeature):
        return True
    elif isinstance(env, FurnitureSimEnv):
        return True
    else:
        return False


def evaluate(
    agent: nn.Module,
    env: gym.Env,
    num_episodes: int,
    expl_noise: float = 0.00,
    state: Dict[str, Any] = None,
    video_recorder=None,
    step: int = 0,
) -> Dict[str, float]:
    # stats = {"return": [], "length": [], "success": [], "spl": []}
    is_furniture_env = recursive_furniture_instance(env)
    stats = {"return": [], "length": [], "success": []}
    ep, total_step = 0, 0
    pbar = trange(num_episodes, desc="evaluation", ncols=0, leave=False)
    use_video_recorder = video_recorder is not None
    if state is not None:
        observation, done = env.reset_to(state), np.zeros((env._num_envs), dtype=bool)
    else:
        observation, done = env.reset(), np.zeros((env._num_envs), dtype=bool)
    if use_video_recorder:
        if isinstance(video_recorder, VideoRecorderList):
            if is_furniture_env:
                video_recorder.init(obs=env.render(mode="rgb_array"), enabled=True)
            else:
                video_recorder.init(obs=observation[list(observation.keys())[0]][0][-1], enabled=True)
    while ep < num_episodes:
        action = agent.sample_actions(observation, expl_noise)
        observation, _, done, info = env.step(action)
        total_step += min(env._num_envs, num_episodes)
        if use_video_recorder:
            if is_furniture_env:
                video_recorder.record(env.render(mode="rgb_array"))
            else:
                video_recorder.record(observation[list(observation.keys())[0]][0][-1])
        for env_idx in range(min(env._num_envs, num_episodes)):
            if done[env_idx].item() is True:
                for k in stats.keys():
                    stats[k].append(info[f"episode_{env_idx}"][k])
                if use_video_recorder:
                    video_recorder.save(f"evaluation_{step}_{ep}", env_idx)
                if state is not None:
                    new_ob = env.reset_env_to(env_idx, state[env_idx])
                else:
                    new_ob = env.reset_env(env_idx)
                for key in observation:
                    observation[key][env_idx] = new_ob[key]
                done[env_idx] = False
                ep += 1
                pbar.update(1)
                if isinstance(video_recorder, VideoRecorderList):
                    if is_furniture_env:
                        video_recorder.init_idx(obs=env.render(mode="rgb_array"), idx=env_idx, enabled=True)
                    else:
                        video_recorder.init_idx(obs=observation[list(observation.keys())[0]][0][-1], idx=env_idx, enabled=True)
        pbar.set_postfix({"total_step": f"{total_step}", "num_success": f"{np.sum(stats['success'], dtype=np.int32)}"})

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats


def evaluate_with_save(
    agent: nn.Module, env: gym.Env, num_episodes: int, expl_noise: float = 0.00, save_dir: Path = None
) -> Dict[str, float]:
    stats = {"return": [], "length": [], "success": [], "spl": []}
    episodes = {env_idx: {key: [] for key in ["observations", "actions"]} for env_idx in range(env.num_envs)}
    ep, total_step = 0, 0
    pbar = trange(num_episodes, desc="evaluation", ncols=0)
    observation, done = env.reset(), np.zeros((env.num_envs), dtype=bool)
    for env_idx in range(min(env.num_envs, num_episodes)):
        episodes[env_idx]["observations"].append(
            {key: observation[key][env_idx, -1] for key in ["color_image1", "color_image2"]}
        )
    while ep < num_episodes:
        action = agent.sample_actions(observation, expl_noise=expl_noise)
        observation, _, done, info = env.step(action)
        total_step += min(env.num_envs, num_episodes)
        for env_idx in range(min(env.num_envs, num_episodes)):
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
