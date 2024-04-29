import time

import gym
import numpy as np

from wrappers.common import TimeStep
from dataset_utils import SHORTEST_PATHS


class EpisodeMonitor(gym.ActionWrapper):
    """A class that computes episode returns and lengths."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._num_envs = env.num_envs
        self._reset_stats()
        self.total_timesteps = 0
        self.num_episodes = 0
        self.success_episodes = 0
        self.do_eval = False

    def _reset_stats(self):
        self.reward_sum = {idx: 0.0 for idx in range(self._num_envs)}
        self.episode_length = {idx: 0 for idx in range(self._num_envs)}
        self.start_time = {idx: time.time() for idx in range(self._num_envs)}
        self.spl = SHORTEST_PATHS[self.env.furniture_name]

    def reset_env(self, idx):
        obs = self.env.reset_env(idx)
        self.refresh()

        self.reward_sum[idx] = 0.0
        self.episode_length[idx] = 0
        self.start_time[idx] = time.time()

        return obs

    def reset_env_to(self, idx, state):
        obs = self.env.reset_env_to(idx, state)

        self.reward_sum[idx] = 0.0
        self.episode_length[idx] = 0
        self.start_time[idx] = time.time()

        return obs

    def set_eval_flag(self):
        self.do_eval = True

    def unset_eval_flag(self):
        self.do_eval = False

    def step(self, action: np.ndarray) -> TimeStep:
        observation, reward, done, info = self.env.step(action)

        for i in range(self._num_envs):
            self.reward_sum[i] += reward[i]
            self.episode_length[i] += 1
            self.total_timesteps += 1

        info["total"] = {"timesteps": self.total_timesteps}

        for i in range(self._num_envs):
            if done[i]:
                info[f"episode_{i}"] = {}
                info[f"episode_{i}"]["success"] = float(self.episode_length[i] < self.env.max_env_steps)
                info[f"episode_{i}"]["spl"] = (
                    self.spl * self.reward_sum[i].item() / max(self.episode_length[i], self.spl)
                )
                info[f"episode_{i}"]["return"] = self.reward_sum[i].item()
                info[f"episode_{i}"]["length"] = self.episode_length[i]
                info[f"episode_{i}"]["duration"] = time.time() - self.start_time[i]

                if not self.do_eval:
                    self.num_episodes += 1
                    info[f"episode_{i}"]["num_episodes"] = self.num_episodes
                    self.success_episodes += info[f"episode_{i}"]["success"]
                    info[f"episode_{i}"]["success_episodes"] = self.success_episodes

                if hasattr(self, "get_normalized_score"):
                    info[f"episode_{i}"]["return"] = self.get_normalized_score(info[f"episode_{i}"]["return"]) * 100.0

        return observation, reward, done, info

    def reset(self) -> np.ndarray:
        self._reset_stats()
        return self.env.reset()

    def reset_to(self, state) -> np.ndarray:
        self._reset_stats()
        return self.env.reset_to(state)
