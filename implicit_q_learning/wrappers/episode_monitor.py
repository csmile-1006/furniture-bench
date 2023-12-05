import time

import os
import gym
import numpy as np

from wrappers.common import TimeStep


class EpisodeMonitor(gym.ActionWrapper):
    """A class that computes episode returns and lengths."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # self.check_skill = hasattr(self.env, "get_assembly_action")
        self._num_envs = env.num_envs
        self.check_skill = False
        self._reset_stats()
        self.total_timesteps = {idx: 0 for idx in range(self._num_envs)}

    def _reset_stats(self):
        self.reward_sum = {idx: 0.0 for idx in range(self._num_envs)}
        self.episode_length = {idx: 0 for idx in range(self._num_envs)}
        self.start_time = {idx: time.time() for idx in range(self._num_envs)}
        # if self.check_skill:
        #     self.skill_complete = 0.0

    def step(self, action: np.ndarray) -> TimeStep:
        observation, reward, done, info = self.env.step(action)

        for i in range(self._num_envs):
            self.reward_sum[i] += reward[i]
            self.episode_length[i] += 1
            self.total_timesteps[i] += 1

        if self.check_skill:
            _, skill_complete = self.env.get_assembly_action()
            self.skill_complete += skill_complete
        info["total"] = {f"timesteps_{i}": self.total_timesteps[i] for i in range(self._num_envs)}

        for i in range(self._num_envs):
            if done[i]:
                info[f"episode_{i}"] = {}
                info[f"episode_{i}"]["success"] = float(self.episode_length[i] < self.env.max_env_steps)
                info[f"episode_{i}"]["return"] = self.reward_sum[i].item()
                info[f"episode_{i}"]["length"] = self.episode_length[i]
                info[f"episode_{i}"]["duration"] = time.time() - self.start_time[i]
                # if self.check_skill:
                #     info[f"episode_{i}"]["skill"] = self.skill_complete

                if hasattr(self, "get_normalized_score"):
                    info[f"episode_{i}"]["return"] = self.get_normalized_score(info[f"episode_{i}"]["return"]) * 100.0

        return observation, reward, done, info

    def reset(self) -> np.ndarray:
        self._reset_stats()
        return self.env.reset()
