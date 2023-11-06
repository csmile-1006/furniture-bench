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
        self.check_skill = False
        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()
        if self.check_skill:
            self.skill_complete = 0.0

    def step(self, action: np.ndarray) -> TimeStep:
        observation, reward, done, info = self.env.step(action)

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        if self.check_skill:
            _, skill_complete = self.env.get_assembly_action()
            self.skill_complete += skill_complete
        info["total"] = {"timesteps": self.total_timesteps}

        if done:
            info["episode"] = {}
            info["episode"]["return"] = self.reward_sum
            info["episode"]["length"] = self.episode_length
            info["episode"]["duration"] = time.time() - self.start_time
            if self.check_skill:
                info["episode"]["skill"] = self.skill_complete

            if hasattr(self, "get_normalized_score"):
                info["episode"]["return"] = (
                    self.get_normalized_score(info["episode"]["return"]) * 100.0
                )

        return observation, reward, done, info

    def reset(self) -> np.ndarray:
        self._reset_stats()
        return self.env.reset()