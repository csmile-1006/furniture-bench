import time

import gym
import numpy as np

from wrappers.common import TimeStep
from wrappers.video_recorder_wrapper import VideoRecorder, VideoRecorderList


class SuccessRecordWrapper(gym.ActionWrapper):
    """Save video of rollouts for successful trajectories during the training."""

    def __init__(
        self, env: gym.Env, video_recorder: VideoRecorderList, skip_episode: int = 10
    ):
        super().__init__(env)
        self._num_envs = env.num_envs

        self.video_recorder = video_recorder
        self.video_recorder.init(obs=env.render(mode="rgb_array"), enabled=True)

        self.episode_counter = 0
        self.skip_episode = skip_episode

    def _reset_record(self, env_idx=None):
        if env_idx is None:
            self.video_recorder.init(
                obs=self.env.render(mode="rgb_array"), enabled=True
            )
        else:
            self.video_recorder.init_idx(
                obs=self.env.render(mode="rgb_array"),
                idx=env_idx,
                enabled=True,
            )

    def step(self, action: np.ndarray) -> TimeStep:
        observation, reward, done, info = self.env.step(action)
        # Save recording.
        
        # self.video_recorder.record(observation)
        self.video_recorder.record(self.env.render(mode="rgb_array"))

        for i in range(self._num_envs):
            if not done[i]:
                continue
            if not reward[i] > 0:
                self.episode_counter += 1
                continue
            if self.episode_counter < self.skip_episode:
                continue
            # Log successful episode.
            self.video_recorder.save(
                file_name=f"success_{self.episode_counter}",
                env_idx=i,
            )
            self.episode_counter = 0
        return observation, reward, done, info

    def reset_env(self, env_idx):
        obs = self.env.reset_env(env_idx)
        self._reset_record(env_idx)
        return obs

    def reset_env_to(self, env_idx, state):
        obs = self.env.reset_env_to(env_idx, state)
        self._reset_record(env_idx)
        return obs

    def reset(self) -> np.ndarray:
        obs = self.env.reset()
        self._reset_record()
        return obs

    def reset_to(self, state) -> np.ndarray:
        obs = self.env.reset_to(state)
        self._reset_record()
        return obs
