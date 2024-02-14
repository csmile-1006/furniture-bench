from typing import Optional

import gym
import numpy as np

import wandb


class WANDBVideo(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        name: str = "video",
        pixel_hw: int = 224,
        render_kwargs={},
        max_videos: Optional[int] = None,
        log_period=200,
    ):
        super().__init__(env)

        self._name = name
        self._pixel_hw = pixel_hw
        self._render_kwargs = render_kwargs
        self._max_videos = max_videos
        self._video = []
        self._total_episodes = 0
        self._log_period = log_period

    def _add_frame(self, obs):
        if self._max_videos is not None and self._max_videos <= 0:
            return
        if isinstance(obs, dict) and "color_image2" in obs:
            if obs["color_image2"].ndim == 4:
                self._video.append(obs["color_image2"][-1, ..., -3:])
            else:
                self._video.append(obs["color_image2"][..., -3:])
        else:
            self._video.append(
                self.render(height=self._pixel_hw, width=self._pixel_hw, mode="rgb_array", **self._render_kwargs)
            )

    def reset(self, **kwargs):
        self._video.clear()
        obs = super().reset(**kwargs)
        self._add_frame(obs)
        return obs

    def step(self, action: np.ndarray):
        obs, reward, done, info = super().step(action)
        self._add_frame(obs)

        if done[-1] is True and len(self._video) > 0:
            self._total_episodes += 1
            if self._max_videos is not None:
                self._max_videos -= 1
            video = np.moveaxis(np.stack(self._video), -1, 1)
            if video.shape[1] == 1:
                video = np.repeat(video, 3, 1)
            if self._total_episodes % self._log_period == 0:
                video = wandb.Video(video, fps=30, format="mp4")
                wandb.log({self._name: video}, commit=False)

        return obs, reward, done, info
