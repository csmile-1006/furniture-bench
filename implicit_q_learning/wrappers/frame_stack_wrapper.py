import os
import gym
import numpy as np
from gym import spaces
from collections import deque


class FrameStackWrapper(gym.ActionWrapper):
    def __init__(self, env, num_frames, skip_frame):
        super().__init__(env)
        self._num_frames = num_frames
        self._skip_frame = skip_frame
        self._i = 0
        self._frames = {frame: {key: deque([], maxlen=num_frames) for key in self.env.observation_space} for frame in range(self._skip_frame)}

    def _transform_observation(self, obs):
        stack = self._frames[self._i % self._skip_frame]
        for key in obs:
            assert len(stack[key]) == self._num_frames, f"{len(stack[key])} != {self._num_frames}"
            obs[key] = np.stack(stack[key])
        return obs

    def reset(self):
        obs = self.env.reset()
        self._i = 0
        for frame in range(self._skip_frame):
            for key in obs:
                for _ in range(self._num_frames):
                    self._frames[frame][key].append(obs[key])
        return self._transform_observation(obs)

    def step(self, action):
        self._i += 1
        obs, reward, done, info = self.env.step(action)
        stack = self._frames[self._i % self._skip_frame]
        for key in obs:
            stack[key].append(obs[key])
        rew = self._transform_observation(obs)
        return rew, reward, done, info

    @property
    def observation_space(self):
        embedding_dim = self.env.embedding_dim
        robot_state_dim = 14

        return spaces.Dict(
            dict(
                robot_state=spaces.Box(-np.inf, np.inf, (self._num_frames, robot_state_dim,)),
                image1=spaces.Box(-np.inf, np.inf, (self._num_frames, self.env.embedding_dim,)),
                image2=spaces.Box(-np.inf, np.inf, (self._num_frames, self.env.embedding_dim,)),
            )
        )


if __name__ == "__main__":
    import furniture_bench
    env_name = "FurnitureSimImageFeature-v0/one_leg"
    env_id, furniture_name = env_name.split("/")
    env = gym.make(env_id,
        furniture=furniture_name,
        data_path="",
        use_encoder=False,
        encoder_type="r3m",
        compute_device_id=0,
        graphics_device_id=0,
        headless=True,
        record=False,
        record_dir="",
        max_env_steps=600 if "Sim" in env_id else 3000
    )
    num_frames = 4
    env = FrameStackWrapper(env, num_frames)
    init = env.reset()
    timestep = 0
    for _ in range(630):
        timestep += 1
        res, rew, done, info = env.step(env.action_space.sample())
        print(f"timestep {timestep}: {res['image2'].shape}")
        if done:
            break
    
 
