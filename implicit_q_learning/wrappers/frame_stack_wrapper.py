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
        self._i = 0
        self._frames = {frame: {key: deque([], maxlen=self._num_frames) for key in self.env.observation_space} for frame in range(self._skip_frame)}
        _obs = self.env.reset()
        for frame in range(self._skip_frame):
            for _ in range(self._num_frames):
                for key in _obs:
                    self._frames[frame][key].append(_obs[key].squeeze())
        return self._transform_observation(_obs)

    def step(self, action):
        self._i += 1
        _obs, reward, done, info = self.env.step(action)
        for key in _obs:
            self._frames[self._i % self._skip_frame][key].append(_obs[key].squeeze())
        obs = self._transform_observation(_obs)
        return obs, reward, done, info

    @property
    def observation_space(self):
        obs_space = {}
        for key, val in self.env.observation_space.items():
            obs_space[key] = spaces.Box(val.low.reshape(-1)[0], val.high.reshape(-1)[0], (self._num_frames, *val.shape))
        return spaces.Dict(obs_space)

if __name__ == "__main__":
    import furniture_bench
    env_name = "FurnitureSimImageFeature-v0/one_leg"
    # env_name = "FurnitureSim-v0/one_leg"
    env_id, furniture_name = env_name.split("/")
    env = gym.make(env_id,
        furniture=furniture_name,
        data_path="",
        use_encoder=False,
        encoder_type="vip",
        compute_device_id=0,
        graphics_device_id=0,
        headless=True,
        record=False,
        # np_step_out=True,
        record_dir="",
        max_env_steps=600 if "Sim" in env_id else 3000,
        # concat_robot_state=True,
    )
    num_frames, skip_frame = 4, 16
    env = FrameStackWrapper(env, num_frames, skip_frame)
    init = env.reset()
    timestep = 0
    print(f"timestep {timestep} / stack step {env._i} / current stack {init['timestep']}")
    for _ in range(630):
        res, rew, done, info = env.step(env.action_space.sample())
        timestep += 1
        color_image2 = res["image2"]
        print(f"timestep {timestep} / stack step {env._i} / current stack {res['timestep']}")
        if done:
            break
    
 
