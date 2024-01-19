import gym
import numpy as np
from gym import spaces
from collections import deque


class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, num_frames, skip_frame):
        super().__init__(env)
        self._num_frames = num_frames
        self._skip_frame = skip_frame
        self._num_envs = self.env.num_envs
        self._i = {env_idx: 0 for env_idx in range(self._num_envs)}
        self.max_env_steps = env.furniture.max_env_steps
        self._frames = {
            env_idx: {
                frame: {key: deque([], maxlen=num_frames) for key in self.env.observation_space}
                for frame in range(self._skip_frame)
            }
            for env_idx in range(self._num_envs)
        }

    def _transform_observation(self, obs):
        new_obs = {key: np.zeros((self._num_envs, self._num_frames, *obs[key].shape[1:])) for key in obs}
        for env_idx in range(self._num_envs):
            stack = self._frames[env_idx][self._i[env_idx] % self._skip_frame]
            for key in obs:
                assert len(stack[key]) == self._num_frames, f"{len(stack[key])} != {self._num_frames}"
                new_obs[key][env_idx] = np.stack(stack[key])
        return new_obs

    def reset_env(self, idx):
        _obs = self.env.reset_env(idx)
        self.env.refresh()

        self._i[idx] = 0
        self._frames[idx] = {
            frame: {key: deque([], maxlen=self._num_frames) for key in self.env.observation_space}
            for frame in range(self._skip_frame)
        }
        if _obs is None:
            _obs = self.env.unwrapped._get_observation()
        for frame in range(self._skip_frame):
            for _ in range(self._num_frames):
                for key in _obs:
                    self._frames[idx][frame][key].append(_obs[key][idx])
        stack = self._frames[idx][0]
        return {key: np.stack(stack[key]) for key in _obs}

    def reset(self):
        self._i = {env_idx: 0 for env_idx in range(self._num_envs)}
        self._frames = {
            env_idx: {
                frame: {key: deque([], maxlen=self._num_frames) for key in self.env.observation_space}
                for frame in range(self._skip_frame)
            }
            for env_idx in range(self._num_envs)
        }
        _obs = self.env.reset()
        for env_idx in range(self._num_envs):
            for frame in range(self._skip_frame):
                for _ in range(self._num_frames):
                    for key in _obs:
                        self._frames[env_idx][frame][key].append(_obs[key][env_idx])
        return self._transform_observation(_obs)

    def step(self, action):
        _obs, reward, done, info = self.env.step(action)
        for env_idx in range(self._num_envs):
            self._i[env_idx] += 1
            for key in _obs:
                self._frames[env_idx][self._i[env_idx] % self._skip_frame][key].append(_obs[key][env_idx])
        obs = self._transform_observation(_obs)
        return obs, reward, done, info

    @property
    def observation_space(self):
        obs_space = {}
        for key, val in self.env.observation_space.items():
            obs_space[key] = spaces.Box(val.low.reshape(-1)[0], val.high.reshape(-1)[0], (self._num_frames, *val.shape))
        return spaces.Dict(obs_space)


if __name__ == "__main__":
    import isaacgym
    import furniture_bench
    import sys

    sys.path.append("/home/changyeon/ICML2024/furniture-bench/implicit_q_learning")
    from wrappers.episode_monitor import EpisodeMonitor

    env_name = "FurnitureSimImageWithFeature-v0/one_leg"
    # env_name = "FurnitureSim-v0/one_leg"
    env_id, furniture_name = env_name.split("/")
    env = gym.make(
        env_id,
        furniture=furniture_name,
        num_envs=10,
        data_path="",
        use_encoder=False,
        encoder_type="vip",
        compute_device_id=1,
        graphics_device_id=1,
        headless=True,
        record=False,
        resize_img=True,
        # np_step_out=True,
        record_dir="",
        max_env_steps=20 if "Sim" in env_id else 3000,
        # concat_robot_state=True,
    )
    num_frames, skip_frame = 4, 16
    env = FrameStackWrapper(env, num_frames, skip_frame)
    env = EpisodeMonitor(env)
    init = env.reset()
    env.reset_env(0)
    timestep = 0
    print(f"timestep {timestep} / stack step {env.env._i} / current stack {init['image1'].shape}")
    for _ in range(630):
        res, rew, done, info = env.step(env.action_space.sample())
        timestep += 1
        print(f"timestep {timestep} / stack step {env.env._i} / current stack {res['image1'].shape}")
        if np.any(done):
            for env_idx in range(10):
                res[env_idx] = env.reset_env(env_idx)
                done[env_idx] = False
            print(f"timestep {timestep} / stack step {env.env._i} / current stack {res['image1'].shape} / done: {done}")
            break
