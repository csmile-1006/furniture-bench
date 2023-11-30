import gym
import numpy as np
from collections import deque


class FlattenWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._target_keys = ["robot_state", "image1", "image2"]

    def _transform_observation(self, obs):
        return np.concatenate([obs[key] for key in self._target_keys], axis=-1)

    def reset(self):
        _obs = self.env.reset()
        return self._transform_observation(_obs)

    def step(self, action):
        _obs, reward, done, info = self.env.step(action)
        obs = self._transform_observation(_obs)
        return obs, reward, done, info

    @property
    def action_space(self):
        return gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.env.pose_dim + 1,),
        )

    @property
    def observation_space(self):
        robot_state_dim = 14
        obs_dim = 0
        for key in self._target_keys:
            if key == "robot_state":
                obs_dim += robot_state_dim
            elif "image" in key:
                obs_dim += self.embedding_dim
        return gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
        )


if __name__ == "__main__":
    import furniture_bench

    env_name = "FurnitureSimImageFeature-v0/one_leg"
    # env_name = "FurnitureSim-v0/one_leg"
    env_id, furniture_name = env_name.split("/")
    env = gym.make(
        env_id,
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
    env = FlattenWrapper(env)
    print(f"observation_space: {env.observation_space}")
    print(f"action_space: {env.action_space}")
    obs = env.reset()
    timestep = 0
    print(f"timestep {timestep} / obs: {obs.shape}")
    for _ in range(630):
        obs, rew, done, info = env.step(env.action_space.sample())
        timestep += 1
        print(f"timestep {timestep} / obs: {obs.shape}")
        if done:
            break
