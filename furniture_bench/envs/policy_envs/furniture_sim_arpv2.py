import numpy as np
from gym import spaces
from pathlib import Path
from collections import deque
import sys

import torch
import clip

from furniture_bench.envs.legacy_envs.furniture_sim_legacy_env import FurnitureSimEnvLegacy  # Deprecated.
from furniture_bench.perception.image_utils import resize, resize_crop
from furniture_bench.robot.robot_state import filter_and_concat_robot_state

sys.path.append("/home/changyeon/ICML2024/BPref-v2/")
from bpref_v2.data.label_reward_furniturebench import load_reward_model  # noqa: E402
from bpref_v2.data.instruct import get_furniturebench_instruct  # noqa: E402


# class FurnitureSimImageFeature(FurnitureSimEnv):
class FurnitureSimARPV2(FurnitureSimEnvLegacy):
    def __init__(self, **kwargs):
        super().__init__(
            concat_robot_state=True,
            resize_img=False,
            np_step_out=True,
            channel_first=True,
            **kwargs,
        )

        device_id = kwargs["compute_device_id"]
        self._device = torch.device(f"cuda:{device_id}")
        assert self.num_envs == 1, "FurnitureSimImageFeature supports only 1 env."
        from vip import load_vip

        self.vip_layer = load_vip().module
        self.vip_layer.requires_grad_(False)
        self.vip_layer.eval()
        self.vip_layer = self.vip_layer.to(self._device)

        self.embedding_dim = 1024

        from liv import load_liv

        self.liv_layer = load_liv().module
        self.liv_layer.requires_grad_(False)
        self.liv_layer.eval()
        self.liv_layer = self.liv_layer.to(self._device)

        import jax

        jax.config.update("jax_default_device", jax.devices()[device_id])

        self._reward_model = load_reward_model(
            rm_type=kwargs["rm_type"], ckpt_path=Path(kwargs["rm_ckpt_path"]).expanduser()
        )

        self.i = 0
        self._window_size = kwargs["window_size"]
        self._skip_frame = kwargs["skip_frame"]
        self._frames = {
            frame: {
                key: deque([], maxlen=self._window_size)
                for key in ["color_image2", "color_image1", "timestep", "attn_mask"]
            }
            for frame in range(self._skip_frame)
        }

        self.phase = 0
        self._current_instruct = self._get_instruct_feature(self.phase)
        self._lambda_mr = kwargs.get("lambda_mr", 0.01)

    @property
    def observation_space(self):
        robot_state_dim = 14

        return spaces.Dict(
            dict(
                robot_state=spaces.Box(-np.inf, np.inf, (robot_state_dim,)),
                image1=spaces.Box(-np.inf, np.inf, (self.embedding_dim,)),
                image2=spaces.Box(-np.inf, np.inf, (self.embedding_dim,)),
            )
        )

    def _get_observation(self):
        obs = super()._get_observation()

        if isinstance(obs["robot_state"], dict):
            # For legacy envs.
            obs["robot_state"] = filter_and_concat_robot_state(obs["robot_state"])

        robot_state = obs["robot_state"].squeeze()
        image1 = obs["color_image1"].squeeze()
        image2 = obs["color_image2"].squeeze()

        image1 = np.moveaxis(resize(np.moveaxis(image1, 0, -1)), -1, 0)
        crop_image2 = resize_crop(np.moveaxis(image2, 0, -1))
        image2 = np.moveaxis(crop_image2, -1, 0)

        return dict(robot_state=robot_state, color_image1=image1, color_image2=image2)

    def _batchify(self, data: np.asarray):
        return np.stack(data)[None, ...]

    def _stack_observation(self, obs):
        stack = self._frames[self.i % self._skip_frame]
        for key in obs.keys():
            assert len(stack[key]) == self._window_size, f"{len(stack[key])} != {self._window_size}"
            obs[key] = self._batchify(stack[key])
        return obs

    def _extract_vip_feature(self, obs):
        image1, image2, robot_state = obs["color_image1"], obs["color_image2"], obs["robot_state"]
        with torch.no_grad():
            image1 = torch.tensor(image1).to(self._device)
            image2 = torch.tensor(image2).to(self._device)

            image1 = self.vip_layer(image1.unsqueeze(0)).squeeze()
            image2 = self.vip_layer(image2.unsqueeze(0)).squeeze()
            image1 = image1.detach().cpu().numpy()
            image2 = image2.detach().cpu().numpy()

        return dict(robot_state=robot_state, image1=image1, image2=image2)

    def _extract_liv_feature(self, obs):
        image1, image2, robot_state = obs["color_image1"], obs["color_image2"], obs["robot_state"]
        with torch.no_grad():
            image1 = torch.tensor(image1).to(self._device)
            image2 = torch.tensor(image2).to(self._device)

            image1 = self.liv_layer(image1.unsqueeze(0), normalize=False).squeeze()
            image2 = self.liv_layer(image2.unsqueeze(0), normalize=False).squeeze()
            image1 = image1.detach().cpu().numpy()
            image2 = image2.detach().cpu().numpy()

        return dict(robot_state=robot_state, color_image1=image1, color_image2=image2)

    def _get_instruct_feature(self, phase):
        instruct = get_furniturebench_instruct(self.furniture_name, phase, output_type="all")
        tokens = clip.tokenize(instruct).to(self._device)
        return self.liv_layer(input=tokens, modality="text").detach().cpu().numpy()

    def reset(self):
        self.i = 0
        self.phase = 0
        self._current_instruct = self._get_instruct_feature(self.phase)
        obs = super().reset()
        for frame in range(self._skip_frame):
            for _ in range(self._window_size):
                for key in ["color_image2", "color_image1"]:
                    self._frames[frame][key].append(np.zeros((self.embedding_dim,)))
                self._frames[frame]["timestep"].append(np.asarray(0).astype(np.int32))
                self._frames[frame]["attn_mask"].append(np.asarray(0).astype(np.int32))

        liv_feat = self._extract_liv_feature(obs)
        for key in ["color_image2", "color_image1"]:
            self._frames[0][key].append(liv_feat[key])
        self._frames[0]["timestep"].append(np.asarray(self.i).astype(np.int32))
        self._frames[0]["attn_mask"].append(np.asarray(1).astype(np.int32))

        return self._extract_vip_feature(obs)

    def _compute_reward(self):
        stack = self._frames[self.i % self._skip_frame]
        stacked_obs = {key: self._batchify(stack[key]) for key in ["color_image2", "color_image1"]}
        new_instruct = self._get_instruct_feature(self.phase + 1)

        # print(f"stacked_obs: {[(key, val.shape) for key, val in stacked_obs.items()]}")
        # print(f"instruct: {self._current_instruct.shape}")
        batch = {
            "image": stacked_obs,
            "instruct": self._current_instruct[None, ...],
            "timestep": self._batchify(stack["timestep"]),
            "attn_mask": self._batchify(stack["attn_mask"]),
        }
        current_reward = np.asarray(self._reward_model.get_reward(batch))
        batch.update(instruct=new_instruct[None, ...])
        next_reward = np.asarray(self._reward_model.get_reward(batch))
        reward = max(current_reward, next_reward) * self._lambda_mr
        if next_reward > current_reward:
            self._current_instruct = new_instruct
            self.phase += 1
        return reward

    def step(self, action):
        self.i += 1
        obs, reward, done, info = super().step(action)

        liv_feat = self._extract_liv_feature(obs)
        stack = self._frames[self.i % self._skip_frame]
        for key in ["color_image2", "color_image1"]:
            stack[key].append(liv_feat[key])
        stack["timestep"].append(np.asarray(self.i).astype(np.int32))
        stack["attn_mask"].append(np.asarray(1).astype(np.int32))

        reward = self._compute_reward()
        return self._extract_vip_feature(obs), reward, done, info


if __name__ == "__main__":
    import furniture_bench  # noqa: F401
    import gym

    env_name = "FurnitureSimARPV2-v0/one_leg"
    # env_name = "FurnitureSim-v0/one_leg"
    env_id, furniture_name = env_name.split("/")
    num_frames, skip_frame, target_keys = 4, 16, ["image1", "image2"]
    use_arp_reward = True
    rm_type = "ARP-V2"
    rm_ckpt_path = Path(
        "/mnt/changyeon/ICML2024/arp_v2/reward_learning/furniturebench-one_leg/ARP-V2/w4-s16-nfp1.0-liv0.0-c1.0-ep1.0-aug_none-liv-img2+1-legacy-withlogit100-demo1000/s0/best_model.pkl"
    ).expanduser()

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
        window_size=num_frames,
        skip_frame=skip_frame,
        rm_type=rm_type,
        rm_ckpt_path=rm_ckpt_path
        # concat_robot_state=True,
    )

    init = env.reset()
    timestep = 0
    print(f"timestep {timestep} / stack step {env.i} / phase {env.phase}")
    for _ in range(630):
        res, rew, done, info = env.step(env.action_space.sample())
        timestep += 1
        print(f"timestep {timestep} / stack step {env.i} / phase {env.phase}")
        if done:
            break
