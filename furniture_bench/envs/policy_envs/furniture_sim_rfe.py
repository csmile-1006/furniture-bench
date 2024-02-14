import isaacgym  # noqa: F401

import numpy as np
from gym import spaces
from pathlib import Path
from collections import deque

import torch
import clip
from kornia.augmentation import Resize, CenterCrop

from furniture_bench.envs.furniture_sim_env import FurnitureSimEnv
from furniture_bench.robot.robot_state import filter_and_concat_robot_state
from furniture_bench.envs.initialization_mode import load_embedding

from bpref_v2.data.instruct import get_furniturebench_instruct


class FurnitureSimRFE(FurnitureSimEnv):
    def __init__(self, **kwargs):
        super().__init__(
            concat_robot_state=True,
            np_step_out=True,
            channel_first=True,
            **kwargs,
        )
        self._resize_img = kwargs["resize_img"]

        device_id = kwargs["compute_device_id"]
        self._device = torch.device(f"cuda:{device_id}")
        self._compute_text_feature = False

        self.img_emb_layer, self.embedding_dim = load_embedding(kwargs["encoder_type"], self._device)
        self.reward_img_emb_layer, _ = load_embedding(kwargs["reward_encoder_type"], self._device)

        import jax

        jax.config.update("jax_default_device", jax.devices()[device_id])

        self._reward_model = kwargs["reward_model"]
        self.i = {env_idx: 0 for env_idx in range(self.num_envs)}

        self._window_size = kwargs["window_size"]
        self._skip_frame = kwargs["skip_frame"]
        self.__frames = {
            env_idx: {
                frame: {
                    key: deque([], maxlen=self._window_size)
                    for key in ["color_image2", "color_image1", "timestep", "attn_mask"]
                }
                for frame in range(self._skip_frame)
            }
            for env_idx in range(self.num_envs)
        }
        self._task_phases = kwargs.get("task_phases", 5)
        self._text_features = {key: self._get_instruct_feature(key) for key in range(self._task_phases)}

        if not self._resize_img:
            self.resize = Resize((224, 224))
            img_size = self.img_size
            ratio = 256 / min(img_size[0], img_size[1])
            ratio_size = (int(img_size[1] * ratio), int(img_size[0] * ratio))
            self.resize_crop = torch.nn.Sequential(Resize(ratio_size), CenterCrop((224, 224)))

    def compute_text_feature(self):
        self._compute_text_feature = True

    def uncompute_text_feature(self):
        self._compute_text_feature = False

    @property
    def observation_space(self):
        robot_state_dim = 14
        return spaces.Dict(
            dict(
                robot_state=spaces.Box(-np.inf, np.inf, (robot_state_dim,)),
                image1=spaces.Box(-np.inf, np.inf, (self.embedding_dim,)),
                image2=spaces.Box(-np.inf, np.inf, (self.embedding_dim,)),
                text_feature=spaces.Box(-np.inf, np.inf, (self.embedding_dim,)),
                color_image1=spaces.Box(-np.inf, np.inf, (self.embedding_dim,)),
                color_image2=spaces.Box(-np.inf, np.inf, (self.embedding_dim,)),
            )
        )

    def _get_observation(self):
        obs = super()._get_observation()

        if isinstance(obs["robot_state"], dict):
            # For legacy envs.
            obs["robot_state"] = filter_and_concat_robot_state(obs["robot_state"])

        robot_state = obs["robot_state"]
        image1 = obs["color_image1"]
        image2 = obs["color_image2"]

        if not self._resize_img:
            with torch.no_grad():
                image1 = self.resize(torch.tensor(image1).float()).detach().cpu().numpy()
                image2 = self.resize_crop(torch.tensor(image2).float()).detach().cpu().numpy()

        return dict(robot_state=robot_state, color_image1=image1, color_image2=image2)

    def _batchify(self, data: np.asarray):
        return np.stack(data)

    def _extract_img_feature(self, layer, obs):
        image1, image2, robot_state = obs["color_image1"], obs["color_image2"], obs["robot_state"]
        with torch.no_grad():
            image1 = torch.tensor(image1).to(self._device)
            image2 = torch.tensor(image2).to(self._device)

            image1 = layer(image1).detach().cpu().numpy()
            image2 = layer(image2).detach().cpu().numpy()

        return dict(robot_state=robot_state, image1=image1, image2=image2)

    def _get_instruct_feature(self, phase):
        instruct = get_furniturebench_instruct(self.furniture_name, phase, output_type="all")
        tokens = clip.tokenize(instruct).detach().cpu().numpy()
        return np.asarray(self._reward_model.get_text_feature({"instruct": tokens}))

    def reset_env(self, idx):
        super().reset_env(idx)
        super().refresh()
        obs = super()._get_observation(video=False)
        reward_img_feat = self._extract_img_feature(self.reward_img_emb_layer, obs)
        if self._compute_text_feature:
            self.i[idx] = 0
            self.__frames[idx] = {
                frame: {
                    key: deque([], maxlen=self._window_size)
                    for key in ["color_image2", "color_image1", "timestep", "attn_mask"]
                }
                for frame in range(self._skip_frame)
            }
            stack = self.__frames[idx]
            for frame in range(self._skip_frame):
                for _ in range(self._window_size):
                    for key in ["color_image2", "color_image1"]:
                        stack[frame][key].append(np.zeros((self.embedding_dim,)))
                    stack[frame]["timestep"].append(np.asarray(0).astype(np.int32))
                    stack[frame]["attn_mask"].append(np.asarray(0).astype(np.int32))
            for src_key, dst_key in [("image2", "color_image2"), ("image1", "color_image1")]:
                stack[0][dst_key].append(reward_img_feat[src_key][idx])
            stack[0]["timestep"].append(np.asarray(self.i[idx]).astype(np.int32))
            stack[0]["attn_mask"].append(np.asarray(1).astype(np.int32))
            phases = np.tile(self._predict_phase(idx), (self.num_envs,))
            text_feature = np.asarray([self._text_features[phase] for phase in phases])
        else:
            text_feature = np.zeros((self.num_envs, self.embedding_dim))

        new_obs = self._extract_img_feature(self.img_emb_layer, obs)
        new_obs.update(dict(text_feature=text_feature))
        new_obs.update(
            dict(
                color_image1=reward_img_feat["image1"],
                color_image2=reward_img_feat["image2"],
            )
        )
        return new_obs

    def reset(self):
        obs = super().reset()
        reward_img_feat = self._extract_img_feature(self.reward_img_emb_layer, obs)
        new_obs = self._extract_img_feature(self.img_emb_layer, obs)
        if self._compute_text_feature:
            phases = np.concatenate([self._predict_phase(env_idx) for env_idx in range(self.num_envs)])
            text_feature = np.asarray([self._text_features[phase] for phase in phases])
            new_obs.update(dict(text_feature=text_feature))
        else:
            new_obs.update(dict(text_feature=np.zeros((self.num_envs, self.embedding_dim))))

        new_obs.update(
            dict(
                color_image1=reward_img_feat["image1"],
                color_image2=reward_img_feat["image2"],
            )
        )
        return new_obs

    def _predict_phase(self, env_idx):
        stacked_obs = {key: [] for key in ["color_image2", "color_image1"]}
        stacked_timestep, stacked_attn_mask = [], []
        stack = self.__frames[env_idx][self.i[env_idx] % self._skip_frame]
        for key in stacked_obs:
            stacked_obs[key].append(self._batchify(stack[key]))
        stacked_timestep.append(self._batchify(stack["timestep"]))
        stacked_attn_mask.append(self._batchify(stack["attn_mask"]))
        stacked_obs = {key: np.stack(val) for key, val in stacked_obs.items()}
        stacked_timestep, stacked_attn_mask = np.stack(stacked_timestep), np.stack(stacked_attn_mask)
        batch = {
            "image": stacked_obs,
            "timestep": stacked_timestep,
            "attn_mask": stacked_attn_mask,
        }
        phases = self._reward_model.get_phase(batch)
        return np.asarray(phases)

    def step(self, action):
        obs, task_reward, done, info = super().step(action)
        reward_img_feat = self._extract_img_feature(self.reward_img_emb_layer, obs)
        new_obs = self._extract_img_feature(self.img_emb_layer, obs)

        if self._compute_text_feature:
            for env_idx in range(self.num_envs):
                self.i[env_idx] += 1
                stack = self.__frames[env_idx][self.i[env_idx] % self._skip_frame]
                for src_key, dst_key in [("image2", "color_image2"), ("image1", "color_image1")]:
                    stack[dst_key].append(reward_img_feat[src_key][env_idx])

                stack["timestep"].append(np.asarray(self.i[env_idx]).astype(np.int32))
                stack["attn_mask"].append(np.asarray(1).astype(np.int32))
            phases = np.concatenate([self._predict_phase(env_idx) for env_idx in range(self.num_envs)])
            text_feature = np.asarray([self._text_features[phase] for phase in phases])
            new_obs.update(dict(text_feature=text_feature))
            info.update({"phases": phases})
        else:
            new_obs.update(dict(text_feature=np.zeros((self.num_envs, self.embedding_dim))))

        new_obs.update(
            dict(
                color_image1=reward_img_feat["image1"],
                color_image2=reward_img_feat["image2"],
            )
        )
        return new_obs, task_reward, done, info


if __name__ == "__main__":
    import furniture_bench  # noqa: F401
    import gym

    env_name = "FurnitureSimRFE-v0/one_leg"
    # env_name = "FurnitureSim-v0/one_leg"
    env_id, furniture_name = env_name.split("/")
    num_frames, skip_frame = 4, 4
    use_ours_reward = True
    rm_type = "ARP-V2"
    rm_ckpt_path = Path(
        "/mnt/changyeon/ICML2024/reward_models/one_leg/w4-s4-nfp1.0-c1.0@0.5-supc1.0-ep0.5-demo100-total-phase/s0/best_model.pkl"
    ).expanduser()

    env = gym.make(
        env_id,
        num_envs=10,
        furniture=furniture_name,
        data_path="",
        use_encoder=False,
        encoder_type="vip",
        compute_device_id=0,
        graphics_device_id=0,
        headless=True,
        record=False,
        resize_img=True,
        record_dir="",
        max_env_steps=600 if "Sim" in env_id else 3000,
        window_size=num_frames,
        skip_frame=skip_frame,
        rm_type=rm_type,
        rm_ckpt_path=rm_ckpt_path,
        task_phases=5,
    )

    init = env.reset()
    timestep = 0
    print(f"timestep {timestep} / stack step {env.i}")
    for _ in range(630):
        res, rew, done, info = env.step(env.action_space.sample())
        timestep += 1
        print(f"timestep {timestep} / stack step {env.i} / rew: {rew} / text_feature: {res['text_feature']}")
        if np.any(done):
            break
