import isaacgym  # noqa: F401

import numpy as np
from gym import spaces
from pathlib import Path
from collections import deque
import sys

import torch
import clip
from kornia.augmentation import Resize, CenterCrop

from furniture_bench.envs.furniture_sim_env import FurnitureSimEnv
from furniture_bench.robot.robot_state import filter_and_concat_robot_state

from bpref_v2.data.instruct import get_furniturebench_instruct
from bpref_v2.data.label_reward_furniturebench import load_reward_model


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

        self.alpha = 1.0
        self.i = {env_idx: 0 for env_idx in range(self.num_envs)}
        self._eta = 10
        self._negative_trend_eta = 1e-2

        self._prev_reward = {env_idx: 0 for env_idx in range(self.num_envs)}
        self._task_pass_threshold = {env_idx: 0 for env_idx in range(self.num_envs)}
        self._task_fail_threshold = {env_idx: 0 for env_idx in range(self.num_envs)}
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

        # self.phase = {env_idx: 0 for env_idx in range(self.num_envs)}
        # self._current_instruct = {
        #     env_idx: self._get_instruct_feature(self.phase[env_idx]) for env_idx in range(self.num_envs)
        # }
        self._lambda_mr = kwargs.get("lambda_mr", 0.1)

        if not self._resize_img:
            self.resize = Resize((224, 224))
            img_size = self.img_size
            ratio = 256 / min(img_size[0], img_size[1])
            ratio_size = (int(img_size[1] * ratio), int(img_size[0] * ratio))
            self.resize_crop = torch.nn.Sequential(Resize(ratio_size), CenterCrop((224, 224)))

    @property
    def observation_space(self):
        robot_state_dim = 14

        return spaces.Dict(
            dict(
                robot_state=spaces.Box(-np.inf, np.inf, (robot_state_dim,)),
                image1=spaces.Box(-np.inf, np.inf, (self.embedding_dim,)),
                image2=spaces.Box(-np.inf, np.inf, (self.embedding_dim,)),
                text_feature=spaces.Box(-np.inf, np.inf, (self.embedding_dim,)),
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

    def _extract_vip_feature(self, obs):
        image1, image2, robot_state = obs["color_image1"], obs["color_image2"], obs["robot_state"]
        with torch.no_grad():
            image1 = torch.tensor(image1).to(self._device)
            image2 = torch.tensor(image2).to(self._device)

            image1 = self.vip_layer(image1).detach().cpu().numpy()
            image2 = self.vip_layer(image2).detach().cpu().numpy()

        return dict(robot_state=robot_state, image1=image1, image2=image2)

    def _extract_liv_feature(self, obs):
        image1, image2, robot_state = obs["color_image1"], obs["color_image2"], obs["robot_state"]
        with torch.no_grad():
            image1 = torch.tensor(image1).to(self._device)
            image2 = torch.tensor(image2).to(self._device)

            image1 = self.liv_layer(image1).detach().cpu().numpy()
            image2 = self.liv_layer(image2).detach().cpu().numpy()

        return dict(robot_state=robot_state, color_image1=image1, color_image2=image2)

    def _get_instruct_feature(self, phase):
        instruct = get_furniturebench_instruct(self.furniture_name, phase, output_type="all")
        tokens = clip.tokenize(instruct).detach().cpu().numpy()
        # return self.liv_layer(input=tokens, modality="text").detach().cpu().numpy()
        return np.asarray(self._reward_model.get_text_feature({"instruct": tokens}))

    def _check_reward_condition(self, env_idx, current_reward, next_reward):
        if (
            current_reward < self._prev_reward[env_idx]
            and abs(current_reward - self._prev_reward[env_idx]) > self._negative_trend_eta
        ):
            return True, False
        if next_reward > current_reward:
            return False, True
        return False, False

    def _save_prev_reward(self, reward):
        for env_idx in range(reward.shape[0]):
            self._prev_reward[env_idx] = (
                (1 - self.alpha) * self._prev_reward[env_idx] + self.alpha * reward[env_idx]
            ) / self._lambda_mr

    def reset_env(self, idx):
        super().reset_env(idx)
        super().refresh()
        self.i[idx] = 0
        # self.phase[idx] = 0
        # self._prev_reward[idx] = 0
        # self._task_pass_threshold[idx] = 0
        # self._task_fail_threshold[idx] = 0

        self.__frames[idx] = {
            frame: {
                key: deque([], maxlen=self._window_size)
                for key in ["color_image2", "color_image1", "timestep", "attn_mask"]
            }
            for frame in range(self._skip_frame)
        }
        _obs = super()._get_observation(video=False)
        liv_feat = self._extract_liv_feature(_obs)
        stack = self.__frames[idx]
        for frame in range(self._skip_frame):
            for _ in range(self._window_size):
                for key in ["color_image2", "color_image1"]:
                    stack[frame][key].append(np.zeros((self.embedding_dim,)))
                stack[frame]["timestep"].append(np.asarray(0).astype(np.int32))
                stack[frame]["attn_mask"].append(np.asarray(0).astype(np.int32))
        for key in ["color_image2", "color_image1"]:
            stack[0][key].append(liv_feat[key][idx])
        stack[0]["timestep"].append(np.asarray(self.i[idx]).astype(np.int32))
        stack[0]["attn_mask"].append(np.asarray(1).astype(np.int32))
        return self._extract_vip_feature(_obs)

    def reset(self):
        self.i = {env_idx: 0 for env_idx in range(self.num_envs)}
        # self.phase = {env_idx: 0 for env_idx in range(self.num_envs)}
        # self._prev_reward = {env_idx: 0 for env_idx in range(self.num_envs)}
        # self._task_pass_threshold = {env_idx: 0 for env_idx in range(self.num_envs)}
        # self._task_fail_threshold = {env_idx: 0 for env_idx in range(self.num_envs)}

        # self._current_instruct = {
        #     env_idx: self._get_instruct_feature(self.phase[env_idx]) for env_idx in range(self.num_envs)
        # }
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

        obs = super().reset()
        liv_feat = self._extract_liv_feature(obs)
        for env_idx in range(self.num_envs):
            for frame in range(self._skip_frame):
                for _ in range(self._window_size):
                    for key in ["color_image2", "color_image1"]:
                        self.__frames[env_idx][frame][key].append(np.zeros((self.embedding_dim,)))
                    self.__frames[env_idx][frame]["timestep"].append(np.asarray(0).astype(np.int32))
                    self.__frames[env_idx][frame]["attn_mask"].append(np.asarray(0).astype(np.int32))

            for key in ["color_image2", "color_image1"]:
                self.__frames[env_idx][0][key].append(liv_feat[key][env_idx])
            self.__frames[env_idx][0]["timestep"].append(np.asarray(0).astype(np.int32))
            self.__frames[env_idx][0]["attn_mask"].append(np.asarray(1).astype(np.int32))

        return self._extract_vip_feature(obs)

    def _predict_phase(self):
        stacked_obs = {key: [] for key in ["color_image2", "color_image1"]}
        stacked_timestep, stacked_attn_mask = [], []
        for env_idx in range(self.num_envs):
            stack = self.__frames[env_idx][self.i[env_idx] % self._skip_frame]
            for key in ["color_image2", "color_image1"]:
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
        liv_feat = self._extract_liv_feature(obs)

        for env_idx in range(self.num_envs):
            self.i[env_idx] += 1
            stack = self.__frames[env_idx][self.i[env_idx] % self._skip_frame]
            for key in ["color_image2", "color_image1"]:
                stack[key].append(liv_feat[key][env_idx])
            stack["timestep"].append(np.asarray(self.i[env_idx]).astype(np.int32))
            stack["attn_mask"].append(np.asarray(1).astype(np.int32))

        phase = self._predict_phase()
        text_feature = np.asarray([self._text_features[key] for key in phase])
        vip_obs = self._extract_vip_feature(obs)
        vip_obs.update(dict(text_feature=text_feature))
        return vip_obs, task_reward, done, info


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
        "/mnt/changyeon/ICML2024/new_arp_v2/reward_learning/furniturebench-one_leg/ARP-V2/w4-s4-nfp1.0-c1.0@0.5-supc1.0-ep0.1-demo500-total-phase/s0/best_model.pkl"
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
