# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import io
import random
import traceback
import collections
from typing import Sequence
import datetime
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset

from dataset_utils import exponential_moving_average, transform_phases, quat_to_theta

Batch = collections.namedtuple("Batch", ["observations", "actions", "rewards", "masks", "next_observations"])
SHORTEST_PATHS = {"one_leg": 402, "cabinet": 816, "lamp": 611, "round_table": 784}
PHASE_TO_REWARD = {"one_leg": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: -1}}


def episode_len(episode):
    # subtract -1 because the dummy last transition
    return np.asarray(next(iter(episode.values()))).shape[0]


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open("wb") as f:
            f.write(bs.read())


def _get_stacked_timesteps(length, window_size, skip_frame):
    stacked_timesteps = []
    timestep_stacks = {key: deque([], maxlen=window_size) for key in range(skip_frame)}
    for _ in range(window_size):
        for j in range(skip_frame):
            timestep_stacks[j].append(0)

    for i in range(length):
        mod = i % skip_frame
        timestep_stack = timestep_stacks[mod]
        timestep_stack.append(i)
        stacked_timesteps.append(np.stack(timestep_stack))

    return stacked_timesteps


def load_episode(
    fn,
    furniture="one_leg",
    reward_type="sparse",
    discount=0.99,
    obs_keys=("image1", "image2"),
    window_size=4,
    skip_frame=4,
    lambda_mr=1.0,
    reward_stat: dict = None,
    smoothe: bool = False,
):
    observations, next_observations, timesteps = [], [], []
    with fn.open("rb") as f:
        episode = np.load(f, allow_pickle=True)
        episode = {k: episode[k] for k in episode.keys()}
        eps_len = episode_len(episode)
        stacked_timesteps = _get_stacked_timesteps(eps_len, window_size, skip_frame)
        for i in range(eps_len):
            observations.append(np.concatenate([episode["observations"][i][key] for key in obs_keys], axis=-1))
            timesteps.append(stacked_timesteps[i])
            next_observations.append(
                np.concatenate([episode["next_observations"][i][key] for key in obs_keys], axis=-1)
            )
        if reward_type == "sparse":
            rewards = episode["rewards"]
        elif reward_type == "step":
            rewards = episode["step_rewards"] / np.max(episode["step_rewards"])
        elif reward_type == "viper":
            rewards = episode["viper_rewards"] / lambda_mr
        elif reward_type == "diffusion":
            rewards = episode["diffusion_rewards"] / lambda_mr
        elif reward_type == "ours":
            _min, _max = reward_stat["min"], reward_stat["max"]
            # rewards = (episode["multimodal_rewards"] - _min) / (_max - _min)
            rewards = episode["multimodal_rewards"] - _min
            rewards = rewards / lambda_mr
            rewards = rewards + (episode["rewards"] / lambda_mr)
            if smoothe:
                rewards = np.asarray(exponential_moving_average(rewards))
        elif reward_type == "ours_shaped":
            phases = episode["phases"]
            rewards = [PHASE_TO_REWARD[furniture][p] for p in phases]
            rewards = np.asarray(rewards, dtype=np.float32)
            # rewards = transform_phases(rewards + (episode["rewards"] / lambda_mr))

            # our_reward = episode["multimodal_rewards"] / lambda_mr
            # next_our_reward = np.asarray(our_reward[1:].tolist() + our_reward[-1:].tolist())
            # delta_our_reward = discount * next_our_reward - our_reward
            # delta_our_reward[-1] = 0.0
            # rewards = delta_our_reward + episode["rewards"]

    return dict(
        observations=np.asarray(observations),
        timesteps=np.asarray(timesteps),
        actions=episode["actions"],
        rewards=rewards,
        masks=1.0 - episode["terminals"],
        dones_float=episode["terminals"],
        next_observations=np.asarray(next_observations),
    )


class ReplayBufferStorage:
    def __init__(self, replay_dir, max_env_steps):
        self._replay_dir = replay_dir
        replay_dir.mkdir(exist_ok=True)
        self._max_env_steps = max_env_steps
        self._preload()

    def __len__(self):
        return self._num_transitions

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.glob("*.npz"):
            _, _, eps_len, *_ = fn.stem.split("_")
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def add_episode(self, episode, env_idx, episode_idx):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        tp = "success" if np.sum(episode["rewards"]) > 0.0 else "failure"
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        eps_fn = f"{ts}_{tp}_{eps_idx}_{eps_len}_env{env_idx}_{episode_idx}.npz"
        save_episode(episode, self._replay_dir / eps_fn)


class ReplayBuffer(IterableDataset):
    def __init__(
        self,
        furniture,
        replay_dir,
        max_size,
        num_workers,
        nstep,
        discount,
        fetch_every,
        save_snapshot=False,
        reward_type: str = "sparse",
        embedding_dim: int = 1024,
        num_demos: dict = None,
        obs_keys: Sequence[str] = ("image1", "image2"),
        window_size: int = 4,
        skip_frame: int = 4,
        lambda_mr: float = 1.0,
        reward_stat: dict = None,
        action_stat: dict = None,
        smoothe: bool = False,
        prefill_replay_buffer: bool = False,
        offline_replay_dir: str = None,
    ):
        self._furniture = furniture
        self._replay_dir = replay_dir
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._episode_path = dict()
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot
        self._reward_type = reward_type
        self._embedding_dim = embedding_dim
        self._num_demos = num_demos
        self._window_size = window_size
        self._skip_frame = skip_frame
        self._obs_keys = obs_keys
        self._lambda_mr = lambda_mr
        self._reward_stat = reward_stat
        self._action_stat = action_stat
        self._smoothe = smoothe
        if prefill_replay_buffer:
            print(f"prefill replay buffer with {self._num_demos}")
            self._offline_replay_dir = offline_replay_dir
            self._try_offline_fetch()

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn, tp="online"):
        try:
            episode = load_episode(
                eps_fn,
                furniture=self._furniture,
                reward_type=self._reward_type,
                discount=self._discount,
                obs_keys=self._obs_keys,
                window_size=self._window_size,
                skip_frame=self._skip_frame,
                lambda_mr=self._lambda_mr,
                reward_stat=self._reward_stat,
            )
        except Exception as e:
            print(f"Failed to load {eps_fn}: {e}")
            return False
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            if not self._save_snapshot:
                self._episode_path[early_eps_fn].unlink(missing_ok=True)
        if tp == "offline":
            tn = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
            eps_key = f"{tn}_{Path(eps_fn).name}"
        else:
            eps_key = Path(eps_fn).name
        self._episode_fns.append(eps_key)
        self._episode_fns.sort()
        self._episodes[eps_key] = episode
        self._episode_path[eps_key] = eps_fn
        self._size += eps_len

        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True

    def _try_offline_fetch(self):
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except Exception:
            worker_id = 0
        for key, num_demo in self._num_demos.items():
            eps_fns = sorted(self._offline_replay_dir.glob(f"{key}_*.npz"), reverse=True)
            for eps_fn in eps_fns[:num_demo]:
                eps_tp, eps_idx, eps_len = eps_fn.stem.split("_")
                eps_idx, eps_len = map(lambda x: int(x), [eps_idx, eps_len])
                assert eps_tp == key, f"{eps_fn} is not {key}."
                if eps_idx % self._num_workers != worker_id:
                    continue
                if eps_fn in self._episodes.keys():
                    break
                if not self._store_episode(eps_fn, tp="offline"):
                    break

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except Exception:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.glob("*.npz"), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            _, eps_tp, eps_idx, eps_len, *_ = eps_fn.stem.split("_")
            eps_idx, eps_len = map(lambda x: int(x), [eps_idx, eps_len])
            if eps_idx % self._num_workers != worker_id:
                continue
            eps_fn_key = Path(eps_fn).name
            if eps_fn_key in self._episodes.keys():
                break
            if fetched_size + eps_len > self._max_size:
                break
            fetched_size += eps_len
            if not self._store_episode(eps_fn):
                break

    def _sample(self):
        try:
            self._try_fetch()
        except Exception:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        return self.__sample()

    def _split_observations(self, obs):
        if obs.shape[-1] == self._embedding_dim * 2:
            image1, image2 = np.split(obs, [self._embedding_dim], axis=-1)
            return dict(image1=image1, image2=image2)
        elif obs.shape[-1] == self._embedding_dim * 2 + 14:
            image1, image2, robot_state = np.split(obs, [self._embedding_dim, self._embedding_dim * 2], axis=-1)
            return dict(image1=image1, image2=image2, robot_state=robot_state)
        elif obs.shape[-1] == self._embedding_dim * 3:
            image1, image2, text_feature = np.split(obs, [self._embedding_dim, self._embedding_dim * 2], axis=-1)
            return dict(image1=image1, image2=image2, text_feature=text_feature)
        else:  # self._embedding_dim * 2 + 14 + self._text_feature_dim
            image1, image2, robot_state, text_feature = np.split(
                obs, [self._embedding_dim, self._embedding_dim * 2, self._embedding_dim * 2 + 14], axis=-1
            )
            return dict(image1=image1, image2=image2, robot_state=robot_state, text_feature=text_feature)

    def __sample(self):
        episode = self._sample_episode()
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1)
        obs = episode["observations"][episode["timesteps"][idx]]

        # action normalization!
        action = episode["actions"][idx]
        # Online trajectory will include action with theta, so we don't need to convert it to quat and normalize.
        if action.shape[-1] == 8:
            delta_pose, delta_quat, gripper_pose = action[:3], action[3:7], action[7:]
            delta_theta = quat_to_theta(delta_quat)
            action = np.concatenate([delta_pose, delta_theta, gripper_pose], axis=-1)
            action = (
                2 * ((action - self._action_stat["low"]) / (self._action_stat["high"] - self._action_stat["low"])) - 1
            )

        next_obs = episode["next_observations"][episode["timesteps"][idx + self._nstep - 1]]
        reward = np.zeros_like(episode["rewards"][idx])
        discount = np.ones_like(episode["masks"][idx])
        for i in range(self._nstep):
            step_reward = episode["rewards"][idx + i]
            reward += discount * step_reward
            discount *= episode["masks"][idx + i] * self._discount

        observation = self._split_observations(obs)
        next_observation = self._split_observations(next_obs)

        return Batch(
            observations=observation,
            actions=action,
            rewards=reward,
            masks=discount,
            next_observations=next_observation,
        )

    def __iter__(self):
        while True:
            yield self._sample()


class OfflineReplayBuffer(IterableDataset):
    def __init__(
        self,
        furniture,
        replay_dir,
        max_size,
        num_workers,
        nstep,
        discount,
        fetch_every,
        save_snapshot=True,
        reward_type: str = "sparse",
        embedding_dim: int = 1024,
        num_demos: dict = None,
        obs_keys: Sequence[str] = ("image1", "image2", "text_feature"),
        window_size: int = 4,
        skip_frame: int = 4,
        lambda_mr: float = 1.0,
        reward_stat: dict = None,
        action_stat: dict = None,
        smoothe: bool = False,
    ):
        self._furniture = furniture
        self._replay_dir = replay_dir
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot
        self._reward_type = reward_type
        self._embedding_dim = embedding_dim
        self._num_demos = num_demos
        self._obs_keys = obs_keys
        self._window_size = window_size
        self._skip_frame = skip_frame
        self._lambda_mr = lambda_mr
        self._action_stat = action_stat
        self._reward_stat = reward_stat
        self._smoothe = smoothe
        self._try_fetch()

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(
                eps_fn,
                furniture=self._furniture,
                reward_type=self._reward_type,
                discount=self._discount,
                obs_keys=self._obs_keys,
                window_size=self._window_size,
                skip_frame=self._skip_frame,
                lambda_mr=self._lambda_mr,
                reward_stat=self._reward_stat,
            )
        except Exception as e:
            print(f"Failed to load {eps_fn}: {e}")
            return False
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            if not self._save_snapshot:
                early_eps_fn.unlink(missing_ok=True)
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len

        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True

    def _try_fetch(self):
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except Exception:
            worker_id = 0
        for key, num_demo in self._num_demos.items():
            eps_fns = sorted(self._replay_dir.glob(f"{key}_*.npz"), reverse=True)
            for eps_fn in eps_fns[:num_demo]:
                eps_tp, eps_idx, eps_len = eps_fn.stem.split("_")
                eps_idx, eps_len = map(lambda x: int(x), [eps_idx, eps_len])
                assert eps_tp == key, f"{eps_fn} is not {key}."
                if eps_idx % self._num_workers != worker_id:
                    continue
                if eps_fn in self._episodes.keys():
                    break
                if not self._store_episode(eps_fn):
                    break

    def _split_observations(self, obs):
        if obs.shape[-1] == self._embedding_dim * 2:
            image1, image2 = np.split(obs, [self._embedding_dim], axis=-1)
            return dict(image1=image1, image2=image2)
        elif obs.shape[-1] == self._embedding_dim * 2 + 14:
            image1, image2, robot_state = np.split(obs, [self._embedding_dim, self._embedding_dim * 2], axis=-1)
            return dict(image1=image1, image2=image2, robot_state=robot_state)
        elif obs.shape[-1] == self._embedding_dim * 3:
            image1, image2, text_feature = np.split(obs, [self._embedding_dim, self._embedding_dim * 2], axis=-1)
            return dict(image1=image1, image2=image2, text_feature=text_feature)
        else:  # self._embedding_dim * 2 + 14 + self._text_feature_dim
            image1, image2, robot_state, text_feature = np.split(
                obs, [self._embedding_dim, self._embedding_dim * 2, self._embedding_dim * 2 + 14], axis=-1
            )
            return dict(image1=image1, image2=image2, robot_state=robot_state, text_feature=text_feature)

    def _sample(self):
        episode = self._sample_episode()
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1)
        obs = episode["observations"][episode["timesteps"][idx]]

        # action normalization!
        action = episode["actions"][idx]
        delta_pose, delta_quat, gripper_pose = action[:3], action[3:7], action[7:]
        delta_theta = quat_to_theta(delta_quat)
        action = np.concatenate([delta_pose, delta_theta, gripper_pose], axis=-1)
        action = 2 * ((action - self._action_stat["low"]) / (self._action_stat["high"] - self._action_stat["low"])) - 1

        next_obs = episode["next_observations"][episode["timesteps"][idx + self._nstep - 1]]
        reward = np.zeros_like(episode["rewards"][idx])
        discount = np.ones_like(episode["masks"][idx])
        for i in range(self._nstep):
            step_reward = episode["rewards"][idx + i]
            reward += discount * step_reward
            discount *= episode["masks"][idx + i] * self._discount

        observation = self._split_observations(obs)
        next_observation = self._split_observations(next_obs)

        return Batch(
            observations=observation,
            actions=action,
            rewards=reward,
            masks=discount,
            next_observations=next_observation,
        )

    def __iter__(self):
        while True:
            yield self._sample()


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_replay_loader(
    furniture,
    replay_dir,
    max_size,
    batch_size,
    num_workers,
    save_snapshot,
    nstep,
    discount,
    buffer_type: str = "offline",
    reward_type: str = "sparse",
    num_demos: dict = None,
    **kwargs,
):
    max_size_per_worker = max_size // max(1, num_workers)

    if buffer_type == "online":
        iterable = ReplayBuffer(
            furniture,
            replay_dir,
            max_size_per_worker,
            num_workers,
            nstep,
            discount,
            fetch_every=500,
            save_snapshot=save_snapshot,
            reward_type=reward_type,
            num_demos=num_demos,
            **kwargs,
        )
    elif buffer_type == "offline":
        iterable = OfflineReplayBuffer(
            furniture,
            replay_dir,
            max_size_per_worker,
            num_workers,
            nstep,
            discount,
            fetch_every=1e6,
            save_snapshot=save_snapshot,
            reward_type=reward_type,
            num_demos=num_demos,
            **kwargs,
        )

    loader = torch.utils.data.DataLoader(
        iterable, batch_size=batch_size, num_workers=num_workers, pin_memory=True, worker_init_fn=_worker_init_fn
    )
    return loader
