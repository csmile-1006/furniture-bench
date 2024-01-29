# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import io
import random
import traceback
import collections

import numpy as np
import torch
from torch.utils.data import IterableDataset

Batch = collections.namedtuple("Batch", ["observations", "actions", "rewards", "masks", "next_observations"])
SHORTEST_PATHS = {"one_leg": 402, "cabinet": 816, "lamp": 611, "round_table": 784}


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return np.asarray(next(iter(episode.values()))).shape[0] - 1


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open("wb") as f:
            f.write(bs.read())


def load_episode(fn, reward_type="sparse", discount=0.99):
    observations, next_observations, timesteps, next_timesteps = [], [], [], []
    with fn.open("rb") as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        eps_len = episode_len(episode)
        dones_float = np.zeros_like(episode["terminals"], dtype=np.float32)
        for i in range(eps_len):
            observations.append(
                np.concatenate(
                    [episode["observations"][i][key] for key in ["image1", "image2", "robot_state"]], axis=-1
                )
            )
            timesteps.append(episode["observations"][i]["timestep"])
            next_observations.append(
                np.concatenate(
                    [episode["next_observations"][i][key] for key in ["image1", "image2", "robot_state"]], axis=-1
                )
            )
            next_timesteps.append(episode["next_observations"][i]["timestep"])
            if (
                np.linalg.norm(
                    episode["observations"][i + 1]["robot_state"] - episode["next_observations"][i]["robot_state"]
                )
                > 1e-6
                or episode["terminals"][i] == 1.0
            ):
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        if reward_type == "sparse":
            rewards = episode["rewards"]
        elif reward_type == "step":
            rewards = episode["step_rewards"] / np.max(episode["step_rewards"])
        elif reward_type == "viper":
            rewards = episode["viper_rewards"]
        elif reward_type == "diffusion":
            rewards = episode["diffusion_rewards"]
        else:
            our_reward = episode["multimodal_rewards"]
            next_our_reward = np.asarray(our_reward[1:].tolist() + our_reward[-1:].tolist())
            delta_our_reward = discount * next_our_reward - our_reward
            delta_our_reward[-1] = 0.0
            rewards = delta_our_reward + episode["rewards"]

    return dict(
        observations=np.asarray(observations),
        timesteps=np.asarray(timesteps),
        actions=episode["actions"],
        rewards=rewards,
        masks=1.0 - episode["terminals"],
        dones_float=dones_float,
        next_observations=np.asarray(next_observations),
        next_timesteps=np.asarray(next_timesteps),
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
            _, _, eps_len = fn.stem.split("_")
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def add_episode(self, episode):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        tp = "success" if eps_len < self._max_env_steps else "failure"
        self._num_episodes += 1
        self._num_transitions += eps_len
        eps_fn = f"{tp}_{eps_idx}_{eps_len}.npz"
        save_episode(episode, self._replay_dir / eps_fn)


class ReplayBuffer(IterableDataset):
    def __init__(
        self,
        replay_dir,
        max_size,
        num_workers,
        nstep,
        discount,
        fetch_every,
        save_snapshot=False,
        reward_type: str = "sparse",
        embedding_dim: int = 1024,
    ):
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

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(eps_fn, reward_type=self._reward_type, discount=self._discount)
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
            eps_tp, eps_idx, eps_len = [int(x) for x in eps_fn.stem.split("_")]
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
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
        self.__sample()

    def __sample(self):
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1)
        obs = episode["observations"][episode["timesteps"][idx]]
        action = episode["actions"][idx]
        next_obs = episode["next_observations"][episode["next_timesteps"][idx + self._nstep - 1]]
        reward = np.zeros_like(episode["rewards"][idx])
        discount = np.ones_like(episode["masks"][idx])
        for i in range(self._nstep):
            step_reward = episode["rewards"][idx + i]
            reward += discount * step_reward
            discount *= episode["masks"][idx + i] * self._discount

        image1, image2, robot_state = np.split(obs, [self._embedding_dim, self._embedding_dim * 2], axis=-1)
        next_image1, next_image2, next_robot_state = np.split(
            next_obs, [self._embedding_dim, self._embedding_dim * 2], axis=-1
        )

        return Batch(
            observations=dict(image1=image1, image2=image2, robot_state=robot_state),
            actions=action,
            rewards=reward,
            masks=discount,
            next_observations=dict(image1=next_image1, image2=next_image2, robot_state=next_robot_state),
        )

    def __iter__(self):
        while True:
            yield self._sample()


class OfflineReplayBuffer(IterableDataset):
    def __init__(
        self,
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
    ):
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
        self._try_fetch()

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(eps_fn, reward_type=self._reward_type, discount=self._discount)
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

    def _sample(self):
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._nstep)
        obs = episode["observations"][episode["timesteps"][idx]]
        action = episode["actions"][idx]
        next_obs = episode["observations"][episode["next_timesteps"][idx + self._nstep]]
        reward = np.zeros_like(episode["rewards"][idx])
        discount = np.ones_like(episode["masks"][idx])
        for i in range(self._nstep):
            step_reward = episode["rewards"][idx + i]
            reward += discount * step_reward
            discount *= episode["masks"][idx + i] * self._discount

        image1, image2, robot_state = np.split(obs, [self._embedding_dim, self._embedding_dim * 2], axis=-1)
        next_image1, next_image2, next_robot_state = np.split(
            next_obs, [self._embedding_dim, self._embedding_dim * 2], axis=-1
        )

        return Batch(
            observations=dict(image1=image1, image2=image2, robot_state=robot_state),
            actions=action,
            rewards=reward,
            masks=discount,
            next_observations=dict(image1=next_image1, image2=next_image2, robot_state=next_robot_state),
        )

    def __iter__(self):
        while True:
            yield self._sample()


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_replay_loader(
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
):
    max_size_per_worker = max_size // max(1, num_workers)

    if buffer_type == "online":
        iterable = ReplayBuffer(
            replay_dir,
            max_size_per_worker,
            num_workers,
            nstep,
            discount,
            fetch_every=1000,
            save_snapshot=save_snapshot,
            reward_type=reward_type,
        )
    elif buffer_type == "offline":
        iterable = OfflineReplayBuffer(
            replay_dir,
            max_size_per_worker,
            num_workers,
            nstep,
            discount,
            fetch_every=1e6,
            save_snapshot=save_snapshot,
            reward_type=reward_type,
            num_demos=num_demos,
        )

    loader = torch.utils.data.DataLoader(
        iterable, batch_size=batch_size, num_workers=num_workers, pin_memory=True, worker_init_fn=_worker_init_fn
    )
    return loader
