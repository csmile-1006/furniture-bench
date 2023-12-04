import collections
import pickle
from typing import Optional, Dict

# import d4rl
import gym
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm

Batch = collections.namedtuple("Batch", ["observations", "actions", "rewards", "masks", "next_observations"])


def split_into_trajectories(observations, actions, rewards, masks, dones_float, next_observations):
    trajs = [[]]

    for i in tqdm(range(len(observations))):
        trajs[-1].append(
            (
                observations[i],
                actions[i],
                rewards[i],
                masks[i],
                dones_float[i],
                next_observations[i],
            )
        )
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs


def merge_trajectories(trajs):
    observations = []
    actions = []
    rewards = []
    masks = []
    dones_float = []
    next_observations = []

    for traj in trajs:
        for obs, act, rew, mask, done, next_obs in traj:
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            masks.append(mask)
            dones_float.append(done)
            next_observations.append(next_obs)

    return (
        np.stack(observations),
        np.stack(actions),
        np.stack(rewards),
        np.stack(masks),
        np.stack(dones_float),
        np.stack(next_observations),
    )


class Dataset(object):
    def __init__(
        self,
        observations: Dict[str, np.ndarray],
        actions: np.ndarray,
        rewards: np.ndarray,
        masks: np.ndarray,
        dones_float: np.ndarray,
        next_observations: np.ndarray,
        size: int,
        use_encoder: bool = False,
    ):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = size
        self.use_encoder = use_encoder

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)

        obs_img1 = jnp.zeros([batch_size, 224, 224, 3], dtype=jnp.float32)
        obs_img2 = jnp.zeros([batch_size, 224, 224, 3], dtype=jnp.float32)
        next_obs_img1 = jnp.zeros([batch_size, 224, 224, 3], dtype=jnp.float32)
        next_obs_img2 = jnp.zeros([batch_size, 224, 224, 3], dtype=jnp.float32)

        if self.use_encoder:
            # Preprocess the image.
            for i in indx:
                obs_img1 = obs_img1.at[i].set(jnp.array(self.observations[i]["image1"] / 255.0))
                obs_img1 = obs_img1.at[i, :, :, 0].add(-0.485)
                obs_img1 = obs_img1.at[i, :, :, 1].add(-0.456)
                obs_img1 = obs_img1.at[i, :, :, 2].add(-0.406)
                obs_img1 = obs_img1.at[i, :, :, 0].divide(0.229)
                obs_img1 = obs_img1.at[i, :, :, 1].divide(0.224)
                obs_img1 = obs_img1.at[i, :, :, 2].divide(0.225)

                obs_img2 = obs_img2.at[i].set(jnp.array(self.observations[i]["image2"] / 255.0))
                obs_img2 = obs_img2.at[i, :, :, 0].add(-0.485)
                obs_img2 = obs_img2.at[i, :, :, 1].add(-0.456)
                obs_img2 = obs_img2.at[i, :, :, 2].add(-0.406)
                obs_img2 = obs_img2.at[i, :, :, 0].divide(0.229)
                obs_img2 = obs_img2.at[i, :, :, 1].divide(0.224)
                obs_img2 = obs_img2.at[i, :, :, 2].divide(0.225)

                next_obs_img1 = next_obs_img1.at[i].set(jnp.array(self.next_observations[i]["image1"] / 255.0))
                next_obs_img1 = next_obs_img1.at[i, :, :, 0].add(-0.485)
                next_obs_img1 = next_obs_img1.at[i, :, :, 1].add(-0.456)
                next_obs_img1 = next_obs_img1.at[i, :, :, 2].add(-0.406)
                next_obs_img1 = next_obs_img1.at[i, :, :, 0].divide(0.229)
                next_obs_img1 = next_obs_img1.at[i, :, :, 1].divide(0.224)
                next_obs_img1 = next_obs_img1.at[i, :, :, 2].divide(0.225)

                next_obs_img2 = next_obs_img2.at[i].set(jnp.array(self.next_observations[i]["image2"] / 255.0))
                next_obs_img2 = next_obs_img2.at[i, :, :, 0].add(-0.485)
                next_obs_img2 = next_obs_img2.at[i, :, :, 1].add(-0.456)
                next_obs_img2 = next_obs_img2.at[i, :, :, 2].add(-0.406)
                next_obs_img2 = next_obs_img2.at[i, :, :, 0].divide(0.229)
                next_obs_img2 = next_obs_img2.at[i, :, :, 1].divide(0.224)
                next_obs_img2 = next_obs_img2.at[i, :, :, 2].divide(0.225)

                # self.next_observations[i][img] = self.next_observations[i][img] / 255.0
                # self.next_observations[i][img][:, :, 0] = (self.next_observations[i][img][:, :, 0] - 0.485) / 0.229
                # self.next_observations[i][img][:, :, 1] = (self.next_observations[i][img][:, :, 1] - 0.456) / 0.224
                # self.next_observations[i][img][:, :, 2] = (self.next_observations[i][img][:, :, 2] - 0.406) / 0.225
        if self.use_encoder:
            return Batch(
                observations={
                    "image1": obs_img1,
                    "image2": obs_img2,
                    "robot_state": jnp.array([self.observations[i]["robot_state"] for i in indx], dtype=jnp.float32),
                },
                actions=self.actions[indx],
                rewards=self.rewards[indx],
                masks=self.masks[indx],
                next_observations={
                    "image1": next_obs_img1,
                    "image2": next_obs_img2,
                    "robot_state": jnp.array(
                        [self.next_observations[i]["robot_state"] for i in indx], dtype=jnp.float32
                    ),
                },
            )
        return Batch(
            observations={
                "image1": jnp.array([self.observations[i][..., : self.embedding_dim] for i in indx], dtype=jnp.float32),
                "image2": jnp.array(
                    [self.observations[i][..., self.embedding_dim : self.embedding_dim * 2] for i in indx],
                    dtype=jnp.float32,
                ),
                "robot_state": jnp.array(
                    [self.observations[i][..., self.embedding_dim * 2 :] for i in indx], dtype=jnp.float32
                ),
            },
            actions=self.actions[indx],
            rewards=self.rewards[indx],
            masks=self.masks[indx],
            next_observations={
                "image1": jnp.array(
                    [self.next_observations[i][..., : self.embedding_dim] for i in indx], dtype=jnp.float32
                ),
                "image2": jnp.array(
                    [self.next_observations[i][..., self.embedding_dim : self.embedding_dim * 2] for i in indx],
                    dtype=jnp.float32,
                ),
                "robot_state": jnp.array(
                    [self.next_observations[i][..., self.embedding_dim * 2 :] for i in indx], dtype=jnp.float32
                ),
            },
        )


class D4RLDataset(Dataset):
    def __init__(self, env: gym.Env, clip_to_eps: bool = True, eps: float = 1e-5):
        dataset = d4rl.qlearning_dataset(env)

        if clip_to_eps:
            lim = 1 - eps
            dataset["actions"] = np.clip(dataset["actions"], -lim, lim)

        dones_float = np.zeros_like(dataset["rewards"])

        for i in range(len(dones_float) - 1):
            if (
                np.linalg.norm(dataset["observations"][i + 1] - dataset["next_observations"][i]) > 1e-6
                or dataset["terminals"][i] == 1.0
            ):
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        super().__init__(
            dataset["observations"].astype(np.float32),
            actions=dataset["actions"].astype(np.float32),
            rewards=dataset["rewards"].astype(np.float32),
            masks=1.0 - dataset["terminals"].astype(np.float32),
            dones_float=dones_float.astype(np.float32),
            next_observations=dataset["next_observations"].astype(np.float32),
            size=len(dataset["observations"]),
        )


class FurnitureDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        clip_to_eps: bool = True,
        eps: float = 1e-5,
        use_encoder: bool = False,
        use_arp: bool = False,
        use_step: bool = False,
        lambda_mr: float = 1e-1,
    ):
        with open(data_path, "rb") as f:
            dataset = pickle.load(f)

        # if clip_to_eps:
        #     lim = 1 - eps
        #     dataset["actions"] = np.clip(dataset["actions"], -lim, lim)

        dones_float = np.zeros_like(dataset["rewards"], dtype=np.float32)

        # if use_encoder:
        #     # Preprocess the image.
        #     for i in range(len(dataset['observations'])):
        #         # dataset['observations'][i]['image_feature'] = dataset['observations'][i]['image_feature'] / 255.0
        #         # dataset['next_observations'][i]['image_feature'] = dataset['next_observations'][i]['image_feature'] / 255.0
        #         for img in ['image1', 'image2']:
        #             for obs in ['observations', 'next_observations']:
        #                 dataset[obs][i][img] = dataset[obs][i][img] / 255.0
        #                 dataset[obs][i][img][:, :, 0] = (dataset[obs][i][img][:, :, 0] - 0.485) / 0.229
        #                 dataset[obs][i][img][:, :, 1] = (dataset[obs][i][img][:, :, 1] - 0.456) / 0.224
        #                 dataset[obs][i][img][:, :, 2] = (dataset[obs][i][img][:, :, 2] - 0.406) / 0.225

        for i in range(len(dones_float) - 1):
            if (
                np.linalg.norm(
                    dataset["observations"][i + 1]["robot_state"] - dataset["next_observations"][i]["robot_state"]
                )
                > 1e-6
                or dataset["terminals"][i] == 1.0
            ):
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        self.embedding_dim = dataset["observations"][0]["image1"].shape[-1]
        observations, next_observations = [], []
        for i in range(len(dataset["observations"])):
            observations.append(
                np.concatenate(
                    [dataset["observations"][i][key] for key in ["image1", "image2", "robot_state"]], axis=-1
                )
            )
            next_observations.append(
                np.concatenate(
                    [dataset["next_observations"][i][key] for key in ["image1", "image2", "robot_state"]], axis=-1
                )
            )

        dones_float[-1] = 1
        if use_arp:
            # rewards = lambda_mr * dataset["multimodal_rewards"] + dataset["rewards"]
            rewards = lambda_mr * dataset["multimodal_rewards"]
        elif use_step:
            rewards = dataset["step_rewards"] / np.max(dataset["step_rewards"])
        else:
            rewards = dataset["rewards"]

        super().__init__(
            np.asarray(observations),
            actions=dataset["actions"],
            rewards=rewards,
            masks=1.0 - dataset["terminals"],
            dones_float=dones_float,
            next_observations=np.asarray(next_observations),
            size=len(dataset["observations"]),
            use_encoder=use_encoder,
        )


class FurnitureSequenceDataset(FurnitureDataset):
    def __init__(
        self,
        data_path: str,
        clip_to_eps: bool = True,
        eps: float = 1e-5,
    ):
        super().__init__(data_path=data_path, clip_to_eps=clip_to_eps, eps=eps, use_encoder=False)

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)

        # obs_img1 = jnp.zeros([batch_size, self._window_size, 224, 224, 3], dtype=jnp.float32)
        # obs_img2 = jnp.zeros([batch_size, self._window_size, 224, 224, 3], dtype=jnp.float32)
        # next_obs_img1 = jnp.zeros([batch_size, self._window_size, 224, 224, 3], dtype=jnp.float32)
        # next_obs_img2 = jnp.zeros([batch_size, self._window_size, 224, 224, 3], dtype=jnp.float32)

        return Batch(
            observations={
                "image1": jnp.array([self.observations[i]["image1"] for i in indx], dtype=jnp.float32),
                "image2": jnp.array([self.observations[i]["image2"] for i in indx], dtype=jnp.float32),
                "robot_state": jnp.array([self.observations[i]["robot_state"] for i in indx], dtype=jnp.float32),
            },
            actions=self.actions[indx],
            rewards=self.rewards[indx],
            masks=self.masks[indx],
            next_observations={
                "image1": jnp.array([self.next_observations[i]["image1"] for i in indx], dtype=jnp.float32),
                "image2": jnp.array([self.next_observations[i]["image2"] for i in indx], dtype=jnp.float32),
                "robot_state": jnp.array([self.next_observations[i]["robot_state"] for i in indx], dtype=jnp.float32),
            },
        )


class ReplayBuffer(Dataset):
    def __init__(self, observation_space: gym.spaces.Box, action_dim: int, capacity: int):
        obs_shape = (
            observation_space["image1"].shape[0],
            sum([observation_space[key].shape[-1] for key in observation_space]),
        )
        observations = np.empty((capacity, *obs_shape), dtype=observation_space.dtype)
        actions = np.empty((capacity, action_dim), dtype=np.float32)
        rewards = np.empty((capacity,), dtype=np.float32)
        masks = np.empty((capacity,), dtype=np.float32)
        dones_float = np.empty((capacity,), dtype=np.float32)
        next_observations = np.empty((capacity, *obs_shape), dtype=observation_space.dtype)
        super().__init__(
            observations=observations,
            actions=actions,
            rewards=rewards,
            masks=masks,
            dones_float=dones_float,
            next_observations=next_observations,
            size=0,
        )

        self.size = 0

        self.insert_index = 0
        self.capacity = capacity

    def initialize_with_dataset(self, dataset: Dataset, num_samples: Optional[int]):
        assert self.insert_index == 0, "Can insert a batch online in an empty replay buffer."

        self.embedding_dim = dataset.embedding_dim
        dataset_size = len(dataset.observations)

        if num_samples is None:
            num_samples = dataset_size
        else:
            num_samples = min(dataset_size, num_samples)
        assert self.capacity >= num_samples, "Dataset cannot be larger than the replay buffer capacity."

        if num_samples < dataset_size:
            perm = np.random.permutation(dataset_size)
            indices = perm[:num_samples]
        else:
            indices = np.arange(num_samples)

        self.observations[:num_samples] = dataset.observations[indices]
        self.actions[:num_samples] = dataset.actions[indices]
        self.rewards[:num_samples] = dataset.rewards[indices]
        self.masks[:num_samples] = dataset.masks[indices]
        self.dones_float[:num_samples] = dataset.dones_float[indices]
        self.next_observations[:num_samples] = dataset.next_observations[indices]

        self.insert_index = num_samples
        self.size = num_samples

    def insert(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        mask: float,
        done_float: float,
        next_observation: np.ndarray,
    ):
        observation = np.concatenate([observation[key] for key in ["image1", "image2", "robot_state"]], axis=-1)
        next_observation = np.concatenate(
            [next_observation[key] for key in ["image1", "image2", "robot_state"]], axis=-1
        )
        insert_size = observation.shape[0]
        self.observations[self.insert_index : self.insert_index + insert_size] = observation
        self.actions[self.insert_index : self.insert_index + insert_size] = action
        self.rewards[self.insert_index : self.insert_index + insert_size] = reward
        self.masks[self.insert_index : self.insert_index + insert_size] = mask
        self.dones_float[self.insert_index : self.insert_index + insert_size] = done_float
        self.next_observations[self.insert_index : self.insert_index + insert_size] = next_observation

        self.insert_index = (self.insert_index + insert_size) % self.capacity
        self.size = min(self.size + insert_size, self.capacity)
