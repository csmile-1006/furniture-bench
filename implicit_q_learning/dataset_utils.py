import collections
import pickle
from typing import Optional, Dict

# import d4rl
import gym
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
import scipy

Batch = collections.namedtuple("Batch", ["observations", "actions", "rewards", "masks", "next_observations"])


def gaussian_smoothe(rewards, sigma=3.0):
    return scipy.ndimage.gaussian_filter1d(rewards, sigma=sigma, mode="nearest")


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
        timesteps: np.ndarray,
        next_timesteps: np.ndarray,
        use_encoder: bool = False,
    ):
        self.observations = observations
        self.timesteps = timesteps
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.next_timesteps = next_timesteps
        self.size = size
        self.use_encoder = use_encoder

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        observations = self.observations[self.timesteps[indx]]
        image1, image2, robot_state = np.split(observations, (self.embedding_dim, self.embedding_dim * 2), axis=-1)
        next_observations = self.next_observations[self.next_timesteps[indx]]
        next_image1, next_image2, next_robot_state = np.split(
            next_observations, (self.embedding_dim, self.embedding_dim * 2), axis=-1
        )

        return Batch(
            observations=dict(image1=image1, image2=image2, robot_state=robot_state),
            actions=self.actions[indx],
            rewards=self.rewards[indx],
            masks=self.masks[indx],
            next_observations=dict(image1=next_image1, image2=next_image2, robot_state=next_robot_state),
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
        use_viper: bool = False,
        use_diffusion_reward: bool = False,
        lambda_mr: float = 1e-1,
    ):
        with open(data_path, "rb") as f:
            dataset = pickle.load(f)

        dones_float = np.zeros_like(dataset["rewards"], dtype=np.float32)

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
        observations, next_observations, timesteps, next_timesteps = [], [], [], []
        for i in range(len(dataset["observations"])):
            observations.append(
                np.concatenate(
                    [dataset["observations"][i][key] for key in ["image1", "image2", "robot_state"]], axis=-1
                )
            )
            timesteps.append(dataset["observations"][i]["timestep"])
            next_observations.append(
                np.concatenate(
                    [dataset["next_observations"][i][key] for key in ["image1", "image2", "robot_state"]], axis=-1
                )
            )
            next_timesteps.append(dataset["next_observations"][i]["timestep"])

        dones_float[-1] = 1
        if use_arp:
            rewards = lambda_mr * dataset["multimodal_rewards"]
        elif use_step:
            rewards = lambda_mr * dataset["step_rewards"] / np.max(dataset["step_rewards"])
        elif use_viper:
            rewards = lambda_mr * dataset["viper_reward"]
        elif use_diffusion_reward:
            rewards = lambda_mr * dataset["diffusion_reward"]
        else:
            rewards = dataset["rewards"]

        super().__init__(
            observations=np.asarray(observations),
            timesteps=np.asarray(timesteps),
            actions=dataset["actions"],
            rewards=rewards,
            masks=1.0 - dataset["terminals"],
            dones_float=dones_float,
            next_observations=np.asarray(next_observations),
            next_timesteps=np.asarray(next_timesteps),
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
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_dim: int,
        capacity: int,
        window_size: int = 4,
        embedding_dim: int = 1024,
    ):
        obs_shape = (sum([observation_space[key].shape[-1] for key in observation_space]),)
        observations = np.zeros((capacity, *obs_shape), dtype=observation_space.dtype)
        actions = np.zeros((capacity, action_dim), dtype=np.float32)
        rewards = np.zeros((capacity,), dtype=np.float32)
        masks = np.zeros((capacity,), dtype=np.float32)
        dones_float = np.zeros((capacity,), dtype=np.float32)
        next_observations = np.zeros((capacity, *obs_shape), dtype=observation_space.dtype)
        timesteps = np.zeros((capacity, window_size), dtype=np.int32)
        next_timesteps = np.zeros((capacity, window_size), dtype=np.int32)
        super().__init__(
            observations=observations,
            timesteps=timesteps,
            actions=actions,
            rewards=rewards,
            masks=masks,
            dones_float=dones_float,
            next_observations=next_observations,
            next_timesteps=next_timesteps,
            size=0,
        )

        self.size = 0

        self.insert_index = 0
        self.capacity = capacity
        self.embedding_dim = embedding_dim

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
        self.timesteps[:num_samples] = dataset.timesteps[indices]
        self.actions[:num_samples] = dataset.actions[indices]
        self.rewards[:num_samples] = dataset.rewards[indices]
        self.masks[:num_samples] = dataset.masks[indices]
        self.dones_float[:num_samples] = dataset.dones_float[indices]
        self.next_observations[:num_samples] = dataset.next_observations[indices]
        self.next_timesteps[:num_samples] = dataset.next_timesteps[indices]

        self.insert_index = num_samples
        self.size = num_samples

    def insert(
        self,
        observation: np.ndarray,
        timestep: np.ndarray,
        action: np.ndarray,
        reward: float,
        mask: float,
        done_float: float,
        next_observation: np.ndarray,
        next_timestep: np.ndarray,
    ):
        observation = np.stack(
            [np.concatenate([v[key] for key in ["image1", "image2", "robot_state"]], axis=-1) for v in observation]
        )
        next_observation = np.stack(
            [np.concatenate([v[key] for key in ["image1", "image2", "robot_state"]], axis=-1) for v in next_observation]
        )

        insert_size = min(observation.shape[0], self.capacity - self.insert_index)
        self.observations[self.insert_index : self.insert_index + insert_size] = observation[:insert_size]
        self.timesteps[self.insert_index : self.insert_index + insert_size] = timestep[:insert_size]
        self.actions[self.insert_index : self.insert_index + insert_size] = action[:insert_size]
        self.rewards[self.insert_index : self.insert_index + insert_size] = reward.squeeze()[:insert_size]
        self.masks[self.insert_index : self.insert_index + insert_size] = mask[:insert_size]
        self.dones_float[self.insert_index : self.insert_index + insert_size] = done_float.squeeze()[:insert_size]
        self.next_observations[self.insert_index : self.insert_index + insert_size] = next_observation[:insert_size]
        self.next_timesteps[self.insert_index : self.insert_index + insert_size] = next_timestep[:insert_size]

        self.insert_index = (self.insert_index + insert_size) % self.capacity
        self.size = min(self.size + insert_size, self.capacity)

    def insert_episode(self, trajectories: dict, window_size=4, skip_frame=16):
        trajectories = {key: np.asarray(val) for key, val in trajectories.items()}
        len_episode = trajectories["actions"].shape[0]
        stacked_timesteps = []
        timestep_stacks = {key: collections.deque([], maxlen=window_size) for key in range(skip_frame)}

        for _ in range(window_size):
            for j in range(skip_frame):
                timestep_stacks[j].append(0)

        for i in range(len_episode):
            mod = i % skip_frame
            timestep_stack = timestep_stacks[mod]
            timestep_stack.append(i)
            stacked_timesteps.append(np.stack(timestep_stack))

        next_stacked_timesteps = stacked_timesteps[1:].copy()
        next_stacked_timesteps.append(stacked_timesteps[-1])
        timesteps = np.asarray(stacked_timesteps) + self.size
        next_timesteps = np.asarray(next_stacked_timesteps) + self.size

        # smooth reward for stabilizing training.
        rewards = gaussian_smoothe(trajectories["rewards"])

        return self.insert(
            observation=trajectories["observations"],
            timestep=timesteps,
            action=trajectories["actions"],
            reward=rewards,
            mask=trajectories["masks"],
            done_float=trajectories["done_floats"],
            next_observation=trajectories["next_observations"],
            next_timestep=next_timesteps,
        )
