import isaacgym
import os
import pickle
from datetime import datetime
from typing import Tuple
import glob
import copy
import random

import gym
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter
from matplotlib.ticker import ScalarFormatter

import jax.numpy as jnp
from einops import rearrange
import wrappers
from dataset_utils import (
    Batch,
    D4RLDataset,
    ReplayBuffer,
    split_into_trajectories,
    Dataset,
    FurnitureDataset,
    max_normalize,
    replay_chunk_to_seq,
    min_max_normalize,
)
from evaluation import evaluate
from learner import Learner

from furniture_bench.data.collect_enum import CollectEnum


import matplotlib.pyplot as plt
import imageio

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "halfcheetah-expert-v2", "Environment name.")
flags.DEFINE_string("save_dir", "./checkpoints/", "Tensorboard logging dir.")
flags.DEFINE_string("run_name", "debug", "Run specific name")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 10, "Logging interval.")
# flags.DEFINE_integer('eval_interval', 100000, 'Eval interval.')
flags.DEFINE_integer("eval_interval", 1000000, "Eval interval.")
flags.DEFINE_integer("save_interval", 1, "Save interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_string("ckpt_step", None, "Specific checkpoint step")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_integer("max_episodes", int(5e3), "Number of episodes for training")
flags.DEFINE_integer("replay_buffer_size", 2000000, "Replay buffer size (=max_steps if unspecified).")
flags.DEFINE_integer("init_dataset_size", None, "Offline data size (uses all data if unspecified).")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("red_reward", False, "Use learned reward")
flags.DEFINE_boolean("viper_reward", False, "Use learned reward")
flags.DEFINE_boolean("drs_reward", False, "Use learned reward")
flags.DEFINE_enum("reward_type", "REDS", ["REDS", "DrS", "VIPER"], "Type of reward model.")
flags.DEFINE_boolean("use_encoder", False, "Use ResNet18 for the image encoder.")
flags.DEFINE_string("encoder_type", "", "vip or r3m")
flags.DEFINE_boolean("wandb", False, "Use wandb")
flags.DEFINE_string("wandb_project", "furniture-reward", "wandb project")
flags.DEFINE_string("wandb_entity", "clvr", "wandb entity")
config_flags.DEFINE_config_file(
    "config",
    "configs/antmaze_finetune_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

flags.DEFINE_multi_string("data_path", "", "Path to data.")
flags.DEFINE_string("normalization", "", "")
flags.DEFINE_string("iter_n", "-1", "Reward relabeling iteration")
flags.DEFINE_boolean("use_learned_reward", False, "Use learned reward")
flags.DEFINE_string("reward_suffix", "-1", "Reward suffix")
# flags.DEFINE_string("reward_type", "REDS", "Reward type")


# REDS
flags.DEFINE_string("task_name", "furniture_one_leg", "Name of task name.")
flags.DEFINE_string("ckpt_path", "", "ckpt path of reward model.")
flags.DEFINE_string("image_keys", "color_image2|color_image1", "image keys.")
flags.DEFINE_string("rm_type", "RFE", "reward model type.")
flags.DEFINE_integer("window_size", 4, "window size")
flags.DEFINE_integer("skip_frame", 1, "skip frame")

flags.DEFINE_boolean("phase_reward", None, "Use phase reward (for logging or training)")

flags.DEFINE_boolean("keyboard", None, "Use phase reward (for logging or training)")
flags.DEFINE_boolean("save_data", None, "Save the training data.")
flags.DEFINE_boolean("eval", None, "Evaluation.")
flags.DEFINE_boolean("use_layer_norm", None, "Use layer normalization.")
flags.DEFINE_boolean("online_buffer", None, "Use separate online buffer.")
flags.DEFINE_boolean("fixed_init", None, "Use separate online buffer.")
flags.DEFINE_boolean("data_collection", None, "Skip the agent update.")
flags.DEFINE_float("temperature", 0.2, "Action sample temperature.")
flags.DEFINE_boolean("load_finetune_ckpt", None, "Load the fine-tune checkpoint.")
flags.DEFINE_boolean("from_scratch", False, "Train from scratch.")
flags.DEFINE_integer("utd_ratio", 1, "Update to data ratio.")
flags.DEFINE_integer("prefill_episodes", 0, "Pre-fill episodes.")

# DEVICE
flags.DEFINE_integer("device_id", -1, "Device ID for using multiple GPU")
flags.DEFINE_boolean("save_gif", None, "Save reward and observation in gif")
flags.DEFINE_boolean("policy_ddpg_bc", None, "Use DDPG-BC for policy extraction")


def normalize(dataset):
    trajs = split_into_trajectories(
        dataset.observations,
        dataset.actions,
        dataset.rewards,
        dataset.masks,
        dataset.dones_float,
        dataset.next_observations,
    )

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)

    dataset.rewards /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset.rewards *= 1000.0


def gif_maker(gif_dir, observations, rewards, phases, file_index):
    filenames = []

    if not os.path.exists("gif_test"):
        os.makedirs("gif_test")

    for i in tqdm.tqdm(range(len(observations))):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        image = observations[i]["color_image2"].astype(np.uint8)

        ax[0].imshow(image)
        ax[0].axis("off")

        ax2 = ax[1].twinx()
        ax[1].plot(rewards[: i + 1])
        ax2.plot(phases[: i + 1], color="pink", linestyle="dashed")
        ax[1].set_xlim(0, i)
        ax[1].set_ylim(np.min(rewards[: i + 1]), np.max(rewards[: i + 1]) + 0.05 * np.abs(np.max(rewards[: i + 1])))

        filename = f"{gif_dir}/frame_{i:03d}.png"
        filenames.append(filename)
        plt.savefig(filename)
        plt.close(fig)

    with imageio.get_writer(f"{gif_dir}/{file_index}.gif", mode="I", duration=5) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    for filename in filenames:
        os.remove(filename)


def combine(one_dict, other_dict):
    combined = {}
    if isinstance(one_dict, Batch):
        one_dict, other_dict = one_dict._asdict(), other_dict._asdict()
    for k, v in one_dict.items():
        if isinstance(v, dict):
            combined[k] = combine(v, other_dict[k])
        else:
            # Use half.
            tmp = np.empty((v.shape[0] // 2 + other_dict[k].shape[0] // 2, *v.shape[1:]), dtype=v.dtype)
            tmp[0::2] = v[: len(v) // 2]
            tmp[1::2] = other_dict[k][: len(other_dict[k]) // 2]
            combined[k] = tmp
    return combined


def make_env_and_dataset(
    env_name: str,
    seed: int,
    data_path: str,
    use_encoder: bool,
    encoder_type: str,
    red_reward: bool = False,
    viper_reward: bool = False,
    drs_reward: bool = False,
    iter_n: int = -1,
) -> Tuple[gym.Env, D4RLDataset]:
    if "Furniture" in env_name:
        import furniture_bench

        env_id, furniture_name = env_name.split("/")
        # env = gym.make(env_id,
        #                furniture=furniture_name,
        #                data_path=data_path,
        #                use_encoder=use_encoder,
        #    encoder_type=encoder_type)
        env = gym.make(
            env_id,
            furniture=furniture_name,
            # max_env_steps=600,
            headless=True,
            # headless=False,
            num_envs=1,  # Only support 1 for now.
            manual_done=False,
            # resize_img=True,
            # np_step_out=False,  # Always output Tensor in this setting. Will change to numpy in this code.
            # channel_first=False,
            randomness="low",
            compute_device_id=FLAGS.device_id,
            graphics_device_id=FLAGS.device_id,
            # gripper_pos_control=True,
            encoder_type="r3m",
            squeeze_done_reward=True,
            phase_reward=FLAGS.phase_reward,
            fixed_init=FLAGS.fixed_init,
            from_skill=0,
            gpu=FLAGS.device_id
        )
    else:
        env = gym.make(env_name)

    env = wrappers.SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    print("Observation space", env.observation_space)
    print("Action space", env.action_space)

    if "Furniture" in env_name:
        # if FLAGS.iter_n.isdigit():
        #     iter_n = f"iter_{FLAGS.iter_n}"
        # else:
        #     iter_n = FLAGS.iter_n
        if FLAGS.use_learned_reward:
            reward_name = f"{reward_suffix}"
        else:
            iter_n = FLAGS.iter_n
        dataset = FurnitureDataset(
            data_path,
            use_encoder=use_encoder,
            red_reward=red_reward,
            viper_reward=viper_reward,
            drs_reward=drs_reward,
            iter_n=iter_n,
        )
    else:
        dataset = D4RLDataset(env)

    if "antmaze" in env_name:
        dataset.rewards -= 1.0
        # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
        # but I found no difference between (x - 0.5) * 4 and x - 1.0
    elif "halfcheetah" in env_name or "walker2d" in env_name or "hopper" in env_name:
        normalize(dataset)

    return env, dataset


def main(_):
    import jax

    jax.config.update("jax_default_device", jax.devices()[FLAGS.device_id])

    ckpt_step = FLAGS.ckpt_step or FLAGS.max_steps
    root_logdir = os.path.join(
        FLAGS.save_dir,
        "tb",
        f"{FLAGS.run_name}-{ckpt_step}-finetune-tmp-{FLAGS.temperature}-bs{FLAGS.batch_size}.{FLAGS.seed}",
    )
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    env, dataset = make_env_and_dataset(
        FLAGS.env_name,
        FLAGS.seed,
        FLAGS.data_path,
        FLAGS.use_encoder,
        FLAGS.encoder_type,
        FLAGS.red_reward,
        FLAGS.viper_reward,
        FLAGS.drs_reward,
        FLAGS.iter_n,
    )
    offline_traj_for_log = []
    with open(FLAGS.data_path[0], 'rb') as f:
        offline_data = pickle.load(f)
        terminal_idxs =  np.where(offline_data['terminals'] == 1)[0]
        offline_traj_for_log.append((offline_data['observations'][:terminal_idxs[0]], offline_data['actions'][:terminal_idxs[0]]))
        for i in range(len(terminal_idxs) - 1):
            offline_traj_for_log.append((offline_data['observations'][terminal_idxs[i]:terminal_idxs[i+1]], offline_data['actions'][terminal_idxs[i]:terminal_idxs[i+1]]))

    if FLAGS.normalization == "min_max":
        min_max_normalize(dataset)
    if FLAGS.normalization == "max":
        max_rew = np.max(dataset.rewards)
        max_rew = np.abs(max_rew)  # For DrS negative reward.
        dataset.rewards = max_normalize(dataset.rewards, max_rew)

    kwargs = dict(FLAGS.config)
    print(f"kwargs: {kwargs}")
    if FLAGS.wandb:
        import wandb

        wandb.tensorboard.patch(root_logdir=root_logdir)
        wandb.init(
            project=FLAGS.wandb_project,
            entity=FLAGS.wandb_entity,
            name=FLAGS.env_name
            + "-"
            + str(FLAGS.seed)
            + "-"
            + str(FLAGS.run_name)
            + "-finetune"
            + f"-actnoise{FLAGS.temperature}"
            + f"-utd{FLAGS.utd_ratio}",
            # + ("-phase-reward" if FLAGS.phase_reward else ""),
            config=kwargs,
            sync_tensorboard=True,
        )

    if FLAGS.keyboard:
        from furniture_bench.device import make_device

        device_interface = make_device("keyboard")
    else:
        device_interface = None
    summary_writer = SummaryWriter(root_logdir, write_to_disk=True)

    # agent = Learner(FLAGS.seed,
    #                 env.observation_space.sample()[np.newaxis],
    #                 env.action_space.sample()[np.newaxis], **kwargs)
    agent = Learner(
        FLAGS.seed,
        env.observation_space.sample(),
        env.action_space.sample()[np.newaxis],
        max_steps=FLAGS.max_episodes,
        **kwargs,
        use_encoder=FLAGS.use_encoder,
        opt_decay_schedule=None,
        use_layer_norm=FLAGS.use_layer_norm,
        policy_ddpg_bc=FLAGS.policy_ddpg_bc,
    )

    if FLAGS.use_learned_reward:
        # load reward model.
        if FLAGS.reward_type == "REDS":
            from bpref_v2.reward_model.rfe_reward_model import RFERewardModel

            reward_model = RFERewardModel(
                task=FLAGS.task_name,
                model_name=FLAGS.rm_type,
                rm_path=FLAGS.ckpt_path,
                camera_keys=FLAGS.image_keys.split("|"),
                reward_scale=None,
                window_size=FLAGS.window_size,
                skip_frame=FLAGS.skip_frame,
                reward_model_device=0,
                encoding_minibatch_size=16,
                use_task_reward=False,
                use_scale=False,
            )
        elif FLAGS.reward_type == 'DRS':
            from bpref_v2.reward_model.drs_reward_model import DrsRewardModel
            
            reward_model = DrsRewardModel(
                task=FLAGS.task_name,
                model_name="DRS",
                rm_path=FLAGS.ckpt_path,
                camera_keys=FLAGS.image_keys.split("|"),
                reward_scale=None,
                window_size=FLAGS.window_size,
                skip_frame=FLAGS.skip_frame,
                reward_model_device=0,
                encoding_minibatch_size=16,
                use_task_reward=False,
                use_scale=False,
            )
        elif FLAGS.reward_type == 'VIPER':
            from viper_rl.videogpt.reward_models.videogpt_reward_model import VideoGPTRewardModel
            
            domain, task = FLAGS.task_name.split("_", 1)
            reward_model = VideoGPTRewardModel(
                task=FLAGS.task_name,
                vqgan_path=os.path.join(FLAGS.ckpt_path, f"{domain}_vqgan"),
                videogpt_path=os.path.join(FLAGS.ckpt_path, f"{domain}_videogpt_l4_s4"),
                camera_key=FLAGS.image_keys.split("|")[0],
                reward_scale=None,
                minibatch_size=4,
                encoding_minibatch_size=4,
            )

    if FLAGS.viper_reward:
        # load reward model

        import sys

        sys.path.append("/home/changyeon/NeurIPS2024/workspace/viper_rl")
        from viper_rl.videogpt.reward_models.videogpt_reward_model import VideoGPTRewardModel

        domain, task = FLAGS.task_name.split("_", 1)
        reward_model = VideoGPTRewardModel(
            task=FLAGS.task_name,
            vqgan_path=os.path.join(FLAGS.ckpt_path, f"{domain}_vqgan"),
            videogpt_path=os.path.join(FLAGS.ckpt_path, f"{domain}_videogpt_l4_s4"),
            camera_key=FLAGS.image_keys.split("|")[0],
            reward_scale=None,
            minibatch_size=16,
            encoding_minibatch_size=16,
        )

    if FLAGS.drs_reward:
        from bpref_v2.reward_model.drs_reward_model import DrsRewardModel

        reward_model = DrsRewardModel(
            task=FLAGS.task_name,
            model_name="DRS",
            rm_path=FLAGS.ckpt_path,
            camera_keys=FLAGS.image_keys.split("|"),
            reward_scale=None,
            window_size=FLAGS.window_size,
            skip_frame=FLAGS.skip_frame,
            reward_model_device=0,
            encoding_minibatch_size=FLAGS.batch_size,
            use_task_reward=False,
            use_scale=False,
        )

    ckpt_dir = os.path.join(FLAGS.save_dir, "ckpt", f"{FLAGS.run_name}.{FLAGS.seed}")
    if not FLAGS.from_scratch:
        agent.load(ckpt_dir, ckpt_step)
        
    eval_returns = []
    observation, done = env.reset(), False
    phase = 0

    observations = []
    actions = []
    rewards = []
    done_floats = []
    masks = []
    next_observations = []
    log_online_avg_reward = []
    log_online_avg_return = []
    log_phases = []
    collected_data = 0

    if "Bench" in FLAGS.env_name:
        assert FLAGS.save_data
        assert FLAGS.keyboard

    finetune_ckpt_dir = os.path.join(
        FLAGS.save_dir,
        "ckpt",
        f"{FLAGS.run_name}-{ckpt_step}-finetune-tmp-{FLAGS.temperature}-bs{FLAGS.batch_size}.{FLAGS.seed}",
    )

    # Load the online data.
    online_data_dir = os.path.join(finetune_ckpt_dir, "online_dataset")
    data_files = glob.glob(os.path.join(online_data_dir, "*.pkl"))
    # Check the online data if enough. >= 20.
    print(f"Loaded {len(data_files)} data files.")
    # Make sure the data is loaded correctly.

    if FLAGS.online_buffer:
        online_dataset = Dataset(
            observations=None,
            actions=None,
            rewards=None,
            masks=None,
            dones_float=None,
            next_observations=None,
            size=0,
        )
    online_traj_for_log = []
    for data_file in tqdm.tqdm(data_files):
        with open(data_file, "rb") as f:
            data = pickle.load(f)
            # Add to the replay buffer
            if FLAGS.online_buffer:
                online_dataset.add_trajectory(
                    data["observations"],
                    data["actions"],
                    data["rewards"],
                    data["masks"],
                    data["done_floats"],
                    data["next_observations"],
                )
            else:
                dataset.add_trajectory(
                    data["observations"],
                    data["actions"],
                    data["rewards"],
                    data["masks"],
                    data["done_floats"],
                    data["next_observations"],
                )
            # Tuple of observations and actions.
            online_traj_for_log.append((data['observations'], data['actions']))

    # Load the fine-tune checkpoint if any.
    ckpt_idx = len(data_files)
    # if ckpt_idx > 0:
    if not FLAGS.data_collection and FLAGS.load_finetune_ckpt:
        # breakpoint()
        agent.load(finetune_ckpt_dir, ckpt_idx - FLAGS.prefill_episodes)
        print(f"Loading fine-tune checkpoint {finetune_ckpt_dir} at {ckpt_idx - FLAGS.prefill_episodes}")
    data_rew_min = np.min(dataset.rewards)

    for i in tqdm.tqdm(
        range(ckpt_idx, FLAGS.max_episodes + FLAGS.prefill_episodes + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        observations = []
        actions = []
        rewards = []
        phases = []
        done_floats = []
        masks = []
        next_observations = []

        online_stds = []

        observation, done = env.reset(), False
        phase = 0

        while not done:
            obs_without_rgb = {k: v for k, v in observation.items() if k != "color_image1" and k != "color_image2"}
            action = agent.sample_actions(obs_without_rgb, temperature=FLAGS.temperature)
            # Get std for logging.
            dists = agent.dist_actions(obs_without_rgb)
            std = dists.stddev()
            online_stds.append(std)
            action = np.clip(action, -1, 1)
            next_observation, reward, done, info = env.step(action)

            if device_interface:
                _, collect_enum = device_interface.get_action()
                if collect_enum in [CollectEnum.FAIL, CollectEnum.SUCCESS]:
                    done = True
                if collect_enum == CollectEnum.REWARD:
                    reward = device_interface.rew_key
                print(env.env_steps)
            if "phase" in info:
                phase = max(phase, info["phase"])
            if not done or "TimeLimit.truncated" in info:
                mask = 1.0
            else:
                mask = 0.0
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            done_floats.append(float(done))
            masks.append(mask)
            next_observations.append(next_observation)
            phases.append(phase)

            observation = next_observation
            
        # env.robot.reset('low')
        
        if device_interface and collect_enum == CollectEnum.FAIL:
            # Skip the data saving, agent update, and evaluation.
            continue
        len_curr_traj = len(observations)
        assert len_curr_traj == len(actions) == len(rewards) == len(masks) == len(next_observations)
        assert done_floats[-1] == 1.0

        if FLAGS.red_reward or FLAGS.viper_reward or FLAGS.drs_reward:
            # compute reds reward
            x = {
                "observations": observations,
                "actions": actions,
                "rewards": phases,
            }
            if FLAGS.reward_type == "DRS":
                drs = True
            else:
                drs = False
            if FLAGS.reward_type == "VIPER":
                viper = True
            else:
                viper = False

            if viper:
                if x["observations"][0]["color_image1"].shape[0] == 3:
                    for i in range(len(x["observations"])):
                        x["observations"][i]["color_image1"] = x["observations"][i]["color_image1"].transpose((1, 2, 0))
                        x["observations"][i]["color_image2"] = x["observations"][i]["color_image2"].transpose((1, 2, 0))
            seq = reward_model(replay_chunk_to_seq(x, FLAGS.window_size, drs=drs, viper=viper))
            rewards = np.asarray([elem[reward_model.PUBLIC_LIKELIHOOD_KEY] for elem in seq])
            if FLAGS.normalization == "min_max":
                min_max_normalize(rewards)
            if FLAGS.normalization == "max":
                rewards = max_normalize(rewards, max_rew)
                
            if len(rewards) < 500:
                # rewards[-1] = 5 * np.min(dataset.rewards)
                rewards[-1] = 5 * data_rew_min

        if FLAGS.save_gif:
            gif_dir = os.path.join(finetune_ckpt_dir, "gifs")
            if not os.path.exists(gif_dir):
                os.makedirs(gif_dir)
            gif_maker(gif_dir, observations, rewards, done_floats, i)
            print('rewards:', rewards)
            print(f"Saved gif at {gif_dir}")

        if i % FLAGS.log_interval == 0:
            train_log_videos = np.asarray(
                [obs["color_image2"].transpose(2, 0, 1) for obs in observations], dtype=np.uint8
            )
            name = "train_video"
            fps = 20
            if FLAGS.wandb:
                log_dict = {name: wandb.Video(train_log_videos, fps=fps, format="mp4")}
                # log_dict = {name: [wandb.Video(vid, fps=fps, format="mp4") for vid in vids]}
                wandb.log(log_dict, step=i)        

        # Remove RGB from the observations.
        observations = [
            {k: v for k, v in obs.items() if k != "color_image1" and k != "color_image2"} for obs in observations
        ]
        next_observations = [
            {k: v for k, v in obs.items() if k != "color_image1" and k != "color_image2"} for obs in next_observations
        ]
        
        if FLAGS.save_data and not FLAGS.eval:
            if not os.path.exists(online_data_dir):
                os.makedirs(online_data_dir)
            data_name = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
            data_path = os.path.join(online_data_dir, f"sample_{data_name}.pkl")
            with open(data_path, "wb") as f:
                data = {
                    "observations": observations,
                    "actions": actions,
                    "rewards": rewards,
                    "masks": masks,
                    "done_floats": done_floats,
                    "next_observations": next_observations,
                }
                pickle.dump(data, f)
                collected_data += 1
            print(f"Data saved at {data_path}")
            print(f"Total data collected: {collected_data}")

            if FLAGS.data_collection:
                continue

        # Remove RGB from the observations.
        observations = [
            {k: v for k, v in obs.items() if k != "color_image1" and k != "color_image2"} for obs in observations
        ]
        next_observations = [
            {k: v for k, v in obs.items() if k != "color_image1" and k != "color_image2"} for obs in next_observations
        ]

        # Append to dataset.
        if FLAGS.online_buffer:
            online_dataset.add_trajectory(observations, actions, rewards, masks, done_floats, next_observations)
        else:
            dataset.add_trajectory(observations, actions, rewards, masks, done_floats, next_observations)

        log_online_avg_reward.append(np.mean(rewards))
        log_online_avg_return.append(np.sum(rewards))
        log_phases.append(phase)
        # for k, v in info['episode'].items():
        #     summary_writer.add_scalar(f'training/{k}', v, info['total']['timesteps'])

        # Update as the length of the current trajectory.
        if i > FLAGS.prefill_episodes and not FLAGS.eval:
            for update_idx in tqdm.trange(
                len_curr_traj, smoothing=0.1, disable=not FLAGS.tqdm, desc="Update", leave=False
            ):
                batch = dataset.sample(FLAGS.batch_size * FLAGS.utd_ratio)
                if FLAGS.online_buffer:
                    online_batch = online_dataset.sample(FLAGS.batch_size * FLAGS.utd_ratio)
                    # Merge batch half by half.
                    batch = combine(batch, online_batch)
                    from dataset_utils import Batch

                    batch = Batch(**batch)
                update_info = agent.update(batch, utd_ratio=FLAGS.utd_ratio)

            for k, v in update_info.items():
                if v.ndim == 0:
                    summary_writer.add_scalar(f"training/{k}", v, i)
                else:
                    summary_writer.add_histogram(f"training/{k}", v, i)

        # Compute the average logprob for the offline and online data.
        offline_logprob, offline_q_value, offline_std = compute_logprob_q_value_std(agent, offline_traj_for_log)
        online_logprob, online_q_value, online_std  = compute_logprob_q_value_std(agent, online_traj_for_log)

        # Flatten and compute the average.
        avg_offline_logprob = np.mean(np.concatenate(offline_logprob))
        avg_online_logprob = np.mean(np.concatenate(online_logprob))
        avg_offline_q_value = np.mean(np.concatenate(offline_q_value))
        avg_online_q_value = np.mean(np.concatenate(online_q_value))
        avg_offline_std = np.mean(np.concatenate(offline_std))
        avg_online_std = np.mean(np.concatenate(online_std))

        summary_writer.add_scalar("offline/average_logprob", avg_offline_logprob, i)
        summary_writer.add_scalar("online/average_logprob", avg_online_logprob, i)
        summary_writer.add_scalar("offline/average_q_value", avg_offline_q_value, i)
        summary_writer.add_scalar("online/average_q_value", avg_online_q_value, i)
        summary_writer.add_scalar("offline/average_std", avg_offline_std, i)
        summary_writer.add_scalar("online/average_std", avg_online_std, i)
 
        summary_writer.add_scalar("online/average_reward", np.mean(log_online_avg_reward), i)
        summary_writer.add_scalar("online/average_return", np.mean(log_online_avg_return), i)
        summary_writer.add_scalar("online/average_std", np.mean(online_stds), i)
        summary_writer.add_scalar("train_phases", np.mean(log_phases), i)
        summary_writer.add_scalar("episode_phase", phase, i)
        summary_writer.flush()

        log_online_avg_reward = []
        log_online_avg_return = []

        if (i - FLAGS.prefill_episodes) % FLAGS.save_interval == 0:
            agent.save(finetune_ckpt_dir, i - FLAGS.prefill_episodes + 1)

        if (
            i > FLAGS.prefill_episodes - 1
            and (i - FLAGS.prefill_episodes) % FLAGS.eval_interval == 0
            and ("Bench" not in FLAGS.env_name)
            and not (i == 0 and FLAGS.from_scratch)
        ):
            if "Sim" in FLAGS.env_name:
                log_video = True
            else:
                log_video = False
            if not FLAGS.red_reward and not FLAGS.viper_reward and not FLAGS.drs_reward:
                reward_model = None

            if FLAGS.red_reward:
                eval_stats, log_videos = evaluate(
                    agent,
                    env,
                    FLAGS.eval_episodes,
                    log_video=log_video,
                    reward_model=reward_model,
                    normalization=FLAGS.normalization,
                    max_rew=max_rew,
                    window_size=FLAGS.window_size,
                )
            else:
                eval_stats, log_videos, eval_traj_for_logging = evaluate(
                    agent,
                    env,
                    FLAGS.eval_episodes,
                    log_video=log_video,
                )
            eval_logprob, eval_q_value, eval_std = compute_logprob_q_value_std(agent, eval_traj_for_logging)
            summary_writer.add_scalar("evaluation/average_logprob", np.mean(np.concatenate(eval_logprob)), i)
            summary_writer.add_scalar("evaluation/average_q_value", np.mean(np.concatenate(eval_q_value)), i)
            summary_writer.add_scalar("evaluation/average_std", np.mean(np.concatenate(eval_std)), i)

            for k, v in eval_stats.items():
                summary_writer.add_scalar(f"evaluation/average_{k}s", v, i)
            summary_writer.flush()

            if log_video:
                max_length = max(vid.shape[0] for vid in log_videos)  # Find the maximum sequence length
                padded_vids = np.array(
                    [
                        np.pad(vid, ((0, max_length - vid.shape[0]), (0, 0), (0, 0), (0, 0)), "constant")
                        for vid in log_videos
                    ]
                )
                # Make it np.int8
                padded_vids = padded_vids.astype(np.uint8)

                name = "rollout_video"
                fps = 20
                vids = rearrange(padded_vids, "b t c h w -> t b c h w")
                # Get dimensions
                num_frames, batch_size, channels, height, width = vids.shape
                # Create a video with graphs for each batch
                new_vids = []
                for b in range(batch_size):
                    batch_vids = []
                    for t in range(num_frames):
                        video_frame = vids[t, b].transpose(1, 2, 0)  # Select the current batch and rearrange for (H, W, C)
                        logprob_frame, q_value_frame, std_frame = create_individual_graph(eval_logprob[b], eval_q_value[b], eval_std[b], t, height, width)
                        combined_frame = np.concatenate((video_frame, logprob_frame, q_value_frame, std_frame), axis=1)
                        batch_vids.append(combined_frame)
                    batch_vids = np.stack(batch_vids, axis=0)
                    new_vids.append(batch_vids)
                new_vids = np.concatenate(new_vids, axis=0)
                new_vids = rearrange(new_vids, "t h w c -> t c h w")  # Rearrange for wandb (T, C, H, W)

                if FLAGS.wandb:
                    log_dict = {name: wandb.Video(new_vids, fps=fps, format="mp4")}
                    wandb.log(log_dict, step=i)

            # eval_returns.append((i, eval_stats['return']))
            # np.savetxt(os.path.join(FLAGS.save_dir, f'{FLAGS.seed}.txt'),
            #            eval_returns,
            #            fmt=['%d', '%.1f'])

        if (
            i > FLAGS.prefill_episodes - 1
            and (i - FLAGS.prefill_episodes) % FLAGS.eval_interval == 0
            and not (i == 0 and FLAGS.from_scratch)
        ):
            # Save last step if it is not saved.
            # if FLAGS.phase_reward:
            #     finetune_ckpt_dir = finetune_ckpt_dir + "-phase-reward"
            if not os.path.exists(finetune_ckpt_dir):
                os.makedirs(finetune_ckpt_dir)
            agent.save(finetune_ckpt_dir, i - FLAGS.prefill_episodes)

    if FLAGS.wandb:
        wandb.finish()


def compute_logprob_q_value_std(agent, traj_for_log):
    log_probs = []
    q_values = []
    stds = []

    # Randomly sample 50 from traj_for_log for speed.
    sample_n = min(50, len(traj_for_log))
    traj_for_log_subset  = random.sample(traj_for_log, sample_n)
    for obs, act in traj_for_log_subset: # For each trajectory.
        # Deepcopy for obs.
        obs = copy.deepcopy(obs)
        # Flatten robot state.
        from furniture_bench.robot.robot_state import filter_and_concat_robot_state
        if isinstance(obs[0]['robot_state'], dict):
            for robot_state_idx in range(len(obs)):
                obs[robot_state_idx]['robot_state'] = filter_and_concat_robot_state(obs[robot_state_idx]['robot_state'])
        # List of dictionary to dictionary of list.
        obs = {k: [obs[i][k] for i in range(len(obs))] for k in obs[0].keys()}
        # Remove `color_image` if exists.
        if 'color_image1' in obs:
            obs.pop('color_image1')
            obs.pop('color_image2')
        # To jnp array.
        obs = {k: jnp.array(v) for k, v in obs.items()}
        act = jnp.array(act)
        # Remove parts_poses if exists.
        if 'parts_poses' in obs:
            obs.pop('parts_poses')
        log_prob = agent.logprob(obs, act)
        log_probs.append(log_prob)
        q_value = agent.q_value(obs, act)
        q_values.append(q_value)

        std = agent.dist_actions(obs).stddev()
        stds.append(std.mean(axis=1)) # along with action space.

    return log_probs, q_values, stds


def create_individual_graph(logprob, q_value, std, t, height, width):
    # Create log probability plot
    fig_logprob, ax_logprob = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax_logprob.plot(logprob[:t+1])
    ax_logprob.set_title('Log Probability')
    ax_logprob.set_xlabel('Time Step')
    ax_logprob.set_ylabel('Log Prob')
    ax_logprob.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax_logprob.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    fig_logprob.subplots_adjust(left=0.3, right=0.95, top=0.9, bottom=0.1)  # Adjust margins

    fig_logprob.canvas.draw()
    img_logprob = np.frombuffer(fig_logprob.canvas.tostring_rgb(), dtype=np.uint8)
    img_logprob = img_logprob.reshape(fig_logprob.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig_logprob)

    # Create Q value plot
    fig_q_value, ax_q_value = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax_q_value.plot(q_value[:t+1])
    ax_q_value.set_title('Q Value')
    ax_q_value.set_xlabel('Time Step')
    ax_q_value.set_ylabel('Q Value')
    ax_q_value.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax_q_value.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    fig_q_value.subplots_adjust(left=0.3, right=0.95, top=0.9, bottom=0.1)  # Adjust margins

    fig_q_value.canvas.draw()
    img_q_value = np.frombuffer(fig_q_value.canvas.tostring_rgb(), dtype=np.uint8)
    img_q_value = img_q_value.reshape(fig_q_value.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig_q_value)

    # Create std plot
    fig_std, ax_std = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax_std.plot(std[:t+1])
    ax_std.set_title('Standard Deviation')
    ax_std.set_xlabel('Time Step')
    ax_std.set_ylabel('Std')
    ax_std.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax_std.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    fig_std.subplots_adjust(left=0.3, right=0.95, top=0.9, bottom=0.1)  # Adjust margins

    fig_std.canvas.draw()
    img_std = np.frombuffer(fig_std.canvas.tostring_rgb(), dtype=np.uint8)
    img_std = img_std.reshape(fig_std.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig_std)

    return img_logprob, img_q_value, img_std


if __name__ == "__main__":
    app.run(main)
