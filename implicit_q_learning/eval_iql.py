import isaacgym  # noqa: F401
import os
from typing import Tuple

import gym
import numpy as np
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter

from einops import rearrange
import wrappers
from dataset_utils import (
    Batch,
    D4RLDataset,
    split_into_trajectories,
    FurnitureDataset,
    max_normalize,
    min_max_normalize,
)
from evaluation import evaluate
from learner import Learner


FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "halfcheetah-expert-v2", "Environment name.")
flags.DEFINE_string("save_dir", "./checkpoints/", "Tensorboard logging dir.")
flags.DEFINE_string("run_name", "debug", "Run specific name")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("ckpt_step", 1, "Specific checkpoint step")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("red_reward", False, "Use learned reward")
flags.DEFINE_boolean("wandb", True, "Use wandb")
flags.DEFINE_string("wandb_project", "furniture-reward-eval", "wandb project")
flags.DEFINE_string("wandb_entity", "clvr", "wandb entity")
config_flags.DEFINE_config_file(
    "config",
    "configs/antmaze_finetune_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

flags.DEFINE_multi_string("data_path", "", "Path to data.")
flags.DEFINE_string("normalization", "", "")
flags.DEFINE_integer("iter_n", -1, "Reward relabeling iteration")

flags.DEFINE_boolean("use_layer_norm", None, "Use layer normalization.")
flags.DEFINE_boolean("fixed_init", None, "Use separate online buffer.")
flags.DEFINE_integer("trial", 0, "Trial number.")
flags.DEFINE_float("temperature", 0.1, "Action noise temperature.")

# REDS
flags.DEFINE_string("task_name", "furniture_one_leg", "Name of task name.")
flags.DEFINE_string("ckpt_path", "", "ckpt path of reward model.")
flags.DEFINE_string("image_keys", "color_image2|color_image1", "image keys.")
flags.DEFINE_string("rm_type", "RFE", "reward model type.")
flags.DEFINE_integer("window_size", 4, "window size")
flags.DEFINE_integer("skip_frame", 1, "skip frame")

# DEVICE
flags.DEFINE_integer("device_id", -1, "Device ID for using multiple GPU")


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
    red_reward: bool = False,
    iter_n: int = -1,
) -> Tuple[gym.Env, D4RLDataset]:
    if "Furniture" in env_name:
        import furniture_bench  # noqa: F401

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
            fixed_init=FLAGS.fixed_init,
            from_skill=0,
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
        dataset = FurnitureDataset(data_path, use_encoder=False, red_reward=red_reward, iter_n=iter_n)
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

    ckpt_step = FLAGS.ckpt_step
    root_logdir = os.path.join(FLAGS.save_dir, "tb", f"{FLAGS.run_name}.{FLAGS.seed}.{FLAGS.trial}")
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    env, dataset = make_env_and_dataset(
        FLAGS.env_name,
        # FLAGS.seed,
        FLAGS.trial,
        FLAGS.data_path,
        FLAGS.red_reward,
        FLAGS.iter_n,
    )

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
            + "-eval"
            + f"-step{FLAGS.ckpt_step}",
            config=kwargs,
            sync_tensorboard=True,
        )

    summary_writer = SummaryWriter(root_logdir, write_to_disk=True)

    agent = Learner(
        FLAGS.trial,
        env.observation_space.sample(),
        env.action_space.sample()[np.newaxis],
        max_steps=1e6,
        **kwargs,
        use_encoder=False,
        opt_decay_schedule=None,
        use_layer_norm=True,
    )

    if FLAGS.red_reward:
        # load reward model.
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

    ckpt_dir = os.path.join(FLAGS.save_dir, "ckpt", f"{FLAGS.run_name}.{FLAGS.seed}")
    agent.load(ckpt_dir, ckpt_step)

    if "Sim" in FLAGS.env_name:
        log_video = True
    else:
        log_video = False
    if not FLAGS.red_reward:
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
        eval_stats, log_videos = evaluate(
            agent,
            env,
            FLAGS.eval_episodes,
            log_video=log_video,
        )

    for k, v in eval_stats.items():
        summary_writer.add_scalar(f"evaluation/average_{k}s", v, ckpt_step)
    summary_writer.flush()

    if log_video:
        max_length = max(vid.shape[0] for vid in log_videos)  # Find the maximum sequence length
        padded_vids = np.array(
            [np.pad(vid, ((0, max_length - vid.shape[0]), (0, 0), (0, 0), (0, 0)), "constant") for vid in log_videos]
        )
        # Make it np.int8
        padded_vids = padded_vids.astype(np.uint8)

        name = "rollout_video"
        fps = 20
        vids = rearrange(padded_vids, "b t c h w -> (b t) c h w")
        log_dict = {name: wandb.Video(vids, fps=fps, format="mp4")}
        # log_dict = {name: [wandb.Video(vid, fps=fps, format="mp4") for vid in vids]}
        wandb.log(log_dict, step=ckpt_step)

    if FLAGS.wandb:
        wandb.finish()


if __name__ == "__main__":
    app.run(main)
