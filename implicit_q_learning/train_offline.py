import isaacgym  # noqa: F401
import os

import gym
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter
import wandb

from furniture_bench.sim_config import sim_config

import wrappers
from dataset_utils import D4RLDataset, FurnitureDataset, split_into_trajectories
from evaluation import evaluate
from learner import Learner

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "halfcheetah-expert-v2", "Environment name.")
flags.DEFINE_integer("num_envs", 1, "number of parallel envs.")
flags.DEFINE_string("save_dir", "./checkpoints/", "Tensorboard logging dir.")
flags.DEFINE_string("run_name", "debug", "Run specific name")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000000, "Eval interval.")
flags.DEFINE_integer("ckpt_interval", 100000, "Ckpt interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("window_size", 4, "Window size.")
flags.DEFINE_integer("skip_frame", 16, "Skipping frame.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_string("data_path", "", "Path to data.")
config_flags.DEFINE_config_file(
    "config",
    "default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)
flags.DEFINE_boolean("use_encoder", True, "Use ResNet18 for the image encoder.")
flags.DEFINE_boolean("use_step", False, "Use step rewards.")
flags.DEFINE_boolean("use_ours", False, "Use ARP rewards.")
flags.DEFINE_boolean("use_viper", False, "Use VIPER rewards.")
flags.DEFINE_boolean("use_diffusion_reward", False, "Use Diffusion Rewards.")
flags.DEFINE_string("encoder_type", "", "vip or r3m or liv")
flags.DEFINE_boolean("wandb", False, "Use wandb")
flags.DEFINE_string("wandb_project", "", "wandb project")
flags.DEFINE_string("wandb_entity", "", "wandb entity")
flags.DEFINE_integer("device_id", 0, "Choose device id for IQL agent.")
flags.DEFINE_float("lambda_mr", 1.0, "lambda value for dataset.")
flags.DEFINE_string("randomness", "low", "randomness of env.")


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


def make_env_and_dataset(
    env_name: str,
    seed: int,
    randomness: str,
    data_path: str,
    use_encoder: bool,
    encoder_type: str,
    use_ours: bool,
    use_step: bool,
    use_viper: bool,
    use_diffusion_reward: bool,
    lambda_mr: float,
):
    #  -> Tuple[gym.Env, D4RLDataset]:
    record_dir = os.path.join(FLAGS.save_dir, "sim_record", env_name, f"{FLAGS.run_name}.{FLAGS.seed}")
    if "Furniture" in env_name:
        import furniture_bench  # noqa: F401

        env_id, furniture_name = env_name.split("/")
        env = gym.make(
            env_id,
            num_envs=FLAGS.num_envs,
            furniture=furniture_name,
            data_path=data_path,
            use_encoder=use_encoder,
            encoder_type=encoder_type,
            headless=True,
            record=True,
            resize_img=True,
            record_every=FLAGS.num_envs,
            randomness=randomness,
            record_dir=record_dir,
            compute_device_id=FLAGS.device_id,
            graphics_device_id=FLAGS.device_id,
            max_env_steps=sim_config["scripted_timeout"][furniture_name] if "Sim" in env_id else 3000,
        )
    else:
        env = gym.make(env_name)

    env = wrappers.SinglePrecision(env)
    env = wrappers.FrameStackWrapper(env, num_frames=FLAGS.window_size, skip_frame=FLAGS.skip_frame)
    env = wrappers.EpisodeMonitor(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    import torch
    import random

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    print("Observation space", env.observation_space)
    print("Action space", env.action_space)

    if "Furniture" in env_name:
        dataset = FurnitureDataset(
            data_path,
            use_encoder=False,
            use_ours=use_ours,
            use_step=use_step,
            lambda_mr=lambda_mr,
            use_viper=use_viper,
            use_diffusion_reward=use_diffusion_reward,
        )
    else:
        dataset = D4RLDataset(env)

    if "antmaze" in env_name:
        dataset.rewards -= 1.0
        # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
        # but I found no difference between (x - 0.5) * 4 and x - 1.0
    # elif FLAGS.use_ours:
    #     print("normalize dataset for arpv2 rewards.")
    #     normalize(dataset)
    elif "halfcheetah" in env_name or "walker2d" in env_name or "hopper" in env_name:
        normalize(dataset)

    return env, dataset


def main(_):
    import jax

    jax.config.update("jax_default_device", jax.devices()[FLAGS.device_id])

    os.makedirs(FLAGS.save_dir, exist_ok=True)
    tb_dir = os.path.join(FLAGS.save_dir, "tb", FLAGS.env_name, f"{FLAGS.run_name}.{FLAGS.seed}")
    ckpt_dir = os.path.join(FLAGS.save_dir, "ckpt", FLAGS.env_name, f"{FLAGS.run_name}.{FLAGS.seed}")

    env, dataset = make_env_and_dataset(
        FLAGS.env_name,
        FLAGS.seed,
        FLAGS.randomness,
        FLAGS.data_path,
        FLAGS.use_encoder,
        FLAGS.encoder_type,
        FLAGS.use_ours,
        FLAGS.use_step,
        FLAGS.use_viper,
        FLAGS.use_diffusion_reward,
        FLAGS.lambda_mr,
    )

    kwargs = dict(FLAGS.config)

    if FLAGS.wandb:
        wandb.init(
            project=FLAGS.wandb_project,
            entity=FLAGS.wandb_entity,
            name=FLAGS.env_name
            + "-"
            + str(FLAGS.seed)
            + "-"
            + str(FLAGS.data_path.split("/")[-1])
            + "-"
            + str(FLAGS.run_name),
            sync_tensorboard=True,
        )
        wandb.config.update(FLAGS)

    summary_writer = SummaryWriter(tb_dir, write_to_disk=True)

    agent = Learner(
        FLAGS.seed,
        env.observation_space.sample(),
        env.action_space.sample()[:1],
        max_steps=FLAGS.max_steps,
        **kwargs,
        use_encoder=FLAGS.use_encoder,
    )

    eval_returns = []
    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm, ncols=0, desc="offline training"
    ):
        batch = dataset.sample(FLAGS.batch_size)

        update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                if v.ndim == 0:
                    summary_writer.add_scalar(f"training/{k}", v, i)
                else:
                    summary_writer.add_histogram(f"training/{k}", np.array(v), i)
            summary_writer.flush()

        if i % FLAGS.ckpt_interval == 0:
            agent.save(ckpt_dir, i)

        if i % FLAGS.eval_interval == 0:
            os.makedirs(ckpt_dir, exist_ok=True)
            env.env.episode_cnts = np.zeros(env.env.num_envs, dtype=np.int32)
            eval_stats = evaluate(agent, env, FLAGS.eval_episodes)

            for k, v in eval_stats.items():
                summary_writer.add_scalar(f"evaluation/average_{k}s", v, i)
            summary_writer.flush()

            eval_returns.append((i, eval_stats["return"]))
            np.savetxt(
                os.path.join(ckpt_dir, f"{FLAGS.seed}.txt"),
                eval_returns,
                fmt=["%d", "%.1f"],
            )

    if not i % FLAGS.ckpt_interval == 0:
        # Save last step if it is not saved.
        agent.save(ckpt_dir, i)

    if FLAGS.wandb:
        wandb.finish()


if __name__ == "__main__":
    app.run(main)
