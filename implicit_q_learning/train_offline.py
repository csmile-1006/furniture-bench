import isaacgym  # noqa: F401
import os
from pathlib import Path

import gym
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter
import wandb

from furniture_bench.sim_config import sim_config

import wrappers
from replay_buffer import make_replay_loader
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
flags.DEFINE_integer("skip_frame", 4, "Skipping frame.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_string("data_path", "", "Path to data.")
flags.DEFINE_integer("num_success_demos", 100, "Number of success demonstrations.")
flags.DEFINE_integer("num_failure_demos", 0, "Number of failure demonstrations.")
flags.DEFINE_integer("num_workers", 4, "num_workers must be <= num_envs.")
config_flags.DEFINE_config_file(
    "config",
    "default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)
flags.DEFINE_integer("n_step", 1, "N-step Q-learning.")
flags.DEFINE_string("encoder_type", "", "vip or r3m or liv")
flags.DEFINE_enum("reward_type", "sparse", ["sparse", "step", "ours", "viper", "diffusion"], "reward type")
flags.DEFINE_boolean("wandb", False, "Use wandb")
flags.DEFINE_string("wandb_project", "", "wandb project")
flags.DEFINE_string("wandb_entity", "", "wandb entity")
flags.DEFINE_integer("device_id", 0, "Choose device id for IQL agent.")
flags.DEFINE_float("lambda_mr", 1.0, "lambda value for dataset.")
flags.DEFINE_string("randomness", "low", "randomness of env.")
flags.DEFINE_string("rm_type", "ARP-V2", "type of reward model.")
flags.DEFINE_string("image_keys", "color_image2|color_image1", "image keys used for computing rewards.")
flags.DEFINE_string(
    "rm_ckpt_path",
    "/mnt/changyeon/ICML2024/reward_models",
    "reward model checkpoint base path.",
)


def make_env_and_dataset(
    env_name: str,
    seed: int,
    randomness: str,
    data_path: str,
    encoder_type: str,
    reward_type: str,
):
    #  -> Tuple[gym.Env, D4RLDataset]:
    record_dir = os.path.join(FLAGS.save_dir, "sim_record", env_name, f"{FLAGS.run_name}.{FLAGS.seed}")
    if "Furniture" in env_name:
        import furniture_bench  # noqa: F401

        rm_ckpt_path = (
            Path(FLAGS.rm_ckpt_path).expanduser()
            / FLAGS.env_name.split("/")[-1]
            / f"w{FLAGS.window_size}-s{FLAGS.skip_frame}-nfp1.0-c1.0@0.5-supc1.0-ep0.5-demo100-total-phase"
            / "s0"
            / "best_model.pkl"
        )

        env_id, furniture_name = env_name.split("/")
        env = gym.make(
            env_id,
            num_envs=FLAGS.num_envs,
            furniture=furniture_name,
            data_path=data_path,
            encoder_type=encoder_type,
            headless=True,
            record=True,
            resize_img=True,
            record_every=2,
            randomness=randomness,
            record_dir=record_dir,
            compute_device_id=FLAGS.device_id,
            graphics_device_id=FLAGS.device_id,
            max_env_steps=sim_config["scripted_timeout"][furniture_name] if "Sim" in env_id else 3000,
            window_size=FLAGS.window_size,
            skip_frame=FLAGS.skip_frame,
            rm_type=FLAGS.rm_type,
            rm_ckpt_path=rm_ckpt_path,
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

    dataloader = make_replay_loader(
        replay_dir=Path(data_path).expanduser(),
        max_size=1e6,
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers,
        save_snapshot=True,
        nstep=FLAGS.n_step,
        discount=FLAGS.config.discount,
        buffer_type="offline",
        reward_type=FLAGS.reward_type,
        num_demos={
            "success": FLAGS.num_success_demos,
            "failure": FLAGS.num_failure_demos,
        },
        obs_keys=tuple([key for key in env.observation_space.spaces.keys() if key != "robot_state"]),
        lambda_mr=FLAGS.lambda_mr,
    )
    return env, dataloader


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
        FLAGS.encoder_type,
        FLAGS.reward_type,
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
        obs_keys=tuple([key for key in env.observation_space.spaces.keys() if key != "robot_state"]),
        **kwargs,
    )

    eval_returns = []
    for i, batch in tqdm.tqdm(
        zip(range(1, FLAGS.max_steps + 1), dataset),
        smoothing=0.1,
        disable=not FLAGS.tqdm,
        total=FLAGS.max_steps,
        ncols=0,
        desc="offline training",
    ):
        # batch = dataset.sample(FLAGS.batch_size, gamma=FLAGS.config.discount)
        batch = jax.tree_util.tree_map(lambda x: x.numpy(), batch)

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
