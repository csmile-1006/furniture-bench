import isaacgym  # noqa: F401
import os
from pathlib import Path

import flax.linen as nn
import gym
import numpy as np
import wrappers
from absl import app, flags
from agents.awac.awac_learner import AWACLearner
from agents.dapg.dapg_learner import DAPGLearner
from agents.iql.iql_learner import IQLLearner
from agents.td3.td3_learner import TD3Learner
from agents.bc.bc_learner import BCLearner
from evaluation import evaluate_with_save
from ml_collections import config_flags
from rich.console import Console

from furniture_bench.sim_config import sim_config

console = Console()
FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "halfcheetah-expert-v2", "Environment name.")
flags.DEFINE_integer("num_envs", 1, "number of parallel envs.")
flags.DEFINE_string("ckpt_dir", "./checkpoints/", "Tensorboard logging dir.")
flags.DEFINE_string("save_dir", "./checkpoints/", "Evaluation output dir.")
flags.DEFINE_string("run_name", "debug", "Run specific name")
flags.DEFINE_string("ckpt_step", None, "Specific checkpoint step")
flags.DEFINE_string("randomness", "low", "Random mode")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_integer("from_skill", int(0), "Skill to start from.")
flags.DEFINE_integer("skill", int(-1), "Skill to evaluate.")
flags.DEFINE_integer("high_random_idx", int(0), "High random idx.")
flags.DEFINE_string("data_path", "", "Path to data.")
flags.DEFINE_enum("agent_type", "awac", ["awac", "dapg", "iql", "td3", "bc"], "agent type.")

flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")

flags.DEFINE_integer("window_size", 4, "Number of frames in context window.")
flags.DEFINE_integer("skip_frame", 4, "how often skip frame.")
flags.DEFINE_integer("reward_window_size", 4, "Number of frames in context window in reward model.")
flags.DEFINE_integer("reward_skip_frame", 4, "how often skip frame in reward model.")

flags.DEFINE_boolean("image", True, "Image-based model")
flags.DEFINE_boolean("record", True, "Record video")
flags.DEFINE_string("encoder_type", "r3m", "vip or r3m")
flags.DEFINE_float("temperature", 0.00, "Temperature for the policy.")
flags.DEFINE_boolean("headless", False, "Run in headless mode")
flags.DEFINE_integer("device_id", 0, "device_id")

config_flags.DEFINE_config_file(
    "config",
    "default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def load_action_stat(data_path):
    stat_path = data_path / "action_stats.npz"
    if stat_path.exists():
        print(f"load action stat file from {stat_path}.")
        action_stat = np.load(stat_path)
        action_stat = {key: action_stat[key] for key in action_stat}
    else:
        print("no stat file in this folder.")
        action_stat = {"low": np.full((7,), -1, dtype=np.float32), "high": np.ones((7,), dtype=np.float32)}
    return action_stat


def make_env(
    env_name: str,
    seed: int,
    randomness: str,
    encoder_type: str,
    reward_model: nn.Module = None,
    action_stat: str = None,
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
            encoder_type=encoder_type,
            reward_encoder_type=encoder_type,
            headless=True,
            record=True,
            resize_img=True,
            randomness=randomness,
            record_every=3,
            record_dir=record_dir,
            compute_device_id=FLAGS.device_id,
            graphics_device_id=FLAGS.device_id,
            window_size=FLAGS.reward_window_size,
            skip_frame=FLAGS.reward_skip_frame,
            max_env_steps=sim_config["scripted_timeout"][furniture_name] if "Sim" in env_id else 3000,
            reward_model=reward_model,
        )
    else:
        env = gym.make(env_name)

    env = wrappers.SinglePrecision(env)
    env = wrappers.FrameStackWrapper(env, num_frames=FLAGS.window_size, skip_frame=FLAGS.skip_frame)
    env = wrappers.EpisodeMonitor(env)
    env = wrappers.ActionUnnormalizeWrapper(env, action_stat)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    import random

    import torch

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    console.print("Observation space", env.observation_space)
    console.print("Action space", env.action_space)

    return env


def main(_):
    import jax

    jax.config.update("jax_default_device", jax.devices()[FLAGS.device_id])

    ep_dir = os.path.join(FLAGS.save_dir, FLAGS.env_name, f"{FLAGS.run_name}.{FLAGS.seed}", "episode")
    os.makedirs(ep_dir, exist_ok=True)

    if "Sim" in FLAGS.env_name:
        import isaacgym  # noqa: F401

    action_stat = load_action_stat(Path(FLAGS.data_path))
    env = make_env(
        FLAGS.env_name, FLAGS.seed, FLAGS.randomness, FLAGS.encoder_type, reward_model=None, action_stat=action_stat
    )

    kwargs = dict(FLAGS.config)
    if FLAGS.agent_type == "iql":
        agent = IQLLearner(
            FLAGS.seed,
            env.observation_space.sample(),
            env.action_space.sample()[:1],
            max_steps=FLAGS.max_steps,
            **kwargs,
        )
    elif FLAGS.agent_type == "awac":
        agent = AWACLearner(
            FLAGS.seed,
            env.observation_space.sample(),
            env.action_space.sample()[:1],
            **kwargs,
        )
    elif FLAGS.agent_type == "dapg":
        agent = DAPGLearner(
            FLAGS.seed,
            env.observation_space.sample(),
            env.action_space.sample()[:1],
            **kwargs,
        )
    elif FLAGS.agent_type == "td3":
        agent = TD3Learner(
            FLAGS.seed,
            env.observation_space.sample(),
            env.action_space.sample()[:1],
            **kwargs,
        )
    elif FLAGS.agent_type == "bc":
        agent = BCLearner(
            FLAGS.seed,
            env.observation_space.sample(),
            env.action_space.sample()[:1],
            **kwargs,
            max_steps=FLAGS.max_steps,
        )
    else:
        raise ValueError(f"Unknown agent type: {FLAGS.agent_type}")

    ckpt_dir = os.path.join(FLAGS.ckpt_dir, f"{FLAGS.run_name}.{FLAGS.seed}")
    agent.load(ckpt_dir, FLAGS.ckpt_step or FLAGS.max_steps)

    evaluate_with_save(agent, env, FLAGS.eval_episodes, FLAGS.temperature, Path(ep_dir))


if __name__ == "__main__":
    app.run(main)
