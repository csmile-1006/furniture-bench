import os

import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter
from flax.training import checkpoints

from dataset_utils import Batch, ReplayBuffer
from evaluation import evaluate
from train_offline import make_env_and_dataset
from agents import IQLLearner, IQLTransformerLearner  # noqa: F401

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "halfcheetah-expert-v2", "Environment name.")
flags.DEFINE_string("save_dir", "./tmp/", "Tensorboard logging dir.")
flags.DEFINE_string("run_name", "", "Run specific name")
flags.DEFINE_string("ckpt_step", 0, "Specific checkpoint step")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 100, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 100000, "Eval interval.")
flags.DEFINE_integer("ckpt_interval", 100000, "Ckpt interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_integer("num_pretraining_steps", int(1e6), "Number of pretraining steps.")
flags.DEFINE_integer("replay_buffer_size", 2000000, "Replay buffer size (=max_steps if unspecified).")
flags.DEFINE_integer("init_dataset_size", None, "Offline data size (uses all data if unspecified).")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("wandb", True, "Use wandb")
flags.DEFINE_STRING("wandb_project", "furniture-bench", "wandb project")
flags.DEFINE_STRING("wandb_entity", "clvr", "wandb entity")
config_flags.DEFINE_config_file(
    "config",
    "configs/antmaze_finetune_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)
flags.DEFINE_boolean("use_encoder", False, "Use ResNet18 for the image encoder.")
flags.DEFINE_boolean("use_step", False, "Use step rewards.")
flags.DEFINE_boolean("use_arp", False, "Use ARP rewards.")
flags.DEFINE_string("encoder_type", "", "vip or r3m or liv")
flags.DEFINE_boolean("wandb", False, "Use wandb")
flags.DEFINE_string("wandb_project", "", "wandb project")
flags.DEFINE_string("wandb_entity", "", "wandb entity")
flags.DEFINE_integer("device_id", 0, "Choose device id for IQL agent.")
flags.DEFINE_float("lambda_mr", 0.1, "lambda value for dataset.")
flags.DEFINE_string("randomness", "low", "randomness of env.")


def main(_):
    root_logdir = os.path.join(FLAGS.save_dir, "tb", str(FLAGS.seed))
    ckpt_dir = os.path.join(FLAGS.save_dir, "ckpt", f"{FLAGS.run_name}.{FLAGS.seed}")
    ft_ckpt_dir = os.path.join(FLAGS.save_dir, "ft_ckpt", f"{FLAGS.run_name}.{FLAGS.seed}")
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    if "Sim" in FLAGS.env_name:
        import isaacgym  # noqa: F401

    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    env, dataset = make_env_and_dataset(
        FLAGS.env_name,
        FLAGS.seed,
        FLAGS.randomness,
        FLAGS.data_path,
        FLAGS.use_encoder,
        FLAGS.encoder_type,
        FLAGS.use_arp,
        FLAGS.use_step,
        FLAGS.lambda_mr,
        model_cls,
    )

    action_dim = env.action_space.shape[0]
    replay_buffer = ReplayBuffer(env.observation_space, action_dim, FLAGS.replay_buffer_size or FLAGS.max_steps)
    replay_buffer.initialize_with_dataset(dataset, FLAGS.init_dataset_size)

    kwargs = dict(FLAGS.config)
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
            + str(FLAGS.data_path.split("/")[-1])
            + "-"
            + str(FLAGS.run_name),
            # config=log_kwargs,
            sync_tensorboard=True,
        )
        wandb.config.update(FLAGS)

    summary_writer = SummaryWriter(root_logdir, write_to_disk=True)

    agent = globals()[model_cls].create(
        seed=FLAGS.seed,
        observation_space=env.observation_space,
        action_space=env.action_space,
        decay_steps=FLAGS.max_steps,
        **kwargs,
    )

    eval_returns = []
    observation, done = env.reset(), False
    if FLAGS.run_name != "" and FLAGS.ckpt_step != 0:
        chkpt = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=None, step=FLAGS.ckpt_step)
        agent.load(chkpt)
        start_step, steps = FLAGS.ckpt_step, range(FLAGS.max_steps + 1)
    else:
        start_step, steps = FLAGS.num_pretraining_steps, range(1 - FLAGS.num_pretraining_steps, FLAGS.max_steps + 1)

    # Use negative indices for pretraining steps.
    for i in tqdm.tqdm(steps, smoothing=0.1, disable=not FLAGS.tqdm):
        if i >= 1:
            action = agent.sample_actions(observation)
            action = np.clip(action, -1, 1)
            next_observation, reward, done, info = env.step(action)

            if not done or "TimeLimit.truncated" in info:
                mask = 1.0
            else:
                mask = 0.0

            replay_buffer.insert(observation, action, reward, mask, float(done), next_observation)
            observation = next_observation

            if done:
                observation, done = env.reset(), False
                for k, v in info["episode"].items():
                    summary_writer.add_scalar(f"training/{k}", v, info["total"]["timesteps"])
        else:
            info = {}
            info["total"] = {"timesteps": i}

        batch = replay_buffer.sample(FLAGS.batch_size)
        if "antmaze" in FLAGS.env_name:
            batch = Batch(
                observations=batch.observations,
                actions=batch.actions,
                rewards=batch.rewards - 1,
                masks=batch.masks,
                next_observations=batch.next_observations,
            )
        agent, update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                if v.ndim == 0:
                    summary_writer.add_scalar(f"training/{k}", v, i)
                else:
                    summary_writer.add_histogram(f"training/{k}", v, i)
            summary_writer.flush()

        if i % FLAGS.ckpt_interval == 0:
            if i < 0:
                checkpoints.save_checkpoint(ckpt_dir, agent, step=abs(i), keep=20, overwrite=True)
            else:
                checkpoints.save_checkpoint(ft_ckpt_dir, agent, step=i + start_step, keep=20, overwrite=True)

        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(agent, env, FLAGS.eval_episodes)

            for k, v in eval_stats.items():
                summary_writer.add_scalar(f"evaluation/average_{k}s", v, i)
            summary_writer.flush()

            eval_returns.append((i, eval_stats["return"]))
            np.savetxt(os.path.join(FLAGS.save_dir, f"{FLAGS.seed}.txt"), eval_returns, fmt=["%d", "%.1f"])

    if FLAGS.wandb:
        wandb.finish()


if __name__ == "__main__":
    app.run(main)
