import os

import gym
import isaacgym  # noqa: F401
import numpy as np
import tqdm
import wrappers
from absl import app, flags
from agents import IQLLearner, IQLTransformerLearner  # noqa: F401
from dataset_utils import Batch, D4RLDataset, FurnitureDataset, ReplayBuffer, split_into_trajectories
from evaluation import evaluate
from flax.training import checkpoints
from ml_collections import config_flags
from tensorboardX import SummaryWriter

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "halfcheetah-expert-v2", "Environment name.")
flags.DEFINE_string("save_dir", "./tmp/", "Tensorboard logging dir.")
flags.DEFINE_string("run_name", "", "Run specific name")
flags.DEFINE_integer("ckpt_step", 0, "Specific checkpoint step")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 100000, "Eval interval.")
flags.DEFINE_integer("ckpt_interval", 100000, "Ckpt interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_integer("num_pretraining_steps", int(1e6), "Number of pretraining steps.")
flags.DEFINE_integer("replay_buffer_size", 2000000, "Replay buffer size (=max_steps if unspecified).")
flags.DEFINE_integer("init_dataset_size", None, "Offline data size (uses all data if unspecified).")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_string("data_path", "", "Path to data.")
config_flags.DEFINE_config_file(
    "config",
    "configs/antmaze_finetune_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)
flags.DEFINE_boolean("use_step", False, "Use step rewards.")
flags.DEFINE_boolean("use_arp", False, "Use ARP rewards.")
flags.DEFINE_string("encoder_type", "", "vip or r3m or liv")
flags.DEFINE_boolean("wandb", False, "Use wandb")
flags.DEFINE_string("wandb_project", "", "wandb project")
flags.DEFINE_string("wandb_entity", "", "wandb entity")
flags.DEFINE_integer("device_id", 0, "Choose device id for IQL agent.")
flags.DEFINE_float("lambda_mr", 0.1, "lambda value for dataset.")
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
    encoder_type: str,
    use_arp: bool,
    use_step: bool,
    lambda_mr: float,
    model_cls: str,
):
    #  -> Tuple[gym.Env, D4RLDataset]:
    record_dir = os.path.join(FLAGS.save_dir, "sim_record", f"{FLAGS.run_name}.{FLAGS.seed}")
    if "Furniture" in env_name:
        import furniture_bench  # noqa: F401

        env_id, furniture_name = env_name.split("/")
        env = gym.make(
            env_id,
            furniture=furniture_name,
            data_path=data_path,
            encoder_type=encoder_type,
            headless=True,
            record=True,
            randomness=randomness,
            record_dir=record_dir,
            compute_device_id=FLAGS.device_id,
            graphics_device_id=FLAGS.device_id,
            max_env_steps=600 if "Sim" in env_id else 3000,
        )
    else:
        env = gym.make(env_name)

    env = wrappers.SinglePrecision(env)
    if model_cls == "IQLLearner":
        env = wrappers.FlattenWrapper(env)
    elif model_cls == "IQLTransformerLearner":
        env = wrappers.FrameStackWrapper(env, num_frames=4, skip_frame=16)
    env = wrappers.EpisodeMonitor(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    print("Observation space", env.observation_space)
    print("Action space", env.action_space)

    if "Furniture" in env_name:
        dataset = FurnitureDataset(
            data_path, use_encoder=False, use_arp=use_arp, use_step=use_step, lambda_mr=lambda_mr
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
    root_logdir = os.path.join(FLAGS.save_dir, "tb", str(FLAGS.seed))
    ckpt_dir = os.path.join(FLAGS.save_dir, "ckpt", f"{FLAGS.run_name}.{FLAGS.seed}")
    ft_ckpt_dir = os.path.join(FLAGS.save_dir, "ft_ckpt", f"{FLAGS.run_name}.{FLAGS.seed}")
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    env, dataset = make_env_and_dataset(
        FLAGS.env_name,
        FLAGS.seed,
        FLAGS.randomness,
        FLAGS.data_path,
        FLAGS.encoder_type,
        FLAGS.use_arp,
        FLAGS.use_step,
        FLAGS.lambda_mr,
        model_cls,
    )

    action_dim = env.action_space.shape[0]
    replay_buffer = ReplayBuffer(env.observation_space, action_dim, FLAGS.replay_buffer_size or FLAGS.max_steps)
    replay_buffer.initialize_with_dataset(dataset, FLAGS.init_dataset_size)

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
        print(f"load trained checkpoints from {ckpt_dir}")
        chkpt = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=None, step=FLAGS.ckpt_step)
        agent.load(chkpt)
        start_step, steps = FLAGS.ckpt_step, range(FLAGS.max_steps + 1)
    else:
        start_step, steps = FLAGS.num_pretraining_steps, range(1 - FLAGS.num_pretraining_steps, FLAGS.max_steps + 1)

    # Use negative indices for pretraining steps.
    for i in tqdm.tqdm(steps, smoothing=0.1, disable=not FLAGS.tqdm):
        if i >= 1:
            action, agent = agent.sample_actions(observation)
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
            if i <= 0:
                checkpoints.save_checkpoint(
                    ckpt_dir, agent, step=abs(i) + FLAGS.num_pretraining_steps, keep=20, overwrite=True
                )
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
