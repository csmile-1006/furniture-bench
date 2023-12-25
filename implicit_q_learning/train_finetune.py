import isaacgym  # noqa: F401
import os

import gym
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter

import wrappers
from dataset_utils import Batch, D4RLDataset, ReplayBuffer, split_into_trajectories, FurnitureDataset
from evaluation import evaluate
from learner import Learner

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "halfcheetah-expert-v2", "Environment name.")
flags.DEFINE_integer("num_envs", 1, "number of parallel envs.")
flags.DEFINE_string("save_dir", "./tmp/", "Tensorboard logging dir.")
flags.DEFINE_string("ckpt_dir", "./tmp/", "Checkpoint dir.")
flags.DEFINE_string("run_name", "debug", "Run specific name")
flags.DEFINE_integer("ckpt_step", 0, "Specific checkpoint step")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 100, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 100000, "Eval interval.")
flags.DEFINE_integer("ckpt_interval", 100000, "Ckpt interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_integer("num_pretraining_steps", int(1e6), "Number of pretraining steps.")
flags.DEFINE_integer("replay_buffer_size", int(1e6), "Replay buffer size (=max_steps if unspecified).")
flags.DEFINE_integer("init_dataset_size", None, "Offline data size (uses all data if unspecified).")
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
flags.DEFINE_boolean("use_arp", False, "Use ARP rewards.")
flags.DEFINE_integer("skip_frame", 16, "how often skip frame.")
flags.DEFINE_integer("num_frames", 4, "Number of frames in context window.")
flags.DEFINE_string("encoder_type", "", "vip or r3m or liv")
flags.DEFINE_boolean("wandb", False, "Use wandb")
flags.DEFINE_string("wandb_project", "", "wandb project")
flags.DEFINE_string("wandb_entity", "", "wandb entity")
flags.DEFINE_integer("device_id", 0, "Choose device id for IQL agent.")
flags.DEFINE_float("lambda_mr", 0.1, "lambda value for dataset.")
flags.DEFINE_string("randomness", "low", "randomness of env.")
flags.DEFINE_string("rm_type", "ARP-V2", "type of reward model.")
flags.DEFINE_string(
    "rm_ckpt_path",
    "/mnt/changyeon/ICML2024/new_arp_v2/reward_learning/furniturebench-one_leg/ARP-V2/furnituresimenv-w4-s16-nfp1.0-liv0.1-c1.0-ep1.0-aug_crop+jitter-liv-img2+1-step-demo500-refactor/s0/best_model.pkl",
    "reward model checkpoint path.",
)


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
    use_arp: bool,
    use_step: bool,
    lambda_mr: float,
):
    #  -> Tuple[gym.Env, D4RLDataset]:
    record_dir = os.path.join(FLAGS.save_dir, "sim_record", f"{FLAGS.run_name}.{FLAGS.seed}")
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
            randomness=randomness,
            record_dir=record_dir,
            compute_device_id=FLAGS.device_id,
            graphics_device_id=FLAGS.device_id,
            window_size=FLAGS.num_frames,
            skip_frame=FLAGS.skip_frame,
            max_env_steps=600 if "Sim" in env_id else 3000,
            rm_type=FLAGS.rm_type,
            rm_ckpt_path=FLAGS.rm_ckpt_path,
            lambda_mr=FLAGS.lambda_mr,
        )
    else:
        env = gym.make(env_name)

    env = wrappers.SinglePrecision(env)
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
    import jax

    jax.config.update("jax_default_device", jax.devices()[FLAGS.device_id])

    os.makedirs(FLAGS.save_dir, exist_ok=True)
    root_logdir = os.path.join(FLAGS.save_dir, "tb", f"{FLAGS.run_name}_{FLAGS.seed}_ft")
    ckpt_dir = os.path.join(FLAGS.save_dir, "ckpt", f"{FLAGS.run_name}.{FLAGS.seed}")
    ft_ckpt_dir = os.path.join(FLAGS.save_dir, "ft_ckpt", f"{FLAGS.run_name}.{FLAGS.seed}")

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
    )

    action_dim = env.action_space.shape[-1]
    replay_buffer = ReplayBuffer(env.observation_space, action_dim, FLAGS.replay_buffer_size or FLAGS.max_steps)
    replay_buffer.initialize_with_dataset(dataset, FLAGS.init_dataset_size)

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

    summary_writer = SummaryWriter(root_logdir, write_to_disk=True)

    agent = Learner(
        FLAGS.seed,
        env.observation_space.sample(),
        env.action_space.sample()[:1],
        max_steps=FLAGS.max_steps,
        **kwargs,
        use_encoder=FLAGS.use_encoder,
    )

    eval_returns = []
    observation, done = env.reset(), np.zeros((env._num_envs,), dtype=bool)
    if FLAGS.run_name != "" and FLAGS.ckpt_step != 0:
        print(f"load trained {FLAGS.ckpt_step} checkpoints from {ckpt_dir}")
        agent.load(ckpt_dir, FLAGS.ckpt_step or FLAGS.max_steps)
        start_step, steps = FLAGS.ckpt_step, FLAGS.ckpt_step + FLAGS.max_steps
    else:
        start_step, steps = -1 * FLAGS.num_pretraining_steps, FLAGS.max_steps

    # Use negative indices for pretraining steps.
    pbar = tqdm.trange(start_step, steps, smoothing=0.1, disable=not FLAGS.tqdm)
    i = start_step

    trajectories = {
        env_idx: {
            key: []
            for key in [
                "observations",
                "actions",
                "rewards",
                "masks",
                "done_floats",
                "next_observations",
            ]
        }
        for env_idx in range(FLAGS.num_envs)
    }
    with pbar:
        while i <= steps:
            if i != start_step and i > 0:
                action = agent.sample_actions(observation, temperature=0.5)
                action = np.clip(action, -1, 1)
                next_observation, reward, done, info = env.step(action)
                for j in range(action.shape[0]):
                    if action[j][6] < 0:
                        action[j] = np.array(action[j])
                        action[j, 3:7] = -1 * action[j, 3:7]  # Make sure quaternion scalar is positive.

                mask = np.zeros((FLAGS.num_envs,), dtype=np.float32)
                for env_idx in range(FLAGS.num_envs):
                    if not done[env_idx] or "TimeLimit.truncated" in info:
                        mask[env_idx] = 1.0
                    else:
                        mask[env_idx] = 0.0
                    trajectories[env_idx]["observations"].append(
                        {key: observation[key][env_idx][-1] for key in observation.keys()}
                    )
                    trajectories[env_idx]["next_observations"].append(
                        {key: next_observation[key][env_idx][-1] for key in next_observation.keys()}
                    )
                    trajectories[env_idx]["actions"].append(action[env_idx])
                    trajectories[env_idx]["rewards"].append(reward[env_idx])
                    trajectories[env_idx]["masks"].append(mask[env_idx])
                    trajectories[env_idx]["done_floats"].append(done[env_idx])

                observation = next_observation

                for env_idx in range(FLAGS.num_envs):
                    if done[env_idx]:
                        replay_buffer.insert_episode(trajectories[env_idx])
                        new_ob = env.reset_env(env_idx)
                        for key in observation:
                            observation[key][env_idx] = new_ob[key]
                        done[env_idx] = False
                        for k, v in info[f"episode_{env_idx}"].items():
                            summary_writer.add_scalar(f"training/{k}", v, info["total"][f"timesteps_{env_idx}"])
                        trajectories[env_idx] = {
                            key: []
                            for key in [
                                "observations",
                                "actions",
                                "rewards",
                                "masks",
                                "done_floats",
                                "next_observations",
                            ]
                        }
            else:
                info = {}
                info["total"] = {"timesteps": i}

            if i != start_step:
                for _ in range(FLAGS.num_envs):
                    batch = replay_buffer.sample(FLAGS.batch_size)
                    if "antmaze" in FLAGS.env_name:
                        batch = Batch(
                            observations=batch.observations,
                            actions=batch.actions,
                            rewards=batch.rewards - 1,
                            masks=batch.masks,
                            next_observations=batch.next_observations,
                        )
                    update_info = agent.update(batch)

                    if i % FLAGS.log_interval == 0:
                        for k, v in update_info.items():
                            if v.ndim == 0:
                                summary_writer.add_scalar(f"training/{k}", v, i)
                            else:
                                summary_writer.add_histogram(f"training/{k}", v, i)

            if i != start_step and i % FLAGS.ckpt_interval == 0:
                if start_step < 0 and i < 0:
                    agent.save(ckpt_dir, i + FLAGS.num_pretraining_steps)
                else:
                    agent.save(ft_ckpt_dir, i)

            if i != start_step and i % FLAGS.eval_interval == 0:
                eval_stats = evaluate(agent, env, FLAGS.eval_episodes)

                for k, v in eval_stats.items():
                    summary_writer.add_scalar(f"evaluation/average_{k}s", v, i)
                summary_writer.flush()

                eval_returns.append((i, eval_stats["return"]))
                np.savetxt(os.path.join(FLAGS.save_dir, f"{FLAGS.seed}.txt"), eval_returns, fmt=["%d", "%.1f"])
                observation, done = env.reset(), np.zeros((env._num_envs,), dtype=bool)

            i += done.shape[0]
            pbar.update(done.shape[0])
            pbar.set_description(f" current {i} / total step {steps}")

    if FLAGS.wandb:
        wandb.finish()


if __name__ == "__main__":
    app.run(main)
