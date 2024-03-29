import isaacgym  # noqa: F401

import os
from collections import deque
from pathlib import Path

import clip
import flax.linen as nn
import gym
import numpy as np
import wandb
import wrappers
from absl import app, flags
from agents.bc.bc_learner import BCLearner
from bpref_v2.data.instruct import CLASS_TO_PHASE, get_furniturebench_instruct
from bpref_v2.data.label_reward_furniturebench import _postprocess_phases, load_reward_model
from evaluation import evaluate
from ml_collections import ConfigDict, config_flags
from replay_buffer import Batch, ReplayBufferStorage, make_replay_loader
from rich.console import Console
from tqdm import tqdm, trange

from furniture_bench.sim_config import sim_config

console = Console()
FLAGS = flags.FLAGS

TASK_TO_PHASE = {
    "one_leg": 5,
    "cabinet": 11,
}

# Environment Setting.
flags.DEFINE_string("env_name", "FurnitureSimImageFeature-V0/one_leg", "Environment name.")
flags.DEFINE_integer("num_envs", 1, "number of parallel envs.")
flags.DEFINE_string("randomness", "low", "randomness of env.")
flags.DEFINE_integer("device_id", 0, "Choose device id for IQL agent.")
flags.DEFINE_string("encoder_type", "", "vip or r3m or liv")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 100, "Number of episodes used for evaluation.")

# Logging Setting.
flags.DEFINE_string("save_dir", "./tmp/", "logging dir.")
flags.DEFINE_string("ckpt_dir", "", "Checkpoint dir.")
flags.DEFINE_string("run_name", "debug", "Run specific name")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("wandb", False, "Use wandb")
flags.DEFINE_string("wandb_project", "", "wandb project")
flags.DEFINE_string("wandb_entity", "", "wandb entity")

# Agent Setting.
config_flags.DEFINE_config_file(
    "config",
    "default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)
flags.DEFINE_enum("agent_type", "bc", ["bc"], "agent type.")

# Training Setting.
flags.DEFINE_integer("ckpt_step", 0, "Specific checkpoint step")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_float("offline_ratio", 0.5, "Offline ratio.")
flags.DEFINE_integer("num_iterations", 10, "Number of iterations for online interactions.")
flags.DEFINE_integer("num_demos", 100, "Number of demonstrations to collect in online interactions.")
flags.DEFINE_integer("num_pretraining_steps", int(1e6), "Number of pretraining steps.")
flags.DEFINE_integer("num_training_steps", int(1e6), "Number of training steps in each iteration.")

# Replay Buffer Setting.
flags.DEFINE_integer("n_step", 1, "N-step Q-learning.")
flags.DEFINE_integer("window_size", 4, "Number of frames in context window.")
flags.DEFINE_integer("skip_frame", 1, "how often skip frame.")
flags.DEFINE_integer("replay_buffer_size", int(1e6), "Replay buffer size (=max_steps if unspecified).")
flags.DEFINE_boolean("save_snapshot", False, "save snapshot of replay buffer.")
flags.DEFINE_string("data_path", "", "Path to data.")
flags.DEFINE_integer("num_success_demos", 100, "Number of success demonstrations.")
flags.DEFINE_integer("num_failure_demos", 0, "Number of failure demonstrations.")
flags.DEFINE_integer("num_workers", 1, "num_workers must be <= num_envs.")
flags.DEFINE_float("lambda_mr", 1.0, "lambda value for dataset.")
flags.DEFINE_boolean("prefill_replay_buffer", True, "prefill replay buffer.")
flags.DEFINE_boolean("filter_trajectories", True, "filter trajectory.")

# Reward Model Setting.
flags.DEFINE_string("reward_encoder_type", "", "vip or r3m or liv")
flags.DEFINE_integer("reward_window_size", 4, "Number of frames in context window in reward model.")
flags.DEFINE_integer("reward_skip_frame", 2, "how often skip frame in reward model.")
flags.DEFINE_enum(
    "reward_type", "sparse", ["sparse", "step", "ours", "viper", "diffusion", "ours_shaped"], "reward type"
)
flags.DEFINE_string("rm_type", "RFE", "type of reward model.")
flags.DEFINE_string("image_keys", "color_image2|color_image1", "image keys used for computing rewards.")
flags.DEFINE_string(
    "rm_ckpt_path",
    "/home/changyeon/ICML2024/rfe_checkpoints",
    "reward model checkpoint base path.",
)
flags.DEFINE_boolean("use_text_feature", False, "use text feature for observation.")


def set_seed(seed):
    import random

    import torch

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def compute_multimodal_reward(reward_model, **kwargs):
    trajectories, args = kwargs["trajectories"], kwargs["args"]
    images = trajectories["observations"]
    task_name, image_keys, window_size, skip_frame = (
        args.task_name,
        args.image_keys.split("|"),
        args.window_size,
        args.skip_frame,
    )
    get_video_feature = kwargs.get("get_video_feature", False)
    get_text_feature = kwargs.get("get_text_feature", False)
    img_features = {}
    insts = {
        phase: clip.tokenize(get_furniturebench_instruct(task_name, phase, output_type="all")).detach().cpu().numpy()
        for phase in range(TASK_TO_PHASE[task_name])
    }
    # deal with failure instructions.
    insts[999] = clip.tokenize(get_furniturebench_instruct(task_name, 999, output_type="all")).detach().cpu().numpy()
    for ik in image_keys:
        img_features[ik] = np.stack([images[idx][ik] for idx in range(len(trajectories["actions"]))])
    image_shape, action_dim = img_features[image_keys[0]][0].shape, 8

    def _get_reward(img_features):
        len_demos = len(img_features[image_keys[0]])
        stacked_images = {ik: [] for ik in image_keys}
        stacked_timesteps, stacked_attn_masks = [], []
        image_stacks = {key: {ik: deque([], maxlen=window_size) for ik in image_keys} for key in range(skip_frame)}
        timestep_stacks = {key: deque([], maxlen=window_size) for key in range(skip_frame)}
        attn_mask_stacks = {key: deque([], maxlen=window_size) for key in range(skip_frame)}
        action_stacks = {key: deque([], maxlen=window_size) for key in range(skip_frame)}
        for _ in range(window_size):
            for j in range(skip_frame):
                for ik in image_keys:
                    image_stacks[j][ik].append(np.zeros(image_shape, dtype=np.float32))
                timestep_stacks[j].append(0)
                attn_mask_stacks[j].append(0)
                action_stacks[j].append(np.zeros((action_dim,), dtype=np.float32))

        for i in range(len_demos):
            mod = i % skip_frame
            image_stack, timestep_stack, attn_mask_stack = (
                image_stacks[mod],
                timestep_stacks[mod],
                attn_mask_stacks[mod],
            )
            for ik in image_keys:
                image_stack[ik].append(img_features[ik][i])
                stacked_images[ik].append(np.stack(image_stack[ik]))

            timestep_stack.append(i)
            mask = 1.0 if i != len_demos - 1 else 0.0
            attn_mask_stack.append(mask)

            stacked_timesteps.append(np.stack(timestep_stack))
            stacked_attn_masks.append(np.stack(attn_mask_stack))

        stacked_images = {ik: np.asarray(val) for ik, val in stacked_images.items()}
        stacked_timesteps = np.asarray(stacked_timesteps)
        stacked_attn_masks = np.asarray(stacked_attn_masks)

        rewards, phases, video_features, text_features = [], [], [], []
        cls2phase = CLASS_TO_PHASE[task_name]
        batch_size = 32
        for i in trange(0, len_demos, batch_size, leave=False, ncols=0, desc="predict phase per batch"):
            _range = range(i, min(i + batch_size, len_demos))
            batch = {
                "image": {ik: stacked_images[ik][_range] for ik in image_keys},
                "timestep": stacked_timesteps[_range],
                "attn_mask": stacked_attn_masks[_range],
                # "action": stacked_actions[_range],
            }
            phase = list(np.array(reward_model.get_phase(batch)))
            phases.extend(phase)

        processed_phases = _postprocess_phases(phases)
        for i in trange(0, len_demos, batch_size, leave=False, ncols=0, desc="reward compute per batch"):
            _range = range(i, min(i + batch_size, len_demos))
            batch = {
                "instruct": np.stack([insts[cls2phase[p]] for p in processed_phases[_range]]),
                "image": {ik: stacked_images[ik][_range] for ik in image_keys},
                "timestep": stacked_timesteps[_range],
                "attn_mask": stacked_attn_masks[_range],
                # "action": stacked_actions[_range],
            }
            output = reward_model.get_reward(
                batch, get_video_feature=get_video_feature, get_text_feature=get_text_feature
            )
            rewards.extend(output["rewards"])
            if get_video_feature:
                video_features.extend(output["video_features"])
            if get_text_feature:
                text_features.extend(output["text_features"])

        output = {"rewards": np.asarray(rewards), "processed_phases": np.asarray(processed_phases)}
        if get_video_feature:
            output["video_features"] = np.asarray(video_features)
        if get_text_feature:
            output["text_features"] = np.asarray(text_features)
        return output

    output = _get_reward(img_features=img_features)
    # You have to move one step forward to get the reward for the first action. (r(s,a,s') = r(s'))
    multimodal_rewards = output["rewards"][1:].tolist()
    multimodal_rewards = np.asarray(multimodal_rewards + multimodal_rewards[-1:]).astype(np.float32)
    output.update(
        {
            "rewards": multimodal_rewards,
            "text_features": output.get("text_features", []),
        }
    )
    return output


def load_state(data_path):
    stat_path = data_path / "states.npz"
    if stat_path.exists():
        console.print(f"load state file from {stat_path}.")
        state = np.load(stat_path)
        state = {key: state[key] for key in state}
    else:
        raise ValueError("no state file in this folder.")
    return state


def load_reward_stat(data_path):
    stat_path = data_path / "reward_stats.npz"
    if stat_path.exists():
        console.print(f"load reward stat file from {stat_path}.")
        reward_stat = np.load(stat_path)
        reward_stat = {key: reward_stat[key] for key in reward_stat}
        reward_stat["min"] = reward_stat["min"] - 0.2 * np.fabs(reward_stat["min"])
    else:
        console.print("no stat file in this folder.")
        reward_stat = {"mean": 0.0, "std": 1.0, "var": 1.0, "min": 0.0, "max": 1.0}
    return reward_stat


def load_action_stat(data_path):
    stat_path = data_path / "action_stats.npz"
    if stat_path.exists():
        console.print(f"load action stat file from {stat_path}.")
        action_stat = np.load(stat_path)
        action_stat = {key: action_stat[key] for key in action_stat}
    else:
        console.print("no stat file in this folder.")
        action_stat = {"low": np.full((7,), -1, dtype=np.float32), "high": np.ones((7,), dtype=np.float32)}
    return action_stat


def filter_trajectories(trajectories):
    phase_predicted = trajectories.get("phases", None)  # phase predicted by reward model.
    if phase_predicted is None:
        return trajectories, np.sum(trajectories["rewards"]) > 0

    if np.sum(trajectories["rewards"]) > 0:
        return trajectories, True

    fail_cnt = 0
    fail_threshold = 10
    succ_idx = 0
    for i in range(len(trajectories["phases"])):
        if trajectories["phases"][i] == TASK_TO_PHASE[FLAGS.env_name.split("/")[-1]]:
            fail_cnt += 1
            if fail_cnt >= fail_threshold:
                break
        else:
            succ_idx = i
            fail_cnt = 0
    return {key: val[:succ_idx] for key, val in trajectories.items()}, succ_idx > fail_threshold


def make_env(
    env_name: str,
    seed: int,
    randomness: str,
    encoder_type: str,
    reward_model: nn.Module = None,
    action_stat: dict = None,
    record_dir: str = None,
):
    #  -> Tuple[gym.Env, D4RLDataset]:
    if "Furniture" in env_name:
        import furniture_bench  # noqa: F401

        env_id, furniture_name = env_name.split("/")
        env = gym.make(
            env_id,
            num_envs=FLAGS.num_envs,
            furniture=furniture_name,
            encoder_type=encoder_type,
            reward_encoder_type=FLAGS.reward_encoder_type,
            headless=True,
            record=True,
            resize_img=True,
            randomness=randomness,
            record_every=1,
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
    env = wrappers.ActionUnnormalizeWrapper(env, action_stat)
    env = wrappers.EpisodeMonitor(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    console.print("Observation space", env.observation_space)
    console.print("Action space", env.action_space)

    return env


def make_offline_loader(furniture, env, data_path, batch_size, obs_keys):
    return make_replay_loader(
        furniture=furniture,
        replay_dir=Path(data_path).expanduser(),
        max_size=1e6,
        batch_size=batch_size,
        num_workers=FLAGS.num_workers,
        save_snapshot=True,
        nstep=FLAGS.n_step,
        discount=FLAGS.config.get("discount", 1.0),
        buffer_type="offline",
        reward_type=FLAGS.reward_type,
        num_demos={
            "success": FLAGS.num_success_demos,
            "failure": FLAGS.num_failure_demos,
        },
        obs_keys=obs_keys,
        window_size=FLAGS.window_size,
        skip_frame=FLAGS.skip_frame,
        lambda_mr=FLAGS.lambda_mr,
        action_stat=load_action_stat(Path(data_path)),
        reward_stat=load_reward_stat(Path(data_path)) if "ours" in FLAGS.reward_type else None,
    )


def combine(one_dict, other_dict):
    combined = {}
    if isinstance(one_dict, Batch):
        one_dict, other_dict = one_dict._asdict(), other_dict._asdict()
    for k, v in one_dict.items():
        if isinstance(v, dict):
            combined[k] = combine(v, other_dict[k])
        else:
            tmp = np.concatenate([v, other_dict[k]], axis=0)
            combined[k] = tmp

    return combined


def _initialize_traj_dict():
    trajectories = {
        env_idx: {key: [] for key in ["observations", "actions", "rewards", "terminals", "next_observations", "phases"]}
        for env_idx in range(FLAGS.num_envs)
    }
    return trajectories


def _reset_traj_dict():
    return {key: [] for key in ["observations", "actions", "rewards", "terminals", "next_observations", "phases"]}


def main(_):
    import jax

    jax.config.update("jax_default_device", jax.devices()[FLAGS.device_id])

    os.makedirs(FLAGS.save_dir, exist_ok=True)
    wandb_dir = os.path.join(FLAGS.save_dir, "wandb", FLAGS.env_name, f"{FLAGS.run_name}.{FLAGS.seed}")
    ckpt_dir = os.path.join(FLAGS.save_dir, "ckpt", FLAGS.env_name, f"{FLAGS.run_name}.{FLAGS.seed}")
    ft_ckpt_dir = os.path.join(FLAGS.save_dir, "ft_ckpt", FLAGS.env_name, f"{FLAGS.run_name}.{FLAGS.seed}")
    buffer_dir = os.path.join(FLAGS.save_dir, "buffer", FLAGS.env_name, f"{FLAGS.run_name}.{FLAGS.seed}")
    os.makedirs(wandb_dir, exist_ok=True)
    os.makedirs(buffer_dir, exist_ok=True)

    reward_model = None
    if "ours" in FLAGS.reward_type and FLAGS.rm_ckpt_path != "":
        # load reward model.
        rm_ckpt_path = (
            Path(FLAGS.rm_ckpt_path).expanduser()
            / FLAGS.env_name.split("/")[-1]
            / f"w{FLAGS.reward_window_size}-s{FLAGS.reward_skip_frame}-nfp1.0-supc1.0-ep0.2-demo100-hier-shaped-newsupcon-failneg-supcliv-vitb16-vipil"
            / "s0"
            / "best_phase_model.pkl"
        )
        reward_model = load_reward_model(
            rm_type=FLAGS.rm_type, task_name=FLAGS.env_name.split("/")[-1], ckpt_path=rm_ckpt_path
        )

        args = ConfigDict()
        args.task_name = FLAGS.env_name.split("/")[-1]
        args.image_keys = FLAGS.image_keys
        args.window_size = FLAGS.reward_window_size
        args.skip_frame = FLAGS.reward_skip_frame
        args.lambda_mr = FLAGS.lambda_mr
        args.get_text_feature = FLAGS.use_text_feature

        reward_stat = load_reward_stat(Path(FLAGS.data_path))

    STATES = load_state(Path(FLAGS.data_path))
    action_stat = load_action_stat(Path(FLAGS.data_path))

    record_dir = os.path.join(FLAGS.save_dir, "sim_record", FLAGS.env_name, f"{FLAGS.run_name}.{FLAGS.seed}")
    set_seed(FLAGS.seed)
    env = make_env(
        FLAGS.env_name,
        FLAGS.seed,
        FLAGS.randomness,
        FLAGS.encoder_type,
        record_dir=record_dir,
        reward_model=reward_model,
        action_stat=action_stat,
    )
    if getattr(env.unwrapped, "compute_text_feature", None):
        env.unwrapped.compute_text_feature()

    kwargs = dict(FLAGS.config)
    wandb.init(
        project=FLAGS.wandb_project,
        dir=wandb_dir,
        entity=FLAGS.wandb_entity,
        mode="online" if FLAGS.wandb else "offline",
        name=FLAGS.env_name
        + "-"
        + str(FLAGS.seed)
        + "-"
        + str(FLAGS.data_path.split("/")[-1])
        + "-"
        + str(FLAGS.run_name),
    )
    wandb.config.update(FLAGS)

    offline_batch_size = int(FLAGS.batch_size * FLAGS.offline_ratio)
    online_batch_size = FLAGS.batch_size - offline_batch_size

    sample_obs = env.observation_space.sample()
    if FLAGS.use_text_feature:
        sample_obs["text_feature"] = np.zeros(
            (
                FLAGS.window_size,
                reward_model.config.embd_dim,
            ),
            dtype=np.float32,
        )
    sample_act = env.action_space.sample()[:1]

    obs_keys = [key for key in env.observation_space.spaces.keys() if key not in ["color_image1", "color_image2"]]
    if FLAGS.use_text_feature:
        obs_keys.append("text_feature")
    obs_keys = tuple(sorted(obs_keys))

    if FLAGS.agent_type == "bc":
        agent = BCLearner(
            FLAGS.seed,
            sample_obs,
            sample_act,
            obs_keys=obs_keys,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown agent type: {FLAGS.agent_type}")

    def batch_to_jax(y):
        return jax.tree_util.tree_map(lambda x: x.numpy(), y)

    offline_loader = make_offline_loader(
        str(FLAGS.env_name.split("/")[-1]), env, FLAGS.data_path, FLAGS.batch_size, obs_keys
    )
    if FLAGS.ckpt_dir != "":
        console.print(
            f"load trained checkpoints trained with {FLAGS.num_pretraining_steps} steps from {FLAGS.ckpt_dir}"
        )
        agent.load(f"{FLAGS.ckpt_dir}.{FLAGS.seed}", FLAGS.num_pretraining_steps)
    else:
        console.print("Start pre-training with offline dataset.")
        start_step, steps = 0, FLAGS.num_pretraining_steps
        for i, offline_batch in tqdm(
            zip(range(start_step, steps), offline_loader),
            smoothing=0.1,
            disable=not FLAGS.tqdm,
            ncols=0,
            desc=f"pre-training using {FLAGS.agent_type.upper()}",
            total=FLAGS.num_pretraining_steps,
        ):
            offline_batch = batch_to_jax(offline_batch)
            update_info = agent.update(offline_batch)
            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f"training/{k}": v}, step=i)
        agent.save(ckpt_dir, i)
    del offline_loader

    offline_loader = make_offline_loader(
        str(FLAGS.env_name.split("/")[-1]), env, FLAGS.data_path, offline_batch_size, obs_keys
    )
    replay_storage = ReplayBufferStorage(
        replay_dir=Path(buffer_dir).expanduser(),
        max_env_steps=env.furniture.max_env_steps,
    )
    online_loader = make_replay_loader(
        furniture=FLAGS.env_name.split("/")[-1],
        replay_dir=Path(buffer_dir).expanduser(),
        max_size=FLAGS.replay_buffer_size,
        batch_size=online_batch_size,
        num_workers=FLAGS.num_workers,
        save_snapshot=FLAGS.save_snapshot or FLAGS.prefill_replay_buffer,
        nstep=FLAGS.n_step,
        discount=FLAGS.config.get("discount", 1.0),
        buffer_type="online",
        reward_type=FLAGS.reward_type,
        obs_keys=obs_keys,
        window_size=FLAGS.window_size,
        skip_frame=FLAGS.skip_frame,
        lambda_mr=FLAGS.lambda_mr,
        action_stat=action_stat,
        reward_stat=reward_stat if "ours" in FLAGS.reward_type else None,
        prefill_replay_buffer=FLAGS.prefill_replay_buffer,
        offline_replay_dir=Path(FLAGS.data_path).expanduser(),
        num_demos={
            "success": int(FLAGS.num_success_demos * 0.01),
            "failure": FLAGS.num_failure_demos,
        },
    )

    iterations = FLAGS.num_iterations
    total_train_step = FLAGS.num_pretraining_steps
    total_env_step = 0
    iteration_pbar = trange(iterations, smoothing=0.1, disable=not FLAGS.tqdm, ncols=0, desc="online iterations")
    offline_replay_iter, online_replay_iter = None, None
    num_episodes = 0

    tqdm.write("Evaluation before online iteration.")
    env.set_eval_flag()
    eval_stats = evaluate(agent, env, FLAGS.eval_episodes)
    for k, v in eval_stats.items():
        wandb.log({f"evaluation/{k}": v}, step=total_train_step)
    env.unset_eval_flag()

    it = 0
    with iteration_pbar:
        while it < iterations:
            num_collected_episodes = 0
            observation, done = env.reset_to(STATES), np.zeros((env._num_envs,), dtype=bool)
            trajectories = _initialize_traj_dict()
            collect_demo_pbar = trange(
                FLAGS.num_demos, smoothing=0.1, disable=not FLAGS.tqdm, ncols=0, desc="collecting demos", leave=False
            )
            with collect_demo_pbar:
                env_step = 0
                while num_collected_episodes < FLAGS.num_demos:
                    action = agent.sample_actions(observation)
                    next_observation, reward, done, info = env.step(action)
                    reward, done = reward.squeeze(), done.squeeze()

                    for env_idx in range(FLAGS.num_envs):
                        trajectories[env_idx]["observations"].append(
                            {key: observation[key][env_idx][-1] for key in observation.keys()}
                        )
                        trajectories[env_idx]["next_observations"].append(
                            {key: next_observation[key][env_idx][-1] for key in next_observation.keys()}
                        )
                        trajectories[env_idx]["actions"].append(action[env_idx])
                        trajectories[env_idx]["rewards"].append(reward[env_idx])
                        trajectories[env_idx]["terminals"].append(done[env_idx])

                        if done[env_idx]:
                            if "ours" in FLAGS.reward_type and FLAGS.rm_ckpt_path != "":
                                output = compute_multimodal_reward(
                                    trajectories=trajectories[env_idx], reward_model=reward_model, args=args
                                )
                                trajectories[env_idx]["multimodal_rewards"] = output["rewards"]
                                trajectories[env_idx]["phases"] = output["processed_phases"]
                                if "text_feature" in env.observation_space.keys():
                                    for idx in range(output["rewards"]):
                                        trajectories[env_idx]["observations"][idx]["text_feature"] = output[
                                            "text_features"
                                        ][idx]
                                        trajectories[env_idx]["next_observations"][idx]["text_feature"] = output[
                                            "text_features"
                                        ][min(idx + 1, len(output["rewards"]) - 1)]
                                info[f"episode_{env_idx}"]["return"] = np.sum(output["rewards"])
                                if num_episodes % 5 * FLAGS.num_envs == 0:
                                    jax.clear_caches()

                            trajs, insert_traj = (
                                filter_trajectories(trajectories[env_idx])
                                if FLAGS.filter_trajectories
                                else (trajectories[env_idx], True)
                            )

                            if insert_traj:
                                replay_storage.add_episode(
                                    trajs,
                                    env_idx,
                                    env.episode_cnts[env_idx],
                                )
                                num_collected_episodes += 1
                                collect_demo_pbar.update(1)

                            new_ob = env.reset_env_to(env_idx, STATES[env_idx])
                            for key in next_observation:
                                next_observation[key][env_idx] = new_ob[key]
                            done[env_idx] = False
                            for k, v in info[f"episode_{env_idx}"].items():
                                wandb.log({f"training/{k}": v})
                            trajectories[env_idx] = _reset_traj_dict()
                            num_episodes += 1

                    observation = next_observation
                    env_step += FLAGS.num_envs
                    total_env_step += FLAGS.num_envs
                    collect_demo_pbar.set_description(f"collecting demo: env step {env_step}")

            if offline_replay_iter is None and online_replay_iter is None:
                offline_replay_iter, online_replay_iter = iter(offline_loader), iter(online_loader)

            for _ in (
                training_pbar := trange(
                    FLAGS.num_training_steps, disable=not FLAGS.tqdm, ncols=0, desc=f"online-training in iteration {it}"
                )
            ):
                offline_batch, online_batch = next(offline_replay_iter), next(online_replay_iter)
                offline_batch, online_batch = batch_to_jax(offline_batch), batch_to_jax(online_batch)
                combined = combine(offline_batch, online_batch)
                batch = Batch(**combined)
                if "antmaze" in FLAGS.env_name:
                    batch = Batch(
                        observations=batch.observations,
                        actions=batch.actions,
                        rewards=batch.rewards - 1,
                        masks=batch.masks,
                        next_observations=batch.next_observations,
                    )
                update_info = agent.update(batch)

                total_train_step += 1
                if total_train_step % FLAGS.log_interval == 0:
                    for k, v in update_info.items():
                        wandb.log({f"training/{k}": v}, step=total_train_step)
                    wandb.log({"training/batch_reward_mean": np.mean(batch.rewards)}, step=total_train_step)
                    wandb.log({"training/batch_reward_min": np.min(batch.rewards)}, step=total_train_step)
                    wandb.log({"training/batch_reward_max": np.max(batch.rewards)}, step=total_train_step)
                training_pbar.set_postfix(total_training_step=total_train_step)

            tqdm.write(f"Evaluation in iteration {it}.")
            env.set_eval_flag()
            eval_stats = evaluate(agent, env, FLAGS.eval_episodes, state=STATES)
            for k, v in eval_stats.items():
                wandb.log({f"evaluation/{k}": v}, step=total_train_step)
            env.unset_eval_flag()

            tqdm.write(f"Save checkpoint in iteration {it}.")
            agent.save(ft_ckpt_dir, total_train_step)
            it += 1
            iteration_pbar.update(1)

    if FLAGS.wandb:
        wandb.finish()


if __name__ == "__main__":
    app.run(main)
