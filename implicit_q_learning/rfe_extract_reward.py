import datetime
import io
import pickle
from pathlib import Path

import numpy as np
import scipy
import torch
import torchvision.transforms as T
from absl import app, flags
from bpref_v2.data.arp_furniturebench_dataset_inmemory_stream import get_failure_skills_and_phases
from bpref_v2.data.label_reward_furniturebench import load_reward_fn, load_reward_model
from ml_collections import ConfigDict
from rich.console import Console

FLAGS = flags.FLAGS
console = Console()

flags.DEFINE_string("furniture", "one_leg", "Name of furniture.")
flags.DEFINE_string("demo_dir", "square_table_parts_state", "Demonstration dir.")
flags.DEFINE_string("out_dir", None, "Path to save converted data.")
flags.DEFINE_integer("num_success_demos", -1, "Number of demos to convert")
flags.DEFINE_integer("num_failure_demos", -1, "Number of demos to convert")
flags.DEFINE_integer("batch_size", 512, "Batch size for encoding images")
flags.DEFINE_string("ckpt_path", "", "ckpt path of reward model.")
flags.DEFINE_string("demo_type", "success", "type of demonstrations.")
flags.DEFINE_string("rm_type", "RFE", "reward model type.")
flags.DEFINE_string("pvr_type", "liv", "pvr type.")
flags.DEFINE_integer("window_size", 4, "window size")
flags.DEFINE_integer("skip_frame", 1, "skip frame")
flags.DEFINE_boolean("reset_label", False, "use this option for resetting rewards labeled in advance.")
flags.DEFINE_boolean("predict_phase", False, "use phase prediction or not.")
flags.DEFINE_boolean("smoothe", False, "smoothe reward or not.")
flags.DEFINE_boolean("save_reward_stats", False, "save reward stats or not.")


device = torch.device("cuda")


def gaussian_smoothe(rewards, sigma=3.0):
    return scipy.ndimage.gaussian_filter1d(rewards, sigma=sigma, mode="nearest")


def exponential_moving_average(a, alpha=0.3):
    """
    Compute the Exponential Moving Average of a numpy array.

    :param a: Numpy array of values to compute the EMA for.
    :param alpha: Smoothing factor in the range [0,1].
                  The closer to 1, the more weight given to recent values.
    :return: Numpy array containing the EMA of the input array.
    """
    ema = np.zeros_like(a)  # Initialize EMA array with the same shape as input
    ema[0] = a[0]  # Set the first value of EMA to the first value of the input array

    # Compute EMA for each point after the first
    for i in range(1, len(a)):
        ema[i] = alpha * a[i] + (1 - alpha) * ema[i - 1]

    return ema


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open("wb") as f:
            f.write(bs.read())


def load_embedding(rep="vip"):
    if rep == "vip":
        from vip import load_vip

        model = load_vip()
        transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
        feature_dim = 1024
    if rep == "r3m":
        from r3m import load_r3m

        model = load_r3m("resnet50")
        transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
        feature_dim = 2048
    if rep == "liv":
        from liv import load_liv

        model = load_liv()
        transform = T.Compose([T.ToTensor()])
        feature_dim = 1024

    if rep == "clip":
        import clip

        model, transform = clip.load("ViT-B/16")
        feature_dim = 512

    model.eval()
    if rep in ["vip", "r3m", "liv"]:
        model = model.to(device)
    return model, transform, feature_dim


def main(_):
    demo_dir = FLAGS.demo_dir

    # load reward model.
    ckpt_path = Path(FLAGS.ckpt_path).expanduser()
    reward_model = load_reward_model(rm_type=FLAGS.rm_type, task_name=FLAGS.furniture, ckpt_path=ckpt_path)
    reward_fn = load_reward_fn(rm_type=FLAGS.rm_type, reward_model=reward_model)
    pvr_model, pvr_transform, feature_dim = load_embedding(rep=FLAGS.pvr_type)

    dir_path = Path(demo_dir)

    demo_type = [f"_{elem}" for elem in FLAGS.demo_type.split("|")]
    files = []
    for _demo_type in demo_type:
        console.print(f"Loading {_demo_type} demos...")
        demo_files = sorted(list(dir_path.glob(f"*{_demo_type}.pkl")))
        len_demos = (
            getattr(FLAGS, f"num{_demo_type}_demos")
            if getattr(FLAGS, f"num{_demo_type}_demos") > 0
            else len(demo_files)
        )
        files.extend([(idx, path) for idx, path in enumerate(demo_files[:len_demos])])

    len_files = len(files)

    if len_files == 0:
        raise ValueError(f"No pkl files found in {dir_path}")

    out_dir = Path(FLAGS.out_dir).expanduser()
    if not FLAGS.out_dir and not out_dir.exists():
        raise ValueError(f"{FLAGS.out_dir} doesn't exist.")

    reward_stats = []

    for idx, file_path in files:
        console.print(f"Loading [{idx+1}/{len_files}] {file_path}...")
        with open(file_path, "rb") as f:
            x = pickle.load(f)
            tp = file_path.stem.split("_")[-1].split(".")[0]

            path = out_dir / f"{tp}_{idx}_{len(x['actions'])}.npz"
            with path.open("rb") as f:
                dst_dataset = np.load(f, allow_pickle=True)
                dst_dataset = {key: dst_dataset[key] for key in dst_dataset.keys()}

            # first check whether it is already labeled.
            if (
                FLAGS.reset_label
                and dst_dataset.get("multimodal_rewards_ckpt_path", None) is not None
                and dst_dataset.get("multimodal_rewards_ckpt_path").item() == FLAGS.ckpt_path
            ):
                console.print(f"Already labeled with {FLAGS.ckpt_path}")
                reward_stats.append(dst_dataset["multimodal_rewards"])
                continue

            if len(x["observations"]) == len(x["actions"]):
                # Dummy
                x["observations"].append(x["observations"][-1])
            length = len(x["observations"])

            img1 = [x["observations"][_l]["color_image1"] for _l in range(length)]
            img2 = [x["observations"][_l]["color_image2"] for _l in range(length)]
            images = {key: val for key, val in [("color_image2", img2), ("color_image1", img1)]}
            for key, val in images.items():
                val = np.asarray(val)
                val = np.transpose(val, (0, 2, 3, 1))
                images[key] = val

            if not FLAGS.predict_phase:
                skills = np.asarray(x["skills"])
                # actions, skills = x["actions"], np.cumsum(np.where(skills > 0.0, skills, 0.0))
                if "success" in file_path.name:
                    actions, skills = x["actions"], np.cumsum(np.where(skills > 0.0, skills, 0.0))
                else:
                    actions, phase = x["actions"], np.cumsum(np.where(skills > 0.0, skills, 0.0))
                    failure_phase = x.get("failure_phase", -1)
                    _, skills = get_failure_skills_and_phases(
                        skill=skills, phase=phase, task_name=FLAGS.furniture, failure_phase=failure_phase
                    )
            else:
                skills = np.asarray([0 for _ in range(length)])
                actions = np.asarray([0 for _ in range(length)])

            args = ConfigDict()
            args.task_name = FLAGS.furniture
            args.image_keys = "color_image2|color_image1"
            args.window_size = FLAGS.window_size
            args.skip_frame = FLAGS.skip_frame
            args.return_images = True

            # rewards, (_, stacked_attn_masks, stacked_timesteps) = reward_fn(
            output = reward_fn(
                images=images,
                actions=actions,
                skills=skills,
                args=args,
                pvr_model=pvr_model,
                pvr_transform=pvr_transform,
                model_type=FLAGS.pvr_type,
                feature_dim=feature_dim,
                texts=None,
                device=device,
                batch_size=FLAGS.batch_size,
                get_text_feature=True,
                predict_phase=FLAGS.predict_phase,
            )
            rewards = output["rewards"]
            if FLAGS.smoothe:
                # rewards = gaussian_smoothe(rewards)
                rewards = exponential_moving_average(rewards)
            # You have to move one step forward to get the reward for the first action. (r(s,a,s') = r(s'))
            rewards = rewards[1:].tolist()
            rewards = np.asarray(rewards + rewards[-1:]).astype(np.float32)
            reward_stats.append(rewards)

            path = out_dir / f"{tp}_{idx}_{rewards.shape[0]}.npz"

            assert len(dst_dataset["observations"]) == len(
                rewards
            ), f"dst_dataset {len(dst_dataset['observations'])} != multimodal_rewards {len(rewards)}"
            dst_dataset["multimodal_rewards"] = rewards
            for idx in range(len(rewards)):
                dst_dataset["observations"][idx]["text_feature"] = output["text_features"][idx]
                dst_dataset["next_observations"][idx]["text_feature"] = output["text_features"][
                    min(idx + 1, len(rewards) - 1)
                ]
            dst_dataset["timestamp"] = datetime.datetime.now().timestamp()
            dst_dataset["multimodal_rewards_ckpt_path"] = FLAGS.ckpt_path
            save_episode(dst_dataset, path)
            console.print(f"Re-saved at {path}")

    if FLAGS.save_reward_stats:
        console.print("save reward stats.")
        reward_stats = np.concatenate(reward_stats, axis=0)
        stat_file = {
            "mean": np.mean(reward_stats),
            "std": np.std(reward_stats),
            "var": np.var(reward_stats),
            "min": np.min(reward_stats),
            "max": np.max(reward_stats),
        }
        console.print(stat_file)
        save_episode(stat_file, out_dir / "stats.npz")


if __name__ == "__main__":
    app.run(main)
