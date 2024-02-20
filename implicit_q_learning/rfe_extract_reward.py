import io
import pickle
import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import scipy
from absl import app, flags
from ml_collections import ConfigDict

from bpref_v2.data.label_reward_furniturebench import load_reward_model, load_reward_fn

FLAGS = flags.FLAGS

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
flags.DEFINE_integer("skip_frame", 16, "skip frame")


device = torch.device("cuda")


def gaussian_smoothe(rewards, sigma=3.0):
    return scipy.ndimage.gaussian_filter1d(rewards, sigma=sigma, mode="nearest")


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
    model.eval()
    model = model.module.to(device)
    return model, transform, feature_dim


def main(_):
    demo_dir = FLAGS.demo_dir

    # load reward model.
    ckpt_path = Path(FLAGS.ckpt_path).expanduser()
    reward_model = load_reward_model(rm_type=FLAGS.rm_type, ckpt_path=ckpt_path)
    reward_fn = load_reward_fn(rm_type=FLAGS.rm_type, reward_model=reward_model)
    pvr_model, pvr_transform, feature_dim = load_embedding(rep=FLAGS.pvr_type)

    dir_path = Path(demo_dir)

    demo_type = [f"_{elem}" for elem in FLAGS.demo_type.split("|")]
    files = []
    for _demo_type in demo_type:
        print(f"Loading {_demo_type} demos...")
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

    statistics = []
    for idx, file_path in files:
        print(f"Loading [{idx+1}/{len_files}] {file_path}...")
        with open(file_path, "rb") as f:
            x = pickle.load(f)
            tp = file_path.stem.split("_")[-1].split(".")[0]

            path = out_dir / f"{tp}_{idx}_{len(x['actions'])}.npz"
            with path.open("rb") as f:
                dst_dataset = np.load(f, allow_pickle=True)
                dst_dataset = {key: dst_dataset[key] for key in dst_dataset.keys()}

            # first check whether it is already labeled.
            if (
                dst_dataset.get("multimodal_rewards_ckpt_path", None) is not None
                and dst_dataset.get("multimodal_rewards_ckpt_path").item() == FLAGS.ckpt_path
            ):
                print(f"Already labeled with {FLAGS.ckpt_path}")
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

            skills = np.asarray(x["skills"])
            actions, skills = x["actions"], np.cumsum(np.where(skills > 0.0, skills, 0.0))
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
            )
            rewards = output["rewards"]
            rewards = gaussian_smoothe(rewards)
            # You have to move one step forward to get the reward for the first action. (r(s,a,s') = r(s'))
            rewards = rewards[1:].tolist()
            rewards = np.asarray(rewards + rewards[-1:]).astype(np.float32)
            statistics.append(rewards)

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
            print(f"Re-saved at {path}")

    statistics = np.concatenate(statistics)
    total_stat = {
        "mean": np.mean(statistics),
        "std": np.std(statistics),
        "var": np.var(statistics),
        "min": np.min(statistics),
        "max": np.max(statistics),
    }
    save_episode(total_stat, out_dir / "stats.npz")


if __name__ == "__main__":
    app.run(main)
