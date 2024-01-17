import sys
import pickle
import itertools
import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import scipy
from absl import app, flags
from ml_collections import ConfigDict

sys.path.append("/home/changyeon/ICML2024/BPref-v2/")
from bpref_v2.data.label_reward_furniturebench import load_reward_model, load_reward_fn  # noqa: E402

FLAGS = flags.FLAGS

flags.DEFINE_string("furniture", None, "Furniture name.")
flags.DEFINE_string("demo_dir", "square_table_parts_state", "Demonstration dir.")
flags.DEFINE_string("out_file_path", None, "Path to save converted data.")
flags.DEFINE_boolean("use_r3m", False, "Use r3m to encode images.")
flags.DEFINE_boolean("use_vip", False, "Use vip to encode images.")
flags.DEFINE_boolean("use_liv", False, "Use liv to encode images.")
flags.DEFINE_integer("num_threads", int(8), "Set number of threads of PyTorch")
flags.DEFINE_integer("num_demos", None, "Number of demos to convert")
flags.DEFINE_integer("batch_size", 512, "Batch size for encoding images")
flags.DEFINE_string("ckpt_path", "", "ckpt path of reward model.")
flags.DEFINE_string("demo_type", "success", "type of demonstrations.")
flags.DEFINE_string("rm_type", "ARP-V2", "reward model type.")
flags.DEFINE_string("pvr_type", "liv", "pvr type.")


device = torch.device("cuda")


def gaussian_smoothe(rewards, sigma=3.0):
    return scipy.ndimage.gaussian_filter1d(rewards, sigma=sigma, mode="nearest")


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
    if FLAGS.num_threads > 0:
        print(f"Setting torch.num_threads to {FLAGS.num_threads}")
        torch.set_num_threads(FLAGS.num_threads)

    demo_dir = FLAGS.demo_dir

    # load reward model.
    ckpt_path = Path(FLAGS.ckpt_path).expanduser()
    reward_model = load_reward_model(rm_type=FLAGS.rm_type, ckpt_path=ckpt_path)
    reward_fn = load_reward_fn(rm_type=FLAGS.rm_type, reward_model=reward_model)
    pvr_model, pvr_transform, feature_dim = load_embedding(rep=FLAGS.pvr_type)

    dir_path = Path(demo_dir)

    multimodal_reward_ = []

    demo_type = [f"_{elem}" for elem in FLAGS.demo_type.split("|")]
    if len(demo_type) == 1:
        files = sorted(list(dir_path.glob(f"*{demo_type[0]}.pkl")))
    else:
        total_files = [sorted(list(dir_path.glob(f"*{_demo_type}.pkl"))) for _demo_type in demo_type]
        file_per_demo = FLAGS.num_demos // len(total_files)
        files = [elem[i] for elem in total_files for i in range(file_per_demo)]

    len_files = len(files)

    if len_files == 0:
        raise ValueError(f"No pkl files found in {dir_path}")

    for idx, file_path in enumerate(files):
        if FLAGS.num_demos and idx == FLAGS.num_demos:
            break
        print(f"Loading [{idx+1}/{len_files}] {file_path}...")
        with open(file_path, "rb") as f:
            x = pickle.load(f)
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

            actions, skills = x["actions"], np.cumsum(x["skills"])
            args = ConfigDict()
            args.task_name = FLAGS.furniture
            args.image_keys = "color_image2|color_image1"
            args.window_size = 4
            args.skip_frame = 16
            args.return_images = True

            rewards, (_, stacked_attn_masks, stacked_timesteps) = reward_fn(
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
            )
            rewards = gaussian_smoothe(rewards)
            # You have to move one step forward to get the reward for the first action. (r(s,a,s') = r(s'))
            rewards = rewards[1:].tolist()
            rewards = np.asarray(rewards + rewards[-1:]).astype(np.float32)
            multimodal_reward_.extend(rewards)

    multimodal_rewards = np.array(multimodal_reward_).astype(np.float32)
    out_file_path = Path(FLAGS.out_file_path).expanduser()
    if not FLAGS.out_file_path and not out_file_path.exists():
        raise ValueError(f"{FLAGS.out_file_path} doesn't exist.")

    with Path(out_file_path).open("rb") as f:
        dst_dataset = pickle.load(f)

    assert len(dst_dataset["observations"]) == len(
        multimodal_rewards
    ), f"dst_dataset {len(dst_dataset['observations'])} != multimodal_rewards {len(multimodal_rewards)}"
    dst_dataset["multimodal_rewards"] = multimodal_rewards
    dst_dataset["timestep"] = datetime.datetime.now().timestamp()

    with Path(out_file_path).open("wb") as f:
        pickle.dump(dst_dataset, f)
        print(f"Re-saved at {out_file_path}")


if __name__ == "__main__":
    app.run(main)
