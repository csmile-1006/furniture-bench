import sys
import pickle
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import scipy
from absl import app, flags
from ml_collections import ConfigDict

from pathlib import Path

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf

# import pickle
# from types import SimpleNamespace
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# import os
# from pathlib import Path

#sys.path.append("/home/changyeon/ICML2024/BPref-v2/")
#from bpref_v2.data.label_reward_furniturebench import load_reward_model, load_reward_fn  # noqa: E402

FLAGS = flags.FLAGS

flags.DEFINE_string("furniture", None, "Furniture name.")
flags.DEFINE_string("demo_dir", "/home/dongyoon/FB_dataset/raw/low/one_leg/train", "Demonstration dir.")
flags.DEFINE_string("out_file_path", '/home/dongyoon/FB_dataset/raw/low/one_leg/train', "Path to save converted data.")
flags.DEFINE_boolean("use_r3m", False, "Use r3m to encode images.")
flags.DEFINE_boolean("use_vip", True, "Use vip to encode images.")
flags.DEFINE_boolean("use_liv", False, "Use liv to encode images.")
flags.DEFINE_integer("num_threads", int(8), "Set number of threads of PyTorch")
flags.DEFINE_integer("num_demos", None, "Number of demos to convert")
flags.DEFINE_integer("batch_size", 512, "Batch size for encoding images")
flags.DEFINE_string("ckpt_path", "", "ckpt path of reward model.")
flags.DEFINE_string("rm_type", "ARP-V2", "reward model type.")
flags.DEFINE_string("pvr_type", "vip", "pvr type.")

torch.cuda.set_device('cuda:6')
device = torch.device("cuda:6")


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

    env_type = "Image"
    furniture = FLAGS.furniture
    demo_dir = FLAGS.demo_dir
    
    # load reward model.
    #ckpt_path = Path(FLAGS.ckpt_path).expanduser()
    #reward_model = load_reward_model(rm_type=FLAGS.rm_type, ckpt_path=ckpt_path)
    #reward_fn = load_reward_fn(rm_type=FLAGS.rm_type, reward_model=reward_model)
    #pvr_model, pvr_transform, feature_dim = load_embedding(rep=FLAGS.pvr_type)

    dir_path = Path(demo_dir)

    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    step_reward_ = []
    multimodal_reward_ = []
    done_ = []

    if FLAGS.use_r3m:
        # Use R3M for the image encoder.
        from r3m import load_r3m

        encoder = load_r3m("resnet50")

    if FLAGS.use_vip:
        # Use VIP for the image encoder.
        from vip import load_vip

        encoder = load_vip()

    if FLAGS.use_liv:
        # Use LIV for the image encoder.
        from liv import load_liv

        encoder = load_liv()

    if FLAGS.use_r3m or FLAGS.use_vip or FLAGS.use_liv:
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False
        encoder.to("cuda:6")
        device = torch.device("cuda:6")

    files = list(dir_path.glob(r"[0-9]*success.pkl"))
    len_files = len(files)

    if len_files == 0:
        raise ValueError(f"No pkl files found in {dir_path}")

    cnt = 0
    for i, file_path in enumerate(files):
        if FLAGS.num_demos and i == FLAGS.num_demos:
            break
        print(f"Loading [{i+1}/{len_files}] {file_path}...")
        with open(file_path, "rb") as f:
            x = pickle.load(f)

            if len(x["observations"]) == len(x["actions"]):
                # Dummy
                x["observations"].append(x["observations"][-1])
            length = len(x["observations"])

            if FLAGS.use_r3m or FLAGS.use_vip or FLAGS.use_liv:
                img1 = [x["observations"][i]["color_image1"] for i in range(length)]
                img2 = [x["observations"][i]["color_image2"] for i in range(length)]
                img1 = torch.from_numpy(np.stack(img1))
                img2 = torch.from_numpy(np.stack(img2))

                if FLAGS.use_r3m:
                    img1_feature = np.zeros((length, 2048), dtype=np.float32)
                    img2_feature = np.zeros((length, 2048), dtype=np.float32)
                elif FLAGS.use_vip or FLAGS.use_liv:
                    img1_feature = np.zeros((length, 1024), dtype=np.float32)
                    img2_feature = np.zeros((length, 1024), dtype=np.float32)

                with torch.no_grad():
                    # Use batch size.
                    for i in range(0, length, FLAGS.batch_size):
                        img1_feature[i : i + FLAGS.batch_size] = (
                            encoder(img1[i : i + FLAGS.batch_size].to(device).reshape(-1, 3, 224, 224))
                            .cpu()
                            .detach()
                            .numpy()
                        )
                        img2_feature[i : i + FLAGS.batch_size] = (
                            encoder(img2[i : i + FLAGS.batch_size].to(device).reshape(-1, 3, 224, 224))
                            .cpu()
                            .detach()
                            .numpy()
                        )

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
            
            
            rewards = x['viper_reward_16']
            stacked_timesteps = x['viper_stacked_timesteps_16']
            print(stacked_timesteps[16])
            print(stacked_timesteps[20])
            print(stacked_timesteps[103])
            # rewards = gaussian_smoothe(rewards) # 이미 스무딩 해놓음
            cumsum_skills = np.cumsum(x["skills"])

            for i in range(length - 1):
                if FLAGS.use_r3m or FLAGS.use_vip or FLAGS.use_liv:
                    image1 = img1_feature[i]
                    next_image1 = img1_feature[min(i + 1, length - 2)]
                    image2 = img2_feature[i]
                    next_image2 = img1_feature[min(i + 1, length - 2)]
                    timestep = cnt + stacked_timesteps[i]
                    next_timestep = cnt + stacked_timesteps[min(i + 1, length - 2)]
                else:
                    raise ValueError("You have to choose either use_r3m or use_vip or use_liv.")

                obs_.append(
                    {
                        # 'image_feature': feature1,
                        "image1": image1,
                        "image2": image2,
                        "timestep": timestep,
                        "robot_state": x["observations"][i]["robot_state"],
                    }
                )
                next_obs_.append(
                    {
                        # 'image_feature': next_feature1,
                        "image1": next_image1,
                        "image2": next_image2,
                        "timestep": next_timestep,
                        "robot_state": x["observations"][min(i + 1, length - 2)]["robot_state"],
                    }
                )

                action_.append(x["actions"][i])
                reward_.append(x["rewards"][i])
                if i == length - 2:
                    step_reward_.append(cumsum_skills[i] + 1)
                else:
                    step_reward_.append(cumsum_skills[i])
                multimodal_reward_.append(rewards[i])
                done_.append(1 if i == length - 2 else 0)
            cnt += length - 1

    dataset = {
        "observations": obs_,
        "actions": np.array(action_),
        "next_observations": next_obs_,
        "rewards": np.array(reward_),
        #"multimodal_rewards": np.array(multimodal_reward_),
        "diffusion_reward": np.array(multimodal_reward_),
        "step_rewards": np.array(step_reward_),
        "terminals": np.array(done_),
    }

    path = f"data/{env_type}/{furniture}.pkl" if FLAGS.out_file_path is None else FLAGS.out_file_path
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    with Path(path).open("wb") as f:
        pickle.dump(dataset, f)
        print(f"Saved at {path}")


if __name__ == "__main__":
    app.run(main)
