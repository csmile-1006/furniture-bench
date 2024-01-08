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

from diffusion_reward.models.video_models.videogpt.transformer import VideoGPTTransformer
import pickle
from types import SimpleNamespace
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from pathlib import Path

#sys.path.append("/home/changyeon/ICML2024/BPref-v2/")
#from bpref_v2.data.label_reward_furniturebench import load_reward_model, load_reward_fn  # noqa: E402

class CustomVIPER(nn.Module):
    def __init__(self, cfg):
        super(CustomVIPER, self).__init__()

        # load video models
        self.model_cfg = OmegaConf.load(cfg.cfg_path)
        self.model = VideoGPTTransformer(self.model_cfg)
        self.model.load_state_dict(torch.load(cfg.ckpt_path))
        self.model.eval()
        for param in self.model.parameters(): 
            param.requires_grad = False

        # set attribute
        for attr_name, attr_value in vars(cfg).items():
            setattr(self, attr_name, attr_value)
        
    def imgs_to_batch(self, x, reward_type='likelihood'):
        '''
        input:
            imgs: B * T * H * W * C
            (mostly): 1 * T * ...
        '''
        seq_len = x.shape[1]
        num_frames = self.model_cfg.num_frames + 1
        n_skip = self.model_cfg.frame_skip
        subseq_len = num_frames * n_skip

        x = x.permute(0, 1, 4, 2 ,3) # B * T * C * H * W
        embs, indices = self.model.encode_to_z(x) 
        indices = indices.reshape(indices.shape[0], seq_len, -1)
        embs = embs.reshape(embs.shape[0], seq_len, indices.shape[-1], -1)
        
        if reward_type == 'likelihood':
            post_idxes = list(range(seq_len - subseq_len + 1))
            batch_indices = [indices[:, idx:idx+subseq_len:n_skip] for idx in post_idxes]
            batch_indices = torch.stack(batch_indices, dim=0)
            batch_indices = batch_indices.squeeze(1).reshape(batch_indices.shape[0], -1)
            batch_embs = [embs[:, idx:idx+subseq_len:n_skip] for idx in post_idxes]
            batch_embs = torch.stack(batch_embs, dim=0)
            batch_embs = batch_embs.squeeze(1).reshape(batch_embs.shape[0], -1, batch_embs.shape[-1])

            pre_batch_indices = [indices[:, idx].tile((1, num_frames)) for idx in range(subseq_len-1)]
            pre_batch_indices = torch.concat(pre_batch_indices, dim=0)
            batch_indices = torch.concat([pre_batch_indices, batch_indices], dim=0)

            pre_batch_embs = [embs[:, idx].tile((1, num_frames, 1)) for idx in range(subseq_len-1)]
            pre_batch_embs = torch.concat(pre_batch_embs, dim=0)
            batch_embs = torch.concat([pre_batch_embs, batch_embs], dim=0)
        elif reward_type == 'entropy':
            post_idxes = list(range(seq_len - subseq_len + 2))
            batch_indices = [indices[:, idx:idx+subseq_len-n_skip:n_skip] for idx in post_idxes]
            batch_indices = torch.stack(batch_indices, dim=0)
            batch_indices = batch_indices.squeeze(1).reshape(batch_indices.shape[0], -1)
            batch_embs = [embs[:, idx:idx+subseq_len-n_skip:n_skip] for idx in post_idxes]
            batch_embs = torch.stack(batch_embs, dim=0)
            batch_embs = batch_embs.squeeze(1).reshape(batch_embs.shape[0], -1, batch_embs.shape[-1])

            pre_batch_indices = [indices[:, idx].tile((1, num_frames-1)) for idx in range(subseq_len-2)]
            pre_batch_indices = torch.concat(pre_batch_indices, dim=0)
            batch_indices = torch.concat([pre_batch_indices, batch_indices], dim=0)

            pre_batch_embs = [embs[:, idx].tile((1, num_frames-1, 1)) for idx in range(subseq_len-2)]
            pre_batch_embs = torch.concat(pre_batch_embs, dim=0)
            batch_embs = torch.concat([pre_batch_embs, batch_embs], dim=0)
        else:
            raise NotImplementedError

        return batch_embs, batch_indices
    
    @torch.no_grad()
    def calc_reward(self, imgs):
        batch_embs, batch_indices = self.imgs_to_batch(imgs, self.reward_type)
        sos_tokens = self.model.calc_sos_tokens(imgs, batch_embs).tile((batch_embs.shape[0], 1, 1))

        rewards = self.cal_log_prob(batch_embs, batch_indices, sos_tokens, target_indices=batch_indices, reward_type=self.reward_type)
        return rewards  
    
    @torch.no_grad()
    def cal_log_prob(self, embs, x, c, target_indices=None, reward_type='likelihood'):
        self.model.eval()
        if not self.model.use_vqemb:
            x = torch.cat((c, x), dim=1) if x is not None else c   
        else:
            x = torch.cat((c, embs), dim=1) if x is not None else c

        logits, _ = self.model.transformer(x[:, :-1])
        probs = F.log_softmax(logits, dim=-1)

        if reward_type == 'likelihood':
            target = F.one_hot(target_indices, num_classes=self.model_cfg.codec.num_codebook_vectors)
            if self.compute_joint:
                rewards = (probs * target).sum(-1).sum(-1, keepdim=True)
            else:
                num_valid_logits = int(logits.shape[1] // (self.model_cfg.num_frames + 1))
                rewards = (probs * target).sum(-1)[:, -num_valid_logits:].sum(-1, keepdim=True)
        elif reward_type == 'entropy':
            num_valid_logits = int(logits.shape[1] // (self.model_cfg.num_frames))
            entropy = (- probs * probs.exp()).sum(-1)[:, -num_valid_logits:].sum(-1, keepdim=True)
            rewards = - entropy
        else:
            raise NotImplementedError

        # if self.use_std:
        #     rewards_std = (rewards - self.stat[0]) / self.stat[1]
        # scaled_rewards = (1 - self.expl_scale) * rewards_std
        return rewards

    def update(self, batch):
        metrics = dict()

        if self.use_expl_reward:
            metrics.update(self.expl_reward.update(batch))
        return metrics

FLAGS = flags.FLAGS

flags.DEFINE_string("furniture", None, "Furniture name.")
flags.DEFINE_string("demo_dir", "/home/dongyoon/FB_dataset/raw/low/one_leg/train", "Demonstration dir.")
flags.DEFINE_string("out_file_path", '/home/dongyoon/diffusion_reward/dongyoon', "Path to save converted data.")
flags.DEFINE_boolean("use_r3m", False, "Use r3m to encode images.")
flags.DEFINE_boolean("use_vip", True, "Use vip to encode images.")
flags.DEFINE_boolean("use_liv", False, "Use liv to encode images.")
flags.DEFINE_integer("num_threads", int(8), "Set number of threads of PyTorch")
flags.DEFINE_integer("num_demos", None, "Number of demos to convert")
flags.DEFINE_integer("batch_size", 512, "Batch size for encoding images")
flags.DEFINE_string("ckpt_path", "", "ckpt path of reward model.")
flags.DEFINE_string("rm_type", "ARP-V2", "reward model type.")
flags.DEFINE_string("pvr_type", "vip", "pvr type.")


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
        encoder.to("cuda")
        device = torch.device("cuda")

    files = list(dir_path.glob(r"[0-9]*.pkl"))
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
            
            # rewards, (_, stacked_attn_masks, stacked_timesteps) = reward_fn(
            #     images=images,
            #     actions=actions,
            #     skills=skills,
            #     args=args,
            #     pvr_model=pvr_model,
            #     pvr_transform=pvr_transform,
            #     model_type=FLAGS.pvr_type,
            #     feature_dim=feature_dim,
            #     texts=None,
            #     device=device,
            # )
            
            with open('/home/dongyoon/diffusion_reward/dongyoon/config/viper.yaml', 'r') as file:
                config = yaml.safe_load(file)
                config = SimpleNamespace(**config)
            reward_model = CustomVIPER(config)
            if torch.cuda.is_available():
                reward_model = reward_model.to('cuda')

            frames_dy = []
            for ts in range(len(x['observations'])):
                frame = x['observations'][i]['color_image2']
                frame = np.transpose(frame, (1, 2, 0)) # chw -> hwc
                img = Image.fromarray(frame)
                resized_img = img.resize((64, 64))
                frame = np.array(resized_img)
                frames_dy.append(frame)
            frames_dy = np.array(frames_dy)
            frames_dy = np.expand_dims(frames_dy, axis=0) # dim 0 for batch
            frames_dy = frames_dy.astype(np.float32)
            frames_dy = frames_dy / 127.5 - 1 # normalize to [-1, 1]
            frames_dy = torch.from_numpy(frames_dy).float().to('cuda')
            rewards = reward_model.calc_reward(frames_dy)
            
            stacked_timesteps = []
            for ts in range(len(x['observations'])):
                timesteps = np.array((np.max(0, ts-16), np.max(0, ts-12), np.max(0, ts-8), np.max(0, ts-4)))
                stacked_timesteps.append(timesteps)
            stacked_timesteps = np.array(stacked_timesteps)
                  
            rewards = gaussian_smoothe(rewards)
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
        "multimodal_rewards": np.array(multimodal_reward_),
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
