import io
import pickle
from pathlib import Path

import numpy as np
import scipy
import torch
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string("demo_dir", "square_table_parts_state", "Demonstration dir.")
flags.DEFINE_string("out_dir", None, "Path to save converted data.")
flags.DEFINE_boolean("use_r3m", False, "Use r3m to encode images.")
flags.DEFINE_boolean("use_vip", False, "Use vip to encode images.")
flags.DEFINE_boolean("use_liv", False, "Use liv to encode images.")
flags.DEFINE_boolean("use_liv_intermediate", False, "Use liv to encode images (+ intermediate features).")
flags.DEFINE_boolean("use_raw", False, "Use raw image.")
flags.DEFINE_integer("num_threads", int(8), "Set number of threads of PyTorch")
flags.DEFINE_integer("num_success_demos", -1, "Number of demos to convert")
flags.DEFINE_integer("num_failure_demos", -1, "Number of demos to convert")
flags.DEFINE_integer("batch_size", 512, "Batch size for encoding images")
flags.DEFINE_string("demo_type", "success", "type of demonstrations.")
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


def main(_):
    if FLAGS.num_threads > 0 and (FLAGS.use_r3m or FLAGS.use_vip or FLAGS.use_liv):
        print(f"Setting torch.num_threads to {FLAGS.num_threads}")
        torch.set_num_threads(FLAGS.num_threads)

    demo_dir = FLAGS.demo_dir

    dir_path = Path(demo_dir)

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

    if FLAGS.use_liv_intermediate:
        # Use LIV for the image encoder.
        from bpref_v2.third_party.openai.model import load_liv_model, normalize_image
        from liv import load_liv

        clip_model, clip_model_var, _ = load_liv_model()

        import jax
        import jax.numpy as jnp

        @jax.jit
        def _get_pvr_feature(clip_model_var, images):
            images = jnp.reshape(images, (-1,) + images.shape[-3:])
            images = jnp.moveaxis(images, -3, -1)
            if images.shape[-3] != 224:
                images = jax.image.resize(
                    images, (images.shape[0], 224, 224, images.shape[-1]), method="bicubic"
                )  # to meet the input size of the clip model
            images = (images / 255.0).astype(jnp.float32)
            images = normalize_image(images)
            image_feat = clip_model.apply(
                clip_model_var,
                images,
                method=clip_model.get_full_image_feature,
            )
            all_features = {idx: image_feat[idx] for idx in range(len(image_feat))}

            return all_features

        from functools import partial

        encoder = partial(_get_pvr_feature, clip_model_var)

    if FLAGS.use_r3m or FLAGS.use_vip or FLAGS.use_liv:
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False
        encoder.to("cuda")
        device = torch.device("cuda")

    demo_type = FLAGS.demo_type.split("|")
    files = []
    for _demo_type in demo_type:
        print(f"Loading {_demo_type} demos...")
        demo_files = sorted(list(dir_path.glob(f"*_{_demo_type}.pkl")))
        len_demos = (
            getattr(FLAGS, f"num_{_demo_type}_demos")
            if getattr(FLAGS, f"num_{_demo_type}_demos") > 0
            else len(demo_files)
        )
        files.extend([(idx, path) for idx, path in enumerate(demo_files[:len_demos])])

    len_files = len(files)

    if len_files == 0:
        raise ValueError(f"No pkl files found in {dir_path}")

    for idx, file_path in files:
        obs_ = []
        next_obs_ = []
        action_ = []
        reward_ = []
        step_reward_ = []
        viper_reward_ = []
        diffusion_reward_ = []
        done_ = []

        print(f"Loading [{idx+1}/{len_files}] {file_path}...")
        with open(file_path, "rb") as f:
            x = pickle.load(f)
            tp = file_path.stem.split("_")[-1].split(".")[0]

            enable_viper = x.get("viper_reward", None) is not None
            enable_diffusion = x.get("diffusion_reward", None) is not None
            if len(x["observations"]) == len(x["actions"]):
                # Dummy
                x["observations"].append(x["observations"][-1])
            length = len(x["observations"])

            if FLAGS.use_r3m or FLAGS.use_vip or FLAGS.use_liv:
                img1 = [x["observations"][_l]["color_image1"] for _l in range(length)]
                img2 = [x["observations"][_l]["color_image2"] for _l in range(length)]
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
                    for _l in range(0, length, FLAGS.batch_size):
                        _img1 = img1[_l : _l + FLAGS.batch_size].to(device)
                        _img2 = img2[_l : _l + FLAGS.batch_size].to(device)
                        if FLAGS.use_liv:
                            _img1, _img2 = _img1 / 255.0, _img2 / 255.0
                        img1_feature[_l : _l + FLAGS.batch_size] = encoder(_img1).cpu().detach().numpy()
                        img2_feature[_l : _l + FLAGS.batch_size] = encoder(_img2).cpu().detach().numpy()

            if FLAGS.use_liv_intermediate:
                img1 = [x["observations"][_l]["color_image1"] for _l in range(length)]
                img2 = [x["observations"][_l]["color_image2"] for _l in range(length)]
                img1 = np.stack(img1)
                img2 = np.stack(img2)

                img1_feature, img2_feature = dict(), dict()

                for _l in range(0, length, FLAGS.batch_size):
                    _img1 = img1[_l : _l + FLAGS.batch_size]
                    _img2 = img2[_l : _l + FLAGS.batch_size]
                    _img1_feature, _img2_feature = encoder(_img1), encoder(_img2)
                    for key in _img1_feature:
                        if img1_feature.get(key) is None:
                            img1_feature[key] = []
                        if img2_feature.get(key) is None:
                            img2_feature[key] = []
                        img1_feature[key].append(np.asarray(_img1_feature[key]))
                        img2_feature[key].append(np.asarray(_img2_feature[key]))

                img1_feature = {key: np.concatenate(val, axis=0) for key, val in img1_feature.items()}
                img2_feature = {key: np.concatenate(val, axis=0) for key, val in img2_feature.items()}

            cumsum_skills = np.cumsum(x["skills"])
            for _len in range(length - 1):
                if FLAGS.use_r3m or FLAGS.use_vip or FLAGS.use_liv:
                    image1 = img1_feature[_len]
                    next_image1 = img1_feature[min(_len + 1, length - 2)]
                    image2 = img2_feature[_len]
                    next_image2 = img1_feature[min(_len + 1, length - 2)]
                    obs_.append(
                        {
                            "image1": image1,
                            "image2": image2,
                            "robot_state": x["observations"][_len]["robot_state"],
                        }
                    )
                    next_obs_.append(
                        {
                            "image1": next_image1,
                            "image2": next_image2,
                            "robot_state": x["observations"][min(_len + 1, length - 2)]["robot_state"],
                        }
                    )

                elif FLAGS.use_liv_intermediate:
                    _image1_feature = {f"image1_{key}": img1_feature[key][_len] for key in img1_feature.keys()}
                    _image1_next_feature = {
                        f"image1_{key}": img1_feature[key][min(_len + 1, length - 2)] for key in img1_feature.keys()
                    }
                    _image2_feature = {f"image2_{key}": img2_feature[key][_len] for key in img2_feature.keys()}
                    _image2_next_feature = {
                        f"image2_{key}": img2_feature[key][min(_len + 1, length - 2)] for key in img2_feature.keys()
                    }
                    obs_.append(
                        {
                            **_image1_feature,
                            **_image2_feature,
                            "robot_state": x["observations"][_len]["robot_state"],
                        }
                    )
                    next_obs_.append(
                        {
                            **_image1_feature,
                            **_image2_feature,
                            "robot_state": x["observations"][_len]["robot_state"],
                        }
                    )
                elif FLAGS.use_raw:
                    obs_.append(
                        {
                            "color_image1": x["observations"][_len]["color_image1"],
                            "color_image2": x["observations"][_len]["color_image2"],
                            "robot_state": x["observations"][_len]["robot_state"],
                        }
                    )
                    next_obs_.append(
                        {
                            "color_image1": x["observations"][min(_len + 1, length - 2)]["color_image1"],
                            "color_image2": x["observations"][min(_len + 1, length - 2)]["color_image2"],
                            "robot_state": x["observations"][min(_len + 1, length - 2)]["robot_state"],
                        }
                    )
                else:
                    raise ValueError(
                        "You have to choose either use_r3m or use_vip or use_liv or use_liv_intermediate or use_raw."
                    )

                action_.append(x["actions"][_len])
                reward_.append(x["rewards"][_len])
                if enable_viper:
                    if FLAGS.skip_frame == 4 and FLAGS.window_size == 4:
                        _viper_reward = x["viper_reward"][_len]
                        viper_reward_.append(_viper_reward)
                    if FLAGS.skip_frame == 16 and FLAGS.window_size == 4:
                        _viper_reward = x["viper_reward_16"][_len]
                        viper_reward_.append(_viper_reward)
                if enable_diffusion:
                    if FLAGS.skip_frame == 4 and FLAGS.window_size == 4:
                        _diff_reward = x["diffusion_reward_4"][len]
                        diffusion_reward_.append(_diff_reward)
                    if FLAGS.skip_frame == 1 and FLAGS.window_size == 2:
                        _diff_reward = x["diffusion_reward"][len]
                        diffusion_reward_.append(_diff_reward)
                    if FLAGS.skip_frame == 16 and FLAGS.window_size == 4:
                        _diff_reward = x["diffusion_reward_16"][len]
                        diffusion_reward_.append(_diff_reward)
                step_reward_.append(cumsum_skills[_len] + 1 if _len == length - 2 else cumsum_skills[_len])
                done_.append(1 if _len == length - 2 else 0)

        dataset = {
            "observations": obs_,
            "actions": np.array(action_).astype(np.float32),
            "next_observations": next_obs_,
            "rewards": np.array(reward_).astype(np.float32),
            "viper_rewards": np.array(viper_reward_).astype(np.float32),
            "diffusion_rewards": np.array(diffusion_reward_).astype(np.float32),
            "step_rewards": np.array(step_reward_).astype(np.float32),
            "terminals": np.array(done_),
        }

        path = Path(FLAGS.out_dir)
        path.mkdir(exist_ok=True, parents=True)
        path = path / f"{tp}_{idx}_{dataset['terminals'].shape[0]}.npz"
        save_episode(dataset, path)
        print(f"Saved at {path}")


if __name__ == "__main__":
    app.run(main)
