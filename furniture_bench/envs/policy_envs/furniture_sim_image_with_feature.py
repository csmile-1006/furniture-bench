import numpy as np
from gym import spaces

import torch
from kornia.augmentation import Resize, CenterCrop

from furniture_bench.envs.furniture_sim_env import FurnitureSimEnv  # noqa: F401
from furniture_bench.envs.legacy_envs.furniture_sim_legacy_env import FurnitureSimEnvLegacy  # Deprecated. # noqa: F401

from furniture_bench.robot.robot_state import filter_and_concat_robot_state


class FurnitureSimImageWithFeature(FurnitureSimEnv):
    # class FurnitureSimImageFeature(FurnitureSimEnvLegacy):
    def __init__(self, **kwargs):
        super().__init__(
            concat_robot_state=True,
            np_step_out=False,
            channel_first=True,
            **kwargs,
        )
        self._resize_img = kwargs["resize_img"]

        device_id = kwargs["compute_device_id"]
        self._device = torch.device(f"cuda:{device_id}")

        if kwargs["encoder_type"] == "r3m":
            from r3m import load_r3m

            self.vip_layer = load_r3m("resnet50").module
            self.embedding_dim = 2048
        elif kwargs["encoder_type"] == "vip":
            from vip import load_vip

            self.vip_layer = load_vip().module
            self.embedding_dim = 1024
        elif kwargs["encoder_type"] == "liv":
            from liv import load_liv

            self.vip_layer = load_liv().module
            self.embedding_dim = 1024
        self.vip_layer.requires_grad_(False)
        self.vip_layer.eval()
        self.vip_layer = self.vip_layer.to(self._device)

        from liv import load_liv

        self.liv_layer = load_liv().module
        self.liv_layer.requires_grad_(False)
        self.liv_layer.eval()
        self.liv_layer = self.liv_layer.to(self._device)

        # Data Augmentation
        if not self._resize_img:
            self.resize = Resize((224, 224))
            img_size = self.img_size
            ratio = 256 / min(img_size[0], img_size[1])
            ratio_size = (int(img_size[1] * ratio), int(img_size[0] * ratio))
            self.resize_crop = torch.nn.Sequential(Resize(ratio_size), CenterCrop((224, 224)))

    @property
    def observation_space(self):
        robot_state_dim = 14
        img_size = reversed(self.img_size)
        img_shape = (3, *img_size) if self.channel_first else (*img_size, 3)

        return spaces.Dict(
            dict(
                robot_state=spaces.Box(
                    -np.inf,
                    np.inf,
                    (robot_state_dim,),
                ),
                image1=spaces.Box(
                    -np.inf,
                    np.inf,
                    (self.embedding_dim,),
                ),
                image2=spaces.Box(
                    -np.inf,
                    np.inf,
                    (self.embedding_dim,),
                ),
                color_image1=spaces.Box(0, 255, img_shape),
                color_image2=spaces.Box(0, 255, img_shape),
            )
        )

    def _reward(self):
        rewards = super()._reward()
        return rewards.cpu().numpy()

    def _done(self):
        dones = super()._done()
        return dones.cpu().numpy().astype(bool)

    def _get_observation(self):
        obs = super()._get_observation()

        if isinstance(obs["robot_state"], dict):
            # For legacy envs.
            obs["robot_state"] = filter_and_concat_robot_state(obs["robot_state"])

        robot_state = obs["robot_state"]
        image1 = obs["color_image1"]
        image2 = obs["color_image2"]
        if not self._resize_img:
            image1 = self.resize(image1.float())
            image2 = self.resize_crop(image2.float())

        vip_image1, vip_image2 = self._extract_vip_feature(image1), self._extract_vip_feature(image2)
        liv_image1, liv_image2 = self._extract_liv_feature(image1), self._extract_liv_feature(image2)

        # with torch.no_grad():
        #     image1 = torch.tensor(image1).cuda()
        #     image2 = torch.tensor(image2).cuda()

        #     if not self._resize_img:
        #         image1 = self.resize(image1.float())
        #         image2 = self.resize_crop(image2.float())

        #     image1 = self.vip_layer(image1).detach().cpu().numpy()
        #     image2 = self.vip_layer(image2).detach().cpu().numpy()

        return dict(
            robot_state=robot_state.detach().cpu().numpy(),
            image1=vip_image1.detach().cpu().numpy(),
            image2=vip_image2.detach().cpu().numpy(),
            color_image1=liv_image1.detach().cpu().numpy(),
            color_image2=liv_image2.detach().cpu().numpy(),
        )

    def _extract_vip_feature(self, image):
        with torch.no_grad():
            vip_image = self.vip_layer(image)
        return vip_image

    def _extract_liv_feature(self, image):
        with torch.no_grad():
            liv_image = self.liv_layer(image)
        return liv_image
