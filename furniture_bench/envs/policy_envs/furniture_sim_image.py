import numpy as np
import torch
from gym import spaces
from kornia.augmentation import CenterCrop, Resize

from furniture_bench.envs.furniture_sim_env import FurnitureSimEnv
from furniture_bench.robot.robot_state import filter_and_concat_robot_state


class FurnitureSimImage(FurnitureSimEnv):
    def __init__(self, **kwargs):
        super().__init__(
            concat_robot_state=True,
            np_step_out=True,
            channel_first=True,
            **kwargs,
        )
        self._resize_img = kwargs["resize_img"]
        # Data Augmentation
        if not self._resize_img:
            self.resize = Resize((224, 224))
            img_size = self.img_size
            ratio = 256 / min(img_size[0], img_size[1])
            ratio_size = (int(img_size[1] * ratio), int(img_size[0] * ratio))
            self.resize_crop = torch.nn.Sequential(Resize(ratio_size), CenterCrop((224, 224)))

    @property
    def observation_space(self):
        img_size = self.img_size
        img_shape = (3, *img_size) if self.channel_first else (*img_size, 3)
        robot_state_dim = 14
        return spaces.Dict(
            dict(
                robot_state=spaces.Box(-np.inf, np.inf, (robot_state_dim,)),
                color_image1=spaces.Box(0, 255, img_shape),
                color_image2=spaces.Box(0, 255, img_shape),
            )
        )

    def _done(self):
        dones = super()._done()
        return dones.astype(bool)

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

        return dict(
            robot_state=robot_state,
            color_image1=image1,
            color_image2=image2,
        )
