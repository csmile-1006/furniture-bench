import gym
import numpy as np

from dataset_utils import theta_to_quat


class ActionUnnormalizeWrapper(gym.Wrapper):
    def __init__(self, env, action_stat):
        super().__init__(env)
        self.action_stat = action_stat

    def step(self, action):
        delta_pose, delta_theta, gripper_pos = action[:, :3], action[:, 3:6], action[:, 6:]

        denormalized_pose = (delta_pose + 1) / 2 * (
            self.action_stat["high"][:3] - self.action_stat["low"][:3]
        ) + self.action_stat["low"][:3]

        denormalized_theta = (delta_theta + 1) / 2 * (
            self.action_stat["high"][3:6] - self.action_stat["low"][3:6]
        ) + self.action_stat["low"][3:6]
        denormalized_quat = np.asarray([theta_to_quat(elem) for elem in denormalized_theta])

        denormalized_gripper = (gripper_pos + 1) / 2 * (
            self.action_stat["high"][6:] - self.action_stat["low"][6:]
        ) + self.action_stat["low"][6:]

        denormalized_action = np.concatenate([denormalized_pose, denormalized_quat, denormalized_gripper], axis=-1)
        return self.env.step(denormalized_action)

    @property
    def action_space(self):
        if self.env.act_rot_repr == "quat":
            pose_dim = 6
        else:  # axis
            raise NotImplementedError
        low = np.tile(self.action_stat["low"], (self.env.num_envs, 1))
        high = np.tile(self.action_stat["high"], (self.env.num_envs, 1))
        return gym.spaces.Box(low, high, (self.env.num_envs, pose_dim + 1))
