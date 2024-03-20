import gym


class ActionUnnormalizeWrapper(gym.Wrapper):
    def __init__(self, env, action_stat):
        super().__init__(env)
        self.action_stat = action_stat

    def step(self, action):
        denormalized_action = (action + 1) / 2 * (
            self.action_stat["high"] - self.action_stat["low"]
        ) + self.action_stat["low"]
        return self.env.step(denormalized_action)
