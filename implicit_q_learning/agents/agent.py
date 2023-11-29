import numpy as np
from flax import struct
from flax.training.train_state import TrainState

from agents.common import eval_actions_jit, sample_actions_jit
from data_types import PRNGKey


class Agent(struct.PyTreeNode):
    actor: TrainState
    rng: PRNGKey

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        actions = eval_actions_jit(self.actor.apply_fn, self.actor.params, observations)
        return np.array(actions)

    def sample_actions(self, observations: np.ndarray) -> np.ndarray:
        rng, actions = sample_actions_jit(self.rng, self.actor.apply_fn, self.actor.params, observations)
        self.replace(rng=rng)
        return np.array(actions)
