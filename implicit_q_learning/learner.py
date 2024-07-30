"""Implementations of algorithms for continuous control."""

from typing import Optional, Sequence, Tuple
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

import policy
import value_net
from actor import update as awr_update_actor
from actor import ddpg_bc_update
from common import Batch, InfoDict, Model, PRNGKey
from critic import update_value_critic


def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_map(lambda p, tp: p * tau + tp * (1 - tau), critic.params, target_critic.params)

    return target_critic.replace(params=new_target_params)


@partial(jax.jit, static_argnames=("utd_ratio", "num_qs", "num_min_qs", "policy_ddpg_bc"))
def _update_jit(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    value: Model,
    target_critic: Model,
    batch: Batch,
    discount: float,
    tau: float,
    expectile: float,
    temperature: float,
    utd_ratio: int,
    num_qs: int,
    num_min_qs: int,
    policy_ddpg_bc: bool
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, InfoDict]:

    new_actor = actor
    new_value = value
    new_critic = critic
    new_target_critic = target_critic

    for i in range(utd_ratio):

        def slice(x):
            assert x.shape[0] % utd_ratio == 0
            batch_size = x.shape[0] // utd_ratio
            return x[batch_size * i : batch_size * (i + 1)]

        mini_batch = jax.tree_util.tree_map(slice, batch)

        key, rng = jax.random.split(rng)
        new_critic, new_value, critic_value_info = update_value_critic(
            key, new_critic, new_value, new_target_critic, mini_batch, discount, expectile, num_qs, num_min_qs
        )
        new_target_critic = target_update(new_critic, new_target_critic, tau)
        if utd_ratio == 1 or (utd_ratio > 1 and (i + 1) % 5 == 0):
            key, rng = jax.random.split(rng)
            if policy_ddpg_bc:
                new_actor, actor_info = ddpg_bc_update(key, actor, new_target_critic, mini_batch, temperature, num_min_qs, num_qs)
            else:
                new_actor, actor_info = awr_update_actor(key, actor, new_target_critic, new_value, mini_batch, temperature)
        else:
            new_actor, actor_info = actor, {}

    return (
        rng,
        new_actor,
        new_critic,
        new_value,
        new_target_critic,
        {**actor_info, **critic_value_info},
    )


class Learner(object):
    def __init__(
        self,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        actor_lr: float = 3e-4,
        value_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (512, 256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        expectile: float = 0.8,
        temperature: float = 0.1,
        dropout_rate: Optional[float] = None,
        max_steps: Optional[int] = None,
        opt_decay_schedule: str = "cosine",
        use_encoder: bool = False,
        use_layer_norm: bool = False,
        num_qs: int = 2,
        num_min_qs: int = 2,
        policy_ddpg_bc: bool = False,
        det_policy: bool= False,
        state_dependent_std: bool = False,
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1801.01290
        """

        self.expectile = expectile
        self.tau = tau
        self.discount = discount
        self.temperature = temperature
        self.policy_ddpg_bc = policy_ddpg_bc

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

        if len(observations["image1"].shape) == 3 or len(observations["image1"].shape) == 1:
            observations["image1"] = observations["image1"][np.newaxis]
            observations["image2"] = observations["image2"][np.newaxis]
        if len(observations["robot_state"].shape) == 1:
            observations["robot_state"] = observations["robot_state"][np.newaxis]

        action_dim = actions.shape[-1]
        actor_def = policy.NormalTanhPolicy(
            hidden_dims,
            action_dim,
            log_std_scale=1e-3,
            dropout_rate=dropout_rate,
            state_dependent_std=state_dependent_std,
            det_policy=det_policy,
            tanh_squash_distribution=False,
            use_encoder=use_encoder,
            use_layer_norm=False,
        )

        if opt_decay_schedule == "cosine":
            schedule_fn = optax.cosine_decay_schedule(-actor_lr, max_steps)
            optimiser = optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(schedule_fn))
        else:
            optimiser = optax.adam(learning_rate=actor_lr)

        actor = Model.create(actor_def, inputs=[actor_key, observations], tx=optimiser)

        critic_cls = partial(
            value_net.Critic,
            hidden_dims=hidden_dims,
            use_encoder=use_encoder,
            use_layer_norm=use_layer_norm,
        )
        critic_def = value_net.Ensemble(net_cls=critic_cls, num=num_qs)
        # critic_def = value_net.DoubleCritic(hidden_dims, use_encoder=use_encoder, use_layer_norm=use_layer_norm)
        critic = Model.create(
            critic_def,
            inputs=[critic_key, observations, actions],
            tx=optax.adam(learning_rate=critic_lr),
        )

        value_def = value_net.ValueCritic(hidden_dims, use_encoder=use_encoder, use_layer_norm=use_layer_norm)
        value = Model.create(
            value_def,
            inputs=[value_key, observations],
            tx=optax.adam(learning_rate=value_lr),
        )

        target_critic = Model.create(critic_def, inputs=[critic_key, observations, actions])

        self.actor = actor
        self.critic = critic
        self.value = value
        self.target_critic = target_critic
        self.rng = rng

        self.num_qs = num_qs
        self.num_min_qs = num_min_qs

    def logprob(self, observations: np.ndarray, actions: np.ndarray) -> jnp.ndarray:
        """Compute the log probability of the actions under the current policy."""
        return policy.logprob(self.actor.apply_fn, self.actor.params, observations, actions)
    
    def q_value(self, observations: np.ndarray, actions: np.ndarray) -> jnp.ndarray:
        """Compute the Q value of the actions under the current policy."""
        # return self.critic.apply_fn(self.critic.params, observations, actions)
        from critic import subsample_ensemble
        key, _ = jax.random.split(self.rng)
        target_params = subsample_ensemble(key, self.target_critic.params, self.num_min_qs, self.num_qs)
        qs = self.target_critic.apply({"params": target_params}, observations, actions)
        q = jnp.min(qs, axis=0)
        return q

    def sample_actions(self, observations: np.ndarray, temperature: float = 1.0) -> jnp.ndarray:
        if 'parts_poses' in observations:
            del observations['parts_poses']

        rng, actions = policy.sample_actions(
            self.rng, self.actor.apply_fn, self.actor.params, observations, temperature
        )
        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def dist_actions(self, observations: np.ndarray, temperature: float = 1.0) -> jnp.ndarray:
        """Get the disctribution of actions."""
        dists = policy.dist_actions(self.actor.apply_fn, self.actor.params, observations, temperature) 
        return dists

    def update(self, batch: Batch, utd_ratio: int = 1) -> InfoDict:
        (
            new_rng,
            new_actor,
            new_critic,
            new_value,
            new_target_critic,
            info,
        ) = _update_jit(
            self.rng,
            self.actor,
            self.critic,
            self.value,
            self.target_critic,
            batch,
            self.discount,
            self.tau,
            self.expectile,
            self.temperature,
            utd_ratio,
            self.num_qs,
            self.num_min_qs,
            self.policy_ddpg_bc
        )

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.value = new_value
        self.target_critic = new_target_critic

        info["mse"] = jnp.mean((batch.actions - self.sample_actions(batch.observations, temperature=0.0)) ** 2)
        import numpy as np

        if np.isnan(info["mse"]):
            import pdb

            pdb.set_trace()

        return info

    def save(self, ckpt_dir, step):
        path = f"{ckpt_dir}/{step}_actor"
        self.actor.save(path)
        path = f"{ckpt_dir}/{step}_critic"
        self.critic.save(path)
        path = f"{ckpt_dir}/{step}_target_critic"
        self.target_critic.save(path)
        path = f"{ckpt_dir}/{step}_value"
        self.value.save(path)

    def load(self, ckpt_dir, step):
        path = f"{ckpt_dir}/{step}_actor"
        self.actor = self.actor.load(path)
        path = f"{ckpt_dir}/{step}_critic"
        self.critic = self.critic.load(path)
        path = f"{ckpt_dir}/{step}_target_critic"
        self.target_critic = self.target_critic.load(path)
        path = f"{ckpt_dir}/{step}_value"
        self.value = self.value.load(path)
