"""Implementations of algorithms for continuous control."""

from functools import partial
from typing import Callable, Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from agents.awac.actor import awac_update_actor
from agents.awac.critic import awac_update_critic, target_update
from agents.iql.iql_learner import _update_bc_jit
from networks import multiplexer, policy, value_net
from networks.common import Batch, CrossAttnTransformerEncoder, InfoDict, Model, PRNGKey, TransformerEncoder


def _share_encoder(source, target):
    replacers = {}

    for k, v in source.params.items():
        if "encoder" in k:
            replacers[k] = v

    # Use critic conv layers in actor:
    new_params = target.params.copy(add_or_replace=replacers)
    return target.replace(params=new_params)


@partial(jax.jit, static_argnames=("update_target", "num_samples"))
def _update_jit(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    target_critic: Model,
    batch: Batch,
    discount: float,
    tau: float,
    num_samples: int,
    beta: float,
    temperature: float,
    update_target: bool,
) -> Tuple[PRNGKey, Model, Model, Model, InfoDict]:
    rng, key = jax.random.split(rng)
    new_critic, critic_info = awac_update_critic(
        key, actor, critic, target_critic, None, batch, discount, temperature, backup_entropy=False
    )
    if update_target:
        new_target_critic = target_update(new_critic, target_critic, tau)
    else:
        new_target_critic = target_critic

    rng, key = jax.random.split(rng)
    new_actor, actor_info = awac_update_actor(key, actor, new_critic, batch, num_samples, beta, temperature)

    return (
        rng,
        new_actor,
        new_critic,
        new_target_critic,
        {**critic_info, **actor_info},
    )


class AWACLearner(object):
    def __init__(
        self,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        actor_optim_kwargs: dict = {
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
        },
        value_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        emb_dim: int = 256,
        depth: int = 2,
        num_heads: int = 8,
        num_samples: int = 1,
        discount: float = 0.99,
        tau: float = 0.005,
        dropout_rate: Optional[float] = None,
        critic_layer_norm: bool = False,
        obs_keys: Sequence[str] = ("image1", "image2"),
        model_type: str = "transformer",
        normalize_inputs: bool = True,
        activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.leaky_relu,
        use_sigmareparam: bool = True,
        target_update_period: int = 1,
        beta: float = 1.0,
        temperature: float = 1.0,
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1801.01290
        """

        self.tau = tau
        self.target_update_period = target_update_period
        self.discount = discount
        self.num_samples = num_samples
        self.beta = beta
        self.temperature = temperature

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, target_critic_key = jax.random.split(rng, 4)

        if len(observations["image1"].shape) == 3 or len(observations["image1"].shape) == 2:
            observations["image1"] = observations["image1"][np.newaxis]
            observations["image2"] = observations["image2"][np.newaxis]
        if len(observations["robot_state"].shape) == 2:
            observations["robot_state"] = observations["robot_state"][np.newaxis]
        if observations.get("text_feature") is not None and len(observations["text_feature"].shape) == 2:
            observations["text_feature"] = observations["text_feature"][np.newaxis]

        if "text_feature" in obs_keys and model_type == "crossattn":
            print("[INFO] use CrossAttnTransformerEncoder")
            critic_encoder_cls = partial(
                CrossAttnTransformerEncoder,
                emb_dim=emb_dim,
                depth=depth,
                num_heads=num_heads,
                att_drop=0.0 if dropout_rate is None else dropout_rate,
                drop=0.0 if dropout_rate is None else dropout_rate,
                normalize_inputs=normalize_inputs,
                activations=activations,
                use_sigmareparam=use_sigmareparam,
            )
            actor_encoder_cls = partial(
                CrossAttnTransformerEncoder,
                emb_dim=emb_dim,
                depth=depth,
                num_heads=num_heads,
                att_drop=0.0 if dropout_rate is None else dropout_rate,
                drop=0.0 if dropout_rate is None else dropout_rate,
                normalize_inputs=normalize_inputs,
                activations=activations,
                use_sigmareparam=use_sigmareparam,
            )
        else:
            print("[INFO] use TransformerEncoder")
            critic_encoder_cls = partial(
                TransformerEncoder,
                emb_dim=emb_dim,
                depth=depth,
                num_heads=num_heads,
                att_drop=0.0 if dropout_rate is None else dropout_rate,
                drop=0.0 if dropout_rate is None else dropout_rate,
                normalize_inputs=normalize_inputs,
                activations=activations,
                use_sigmareparam=use_sigmareparam,
            )
            actor_encoder_cls = partial(
                TransformerEncoder,
                emb_dim=emb_dim,
                depth=depth,
                num_heads=num_heads,
                att_drop=0.0 if dropout_rate is None else dropout_rate,
                drop=0.0 if dropout_rate is None else dropout_rate,
                normalize_inputs=normalize_inputs,
                activations=activations,
                use_sigmareparam=use_sigmareparam,
            )

        action_dim = actions.shape[-1]
        # actor_cls = partial(
        #     policy.NormalTanhPolicy,
        #     hidden_dims,
        #     action_dim,
        #     log_std_scale=1e-3,
        #     log_std_min=-5.0,
        #     dropout_rate=dropout_rate,
        #     state_dependent_std=False,
        #     tanh_squash_distribution=False,
        #     obs_keys=obs_keys,
        # )
        actor_cls = partial(
            policy.NormalTanhMixturePolicy,
            hidden_dims,
            action_dim,
            num_modes=10,
            dropout_rate=dropout_rate,
            std_min=1e-4,
            std_max=3e-2,
            use_tanh=False,
            obs_keys=obs_keys,
        )
        actor_def = multiplexer.Multiplexer(
            encoder_cls=actor_encoder_cls,
            network_cls=actor_cls,
            stop_gradient=False,
        )
        optimiser = optax.adamw(**actor_optim_kwargs)

        actor_key, actor_dropout_key = jax.random.split(actor_key)
        actor = Model.create(
            actor_def, inputs=[{"params": actor_key, "dropout": actor_dropout_key}, observations], tx=optimiser
        )

        critic_cls = partial(
            value_net.DoubleCritic, hidden_dims, emb_dim, critic_layer_norm=critic_layer_norm, obs_keys=obs_keys
        )
        critic_def = multiplexer.Multiplexer(
            encoder_cls=critic_encoder_cls,
            network_cls=critic_cls,
            stop_gradient=False,
        )
        critic_key, critic_dropout_key = jax.random.split(critic_key)
        critic = Model.create(
            critic_def,
            inputs=[{"params": critic_key, "dropout": critic_dropout_key}, observations, actions],
            tx=optax.adam(learning_rate=critic_lr),
        )

        target_critic = Model.create(critic_def, inputs=[target_critic_key, observations, actions])

        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.rng = rng
        self.step = 1

    def sample_actions(self, observations: np.ndarray, temperature: float = 1.0) -> jnp.ndarray:
        variables = {"params": self.actor.params}
        if self.actor.extra_variables:
            variables.update(self.actor.extra_variables)
        rng, actions = policy.sample_actions(self.rng, self.actor.apply_fn, variables, observations, temperature)
        self.rng = rng
        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def prepare_online_step(self):
        print("transfer pre-trained transformer encoder from BC actor.")
        self.critic = _share_encoder(source=self.actor, target=self.critic)
        # if use_bc:
        #     print("detach transformer encoder of BC actor.")
        #     self.actor.apply_fn.disable_gradient()

    def update(self, batch: Batch, update_bc: bool = False) -> InfoDict:
        self.step += 1
        if update_bc:
            new_rng, new_actor, info = _update_bc_jit(self.rng, self.actor, batch)
        else:
            (
                new_rng,
                new_actor,
                new_critic,
                new_target_critic,
                info,
            ) = _update_jit(
                self.rng,
                self.actor,
                self.critic,
                self.target_critic,
                batch,
                self.discount,
                self.tau,
                self.num_samples,
                self.beta,
                self.temperature,
                self.step % self.target_update_period == 0,
            )
            self.critic = new_critic
            self.target_critic = new_target_critic

        self.rng = new_rng
        self.actor = new_actor

        info["mse"] = jnp.mean((batch.actions - self.sample_actions(batch.observations, temperature=0.0)) ** 2)
        info["actor_mse"] = jnp.mean((batch.actions - self.sample_actions(batch.observations)) ** 2)
        return info

    def save(self, ckpt_dir, step):
        path = f"{ckpt_dir}/{step}_actor"
        self.actor.save(path)
        path = f"{ckpt_dir}/{step}_critic"
        self.critic.save(path)
        path = f"{ckpt_dir}/{step}_target_critic"
        self.target_critic.save(path)

    def load(self, ckpt_dir, step):
        path = f"{ckpt_dir}/{step}_actor"
        self.actor = self.actor.load(path)
        path = f"{ckpt_dir}/{step}_critic"
        self.critic = self.critic.load(path)
        path = f"{ckpt_dir}/{step}_target_critic"
        self.target_critic = self.target_critic.load(path)

    def load_actor(self, ckpt_dir, step):
        path = f"{ckpt_dir}/{step}_actor"
        self.actor = self.actor.load(path)
