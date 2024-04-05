"""Implementations of algorithms for continuous control."""

from functools import partial
from typing import Callable, Literal, Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from agents.calql.actor import calql_update_actor
from agents.calql import temperature
from agents.calql.critic import target_update, calql_update_critic
from networks import multiplexer, policy, value_net
from networks.common import Batch, ConcatEncoder, InfoDict, Model, PRNGKey, TransformerEncoder


def _share_encoder(source, target):
    replacers = {}

    for k, v in source.params.items():
        if "encoder" in k:
            replacers[k] = v

    # Use critic conv layers in actor:
    new_params = target.params.copy(add_or_replace=replacers)
    return target.replace(params=new_params)


@partial(
    jax.jit,
    static_argnames=(
        "num_qs",
        "num_min_qs",
        "update_target",
        "use_bc",
        "offline_batch_size",
        "utd_ratio",
        "backup_entropy",
        "cql_n_actions",
        "cql_importance_sample",
        "cql_lagrange",
        "cql_max_target_backup",
        "cql_min_q_weight",
        "cql_min_q_weight_online",
        "enable_calql",
    ),
)
def _update_jit(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    target_critic: Model,
    temp: Model,
    batch: Batch,
    discount: float,
    num_qs: int,
    num_min_qs: int,
    tau: float,
    expl_noise: float,
    target_entropy: float,
    backup_entropy: bool,
    update_target: bool,
    use_bc: bool,
    bc_weight: float,
    offline_batch_size: int,
    cql_n_actions: int,
    cql_importance_sample: bool,
    cql_temp: float,
    cql_lagrange: bool,
    cql_max_target_backup: bool,
    cql_min_q_weight: float,
    enable_calql: bool,
    utd_ratio: int,
) -> Tuple[PRNGKey, Model, Model, Model, InfoDict]:
    actor = _share_encoder(source=critic, target=actor)
    rng, key = jax.random.split(rng)
    new_critic, new_target_critic = critic, target_critic
    for i in range(utd_ratio):
        rng, key = jax.random.split(rng)

        def slice(x):
            assert x.shape[0] % utd_ratio == 0
            batch_size = x.shape[0] // utd_ratio
            return x[batch_size * i : batch_size * (i + 1)]

        mini_batch = jax.tree_util.tree_map(slice, batch)
        new_critic, critic_info = calql_update_critic(
            key,
            actor,
            critic,
            target_critic,
            temp,
            mini_batch,
            discount,
            num_qs,
            num_min_qs,
            expl_noise,
            backup_entropy,
            cql_n_actions,
            cql_importance_sample,
            cql_temp,
            cql_max_target_backup,
            cql_min_q_weight,
            enable_calql,
        )
        if update_target:
            new_target_critic = target_update(new_critic, new_target_critic, tau)
        else:
            new_target_critic = new_target_critic

    rng, key = jax.random.split(rng)
    new_actor, actor_info = calql_update_actor(
        key,
        actor,
        new_critic,
        temp,
        mini_batch,
        expl_noise,
        use_bc,
        bc_weight=bc_weight,
        offline_batch_size=offline_batch_size,
    )
    new_temp, alpha_info = temperature.update(temp, actor_info["entropy"], target_entropy)

    return (
        rng,
        new_actor,
        new_critic,
        new_target_critic,
        new_temp,
        {**critic_info, **actor_info, **alpha_info},
    )


class CalQLLearner(object):
    def __init__(
        self,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        actor_lr: float = 1e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        emb_dim: int = 256,
        depth: int = 2,
        num_heads: int = 8,
        discount: float = 0.99,
        tau: float = 0.005,
        target_update_period: int = 1,
        target_entropy: Optional[float] = None,
        backup_entropy: bool = False,
        init_temperature: float = 1.0,
        fixed_alpha: bool = False,
        init_alpha: float = 1.0,
        init_mean: Optional[np.ndarray] = None,
        policy_final_fc_init_scale: float = 1.0,
        dropout_rate: Optional[float] = None,
        num_qs: int = 10,
        num_min_qs: int = 2,
        critic_max_grad_norm: float = None,
        critic_layer_norm: bool = False,
        obs_keys: Sequence[str] = ("image1", "image2"),
        encoder_type: Literal["transformer", "concat"] = "concat",
        normalize_inputs: bool = True,
        activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.leaky_relu,
        use_sigmareparam: bool = True,
        expl_noise: float = 0.1,
        bc_weight: float = 1.0,
        use_bc: bool = False,
        offline_batch_size: int = 128,
        cql_n_actions: int = 10,
        cql_importance_sample: bool = True,
        cql_target_action_gap: float = 1.0,
        cql_temp: float = 1.0,
        cql_lagrange: bool = False,
        cql_max_target_backup: bool = True,
        cql_clip_diff_min: float = -np.inf,
        cql_clip_diff_max: float = np.inf,
        cql_min_q_weight: float = 5.0,
        cql_min_q_weight_online: float = -1.0,
        enable_calql: bool = True,
    ):
        """
        An implementation of the version of CalQL described in https://arxiv.org/abs/2303.05479
        """

        action_dim = actions.shape[-1]
        if target_entropy is None:
            self.target_entropy = -action_dim
        else:
            self.target_entropy = target_entropy

        self.backup_entropy = backup_entropy
        self.fixed_alpha = fixed_alpha
        self.init_alpha = init_alpha

        self.tau = tau
        self.discount = discount
        self.num_qs = num_qs
        self.num_min_qs = num_min_qs
        self.critic_max_grad_norm = critic_max_grad_norm
        self.expl_noise = expl_noise
        self.bc_weight = bc_weight
        self.use_bc = use_bc
        self.offline_batch_size = offline_batch_size
        self.target_update_period = target_update_period

        # CQL Hyperparameters
        self.cql_n_actions = cql_n_actions
        self.cql_importance_sample = cql_importance_sample
        self.cql_lagrange = cql_lagrange
        self.cql_max_target_backup = cql_max_target_backup
        self.cql_target_action_gap = cql_target_action_gap
        self.cql_temp = cql_temp
        self.cql_clip_diff_min = cql_clip_diff_min
        self.cql_clip_diff_max = cql_clip_diff_max
        self.cql_min_q_weight = cql_min_q_weight
        self.cql_min_q_weight_online = cql_min_q_weight_online
        self.enable_calql = enable_calql

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, target_critic_key, temp_key = jax.random.split(rng, 5)

        if encoder_type == "concat":
            print("[INFO] use ConcatEncoder")
            critic_encoder_cls = actor_encoder_cls = partial(ConcatEncoder, obs_keys=obs_keys)
            multiplexer_cls = multiplexer.ConcatMultiplexer
        elif encoder_type.startswith("transformer"):
            critic_encoder_cls = actor_encoder_cls = partial(
                TransformerEncoder,
                emb_dim=emb_dim,
                depth=depth,
                num_heads=num_heads,
                att_drop=0.0 if dropout_rate is None else dropout_rate,
                drop=0.0 if dropout_rate is None else dropout_rate,
                normalize_inputs=normalize_inputs,
                activations=activations,
                use_sigmareparam=use_sigmareparam,
                obs_keys=obs_keys,
                return_intermeidate=encoder_type == "transformer_intermediate",
            )
            if encoder_type == "transformer":
                print("[INFO] use TransformerEncoder")
                multiplexer_cls = multiplexer.SequentialMultiplexer

        actor_cls = partial(
            policy.NormalTanhPolicy,
            hidden_dims,
            action_dim,
            std_min=1e-5,
            std_max=1e-1,
            dropout_rate=dropout_rate,
            state_dependent_std=True,
            tanh_squash_distribution=False,
        )
        actor_def = multiplexer_cls(
            latent_dim=emb_dim,
            encoder_cls=actor_encoder_cls,
            network_cls=actor_cls,
            stop_gradient=True,
        )
        optimiser = optax.adam(actor_lr)

        actor_key, actor_dropout_key = jax.random.split(actor_key)
        actor = Model.create(
            actor_def, inputs=[{"params": actor_key, "dropout": actor_dropout_key}, observations], tx=optimiser
        )

        critic_cls = partial(
            value_net.CriticEnsemble,
            hidden_dims,
            critic_layer_norm=critic_layer_norm,
            num_qs=self.num_qs,
        )
        critic_def = multiplexer_cls(
            latent_dim=emb_dim,
            encoder_cls=critic_encoder_cls,
            network_cls=critic_cls,
            stop_gradient=False,
        )
        critic_key, critic_dropout_key = jax.random.split(critic_key)
        if self.critic_max_grad_norm is not None:
            critic_tx = optax.chain(
                optax.clip_by_global_norm(self.critic_max_grad_norm),
                optax.adam(learning_rate=critic_lr),
            )
        else:
            critic_tx = optax.adam(learning_rate=critic_lr)
        critic = Model.create(
            critic_def,
            inputs=[{"params": critic_key, "dropout": critic_dropout_key}, observations, actions],
            tx=critic_tx,
        )

        target_critic = Model.create(critic_def, inputs=[target_critic_key, observations, actions])
        temp = Model.create(
            temperature.Temperature(init_temperature), inputs=[temp_key], tx=optax.adam(learning_rate=temp_lr)
        )

        if cql_lagrange:
            temp_key, temp_prime_key = jax.random.split(temp_key)
            temp_prime = Model.create(
                temperature.Temperature(init_temperature), inputs=[temp_prime_key], tx=optax.adam(learning_rate=temp_lr)
            )
            self.temp_prime = temp_prime

        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.temp = temp
        self.rng = rng
        self.step = 1

    def sample_actions(self, observations: np.ndarray, expl_noise: float = None) -> jnp.ndarray:
        if expl_noise is None:
            expl_noise = self.expl_noise

        variables = {"params": self.actor.params}
        if self.actor.extra_variables:
            variables.update(self.actor.extra_variables)
        rng, actions = policy.sample_actions(self.rng, self.actor.apply_fn, variables, observations, expl_noise)
        self.rng = rng
        return np.clip(actions, -1, 1)

    def prepare_online_step(self):
        print("do nothing.")
        return

    def update(self, batch: Batch, update_bc: bool = False, utd_ratio: int = 1, **kwargs) -> InfoDict:
        self.step += 1
        cql_min_q_weight = self.cql_min_q_weight if kwargs.get("offline") else self.cql_min_q_weight_online
        (
            new_rng,
            new_actor,
            new_critic,
            new_target_critic,
            new_temp,
            info,
        ) = _update_jit(
            self.rng,
            self.actor,
            self.critic,
            self.target_critic,
            self.temp,
            batch,
            self.discount,
            self.num_qs,
            self.num_min_qs,
            self.tau,
            self.expl_noise,
            self.target_entropy,
            self.backup_entropy,
            self.step % self.target_update_period == 0,
            self.use_bc,
            self.bc_weight,
            self.offline_batch_size,
            self.cql_n_actions,
            self.cql_importance_sample,
            self.cql_temp,
            self.cql_lagrange,
            self.cql_max_target_backup,
            cql_min_q_weight,
            self.enable_calql,
            utd_ratio,
        )
        self.critic = new_critic
        self.target_critic = new_target_critic
        self.temp = new_temp

        self.rng = new_rng
        self.actor = new_actor

        # info["mse"] = jnp.mean((batch.actions - self.sample_actions(batch.observations, expl_noise=0.0)) ** 2)
        # info["actor_mse"] = jnp.mean((batch.actions - self.sample_actions(batch.observations)) ** 2)
        return info

    def save(self, ckpt_dir, step):
        path = f"{ckpt_dir}/{step}_actor"
        self.actor.save(path)
        path = f"{ckpt_dir}/{step}_critic"
        self.critic.save(path)
        path = f"{ckpt_dir}/{step}_target_critic"
        self.target_critic.save(path)
        path = f"{ckpt_dir}/{step}_temperature"
        self.temp.save(path)

    def load(self, ckpt_dir, step):
        path = f"{ckpt_dir}/{step}_actor"
        self.actor = self.actor.load(path)
        path = f"{ckpt_dir}/{step}_critic"
        self.critic = self.critic.load(path)
        path = f"{ckpt_dir}/{step}_target_critic"
        self.target_critic = self.target_critic.load(path)
        path = f"{ckpt_dir}/{step}_temperature"
        self.temp = self.temp.load(path)

    def load_actor(self, ckpt_dir, step):
        path = f"{ckpt_dir}/{step}_actor"
        self.actor = self.actor.load(path)
