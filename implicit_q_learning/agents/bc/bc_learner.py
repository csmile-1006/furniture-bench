"""Implementations of algorithms for continuous control."""

from functools import partial
from typing import Callable, Literal, Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from agents.bc.actor import bc_update_actor
from networks import multiplexer, policy
from networks.common import Batch, ConcatEncoder, InfoDict, Model, PRNGKey, TransformerEncoder


@partial(jax.jit)
def _update_bc_jit(
    rng: PRNGKey, actor: Model, batch: Batch, expl_noise: float
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, InfoDict]:
    key, rng = jax.random.split(rng)
    new_actor, actor_info = bc_update_actor(key, actor, batch, expl_noise)

    return (
        rng,
        new_actor,
        actor_info,
    )


class BCLearner(object):
    def __init__(
        self,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        actor_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        emb_dim: int = 256,
        depth: int = 2,
        num_heads: int = 8,
        dropout_rate: Optional[float] = None,
        obs_keys: Sequence[str] = ("image1", "image2"),
        encoder_type: Literal["transformer", "concat"] = "concat",
        normalize_inputs: bool = True,
        activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.leaky_relu,
        use_sigmareparam: bool = True,
        expl_noise: float = 1.0,
    ):
        """
        An implementation of Behavior Cloning.
        """

        self.expl_noise = expl_noise

        rng = jax.random.PRNGKey(seed)
        rng, actor_key = jax.random.split(rng, 2)

        if len(observations["image1"].shape) == 3 or len(observations["image1"].shape) == 2:
            observations["image1"] = observations["image1"][np.newaxis]
            observations["image2"] = observations["image2"][np.newaxis]
        if len(observations["robot_state"].shape) == 2:
            observations["robot_state"] = observations["robot_state"][np.newaxis]
        if observations.get("text_feature") is not None and len(observations["text_feature"].shape) == 2:
            observations["text_feature"] = observations["text_feature"][np.newaxis]

        if encoder_type == "concat":
            print("[INFO] use ConcatEncoder")
            actor_encoder_cls = partial(ConcatEncoder, obs_keys=obs_keys)
            multiplexer_cls = multiplexer.ConcatMultiplexer
        elif encoder_type.startswith("transformer"):
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
                obs_keys=obs_keys,
                return_intermeidate=encoder_type == "transformer_intermediate",
            )
            if encoder_type == "transformer":
                print("[INFO] use TransformerEncoder")
                multiplexer_cls = multiplexer.SequentialMultiplexer
            if encoder_type == "transformer_intermediate":
                print("[INFO] use TransformerEncoder with intermediate values.")
                multiplexer_cls = multiplexer.SequentialInterMediateMultiplexer

        action_dim = actions.shape[-1]
        actor_cls = partial(
            policy.NormalTanhPolicy,
            hidden_dims,
            action_dim,
            std_min=0.01,
            std_max=0.2,
            dropout_rate=dropout_rate,
            state_dependent_std=True,
            tanh_squash_distribution=False,
        )
        actor_def = multiplexer_cls(
            latent_dim=emb_dim,
            encoder_cls=actor_encoder_cls,
            network_cls=actor_cls,
            stop_gradient=False,
        )
        optimiser = optax.adam(actor_lr)

        actor_key, actor_dropout_key = jax.random.split(actor_key)
        actor = Model.create(
            actor_def, inputs=[{"params": actor_key, "dropout": actor_dropout_key}, observations], tx=optimiser
        )

        self.actor = actor
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
        return np.array(actions)

    def prepare_online_step(self):
        print("Nothing to do.")

    def update(self, batch: Batch, update_bc: bool = False, utd_ratio=1, **kwargs) -> InfoDict:
        self.step += 1
        new_rng, new_actor, info = _update_bc_jit(self.rng, self.actor, batch, 1.0)
        self.rng = new_rng
        self.actor = new_actor

        info["mse"] = jnp.mean((batch.actions - self.sample_actions(batch.observations, expl_noise=0.0)) ** 2)
        info["actor_mse"] = jnp.mean((batch.actions - self.sample_actions(batch.observations)) ** 2)
        return info

    def save(self, ckpt_dir, step):
        path = f"{ckpt_dir}/{step}_actor"
        self.actor.save(path)

    def load(self, ckpt_dir, step):
        path = f"{ckpt_dir}/{step}_actor"
        self.actor = self.actor.load(path)

    def load_actor(self, ckpt_dir, step):
        path = f"{ckpt_dir}/{step}_actor"
        self.actor = self.actor.load(path)
