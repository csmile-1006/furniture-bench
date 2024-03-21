"""Implementations of algorithms for continuous control."""

from functools import partial
from typing import Callable, Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from agents.iql.iql_learner import _update_bc_jit
from networks import multiplexer, policy
from networks.common import Batch, CrossAttnTransformerEncoder, InfoDict, Model, TransformerEncoder


class BCLearner(object):
    def __init__(
        self,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        actor_optim_kwargs: dict = {
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
        },
        hidden_dims: Sequence[int] = (256, 256),
        emb_dim: int = 256,
        depth: int = 2,
        num_heads: int = 8,
        dropout_rate: Optional[float] = None,
        obs_keys: Sequence[str] = ("image1", "image2"),
        model_type: str = "transformer",
        normalize_inputs: bool = True,
        activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.leaky_relu,
        use_sigmareparam: bool = True,
        expl_noise_init: float = 0.1,
        expl_noise_last: float = 0.01,
        expl_noise_clip: float = 0.1,
        detach_actor: bool = False,
        max_steps: int = 1_000_000,
    ):
        """
        An implementation of Behavior Cloning.
        """

        self.detach_actor = detach_actor

        self.expl_noise_init = expl_noise_init
        self.expl_noise_last = expl_noise_last
        self.expl_noise_clip = expl_noise_clip

        self.max_steps = max_steps

        rng = jax.random.PRNGKey(seed)
        rng, actor_key = jax.random.split(rng, 2)

        if len(observations["image1"].shape) == 3 or len(observations["image1"].shape) == 2:
            observations["image1"] = observations["image1"][np.newaxis]
            observations["image2"] = observations["image2"][np.newaxis]
        if len(observations["robot_state"].shape) == 2:
            observations["robot_state"] = observations["robot_state"][np.newaxis]
        if observations.get("text_feature") is not None and len(observations["text_feature"].shape) == 2:
            observations["text_feature"] = observations["text_feature"][np.newaxis]

        if "text_feature" in obs_keys and model_type == "crossattn":
            print("[INFO] use CrossAttnTransformerEncoder")
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
        actor_cls = partial(
            policy.NormalTanhPolicy,
            hidden_dims,
            action_dim,
            std_min=1e-1,
            std_max=1e-0,
            dropout_rate=dropout_rate,
            state_dependent_std=True,
            tanh_squash_distribution=False,
            obs_keys=obs_keys,
        )
        # actor_cls = partial(
        #     policy.NormalTanhMixturePolicy,
        #     hidden_dims,
        #     action_dim,
        #     num_modes=10,
        #     dropout_rate=dropout_rate,
        #     std_min=1e-1,
        #     std_max=1e-0,
        #     use_tanh=False,
        #     obs_keys=obs_keys,
        # )
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

        self.actor = actor
        self.rng = rng
        self.step = 1

    def _compute_stddev(self, expl_noise):
        mix = np.clip(self.step / self.max_steps, 0.0, 1.0)
        stddev = (1.0 - mix) * self.expl_noise_init + mix * self.expl_noise_last
        return stddev * expl_noise

    def sample_actions(self, observations: np.ndarray, expl_noise: float = 1.0) -> jnp.ndarray:
        variables = {"params": self.actor.params}
        if self.actor.extra_variables:
            variables.update(self.actor.extra_variables)
        rng, actions = policy.sample_actions(self.rng, self.actor.apply_fn, variables, observations, expl_noise)
        self.rng = rng
        return np.array(actions)

    def prepare_online_step(self):
        print("Nothing to do.")
        if self.detach_actor:
            print("detach transformer encoder of BC actor.")
            self.actor.apply_fn.disable_gradient()

    def update(self, batch: Batch, update_bc: bool = False) -> InfoDict:
        self.step += 1
        new_rng, new_actor, info = _update_bc_jit(self.rng, self.actor, batch)
        self.rng = new_rng
        self.actor = new_actor

        info["mse"] = jnp.mean((batch.actions - self.sample_actions(batch.observations, expl_noise=0.0)) ** 2)
        info["actor_mse"] = jnp.mean((batch.actions - self.sample_actions(batch.observations)) ** 2)
        info["stddev"] = self._compute_stddev(1.0)
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
