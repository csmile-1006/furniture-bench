from typing import Tuple

import jax.numpy as jnp
from flax import linen as nn

from networks.common import InfoDict, Model


class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param("log_temp", init_fn=lambda key: jnp.full((), jnp.log(self.initial_temperature)))
        return jnp.exp(log_temp)


def update(temp: Model, entropy: float, target_entropy: float) -> Tuple[Model, InfoDict]:
    def temperature_loss_fn(temp_params):
        temperature, updated_states = temp.apply(temp_params, mutable=temp.extra_variables.keys())
        temp_loss = temperature * (entropy - target_entropy).mean()
        return temp_loss, {"temperature": temperature, "temp_loss": temp_loss, "updated_states": updated_states}

    new_temp, info = temp.apply_gradient(temperature_loss_fn)

    return new_temp, info
