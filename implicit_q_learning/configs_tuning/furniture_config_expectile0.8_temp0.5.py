import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4
    config.value_lr = 3e-4
    config.critic_lr = 3e-4

    config.hidden_dims = (512, 256, 256)

    config.discount = 0.996

    config.expectile = 0.7  # The actual tau for expectiles.
    config.temperature = 0.5
    config.dropout_rate = None
    config.opt_decay_schedule = None

    config.tau = 0.005  # For soft target updates.

    return config
