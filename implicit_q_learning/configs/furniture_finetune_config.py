import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4
    config.value_lr = 3e-4
    config.critic_lr = 3e-4

    config.hidden_dims = (512, 256, 256)

    config.discount = 0.996

    config.expectile = 0.8  # The actual tau for expectiles.
    config.temperature = 0.5
    config.dropout_rate = None

    config.tau = 0.005  # For soft target updates.
    config.opt_decay_schedule = None  # Don't decay optimizer lr

    # transformer setup
    config.emb_dim = 512
    config.depth = 2
    config.num_heads = 8

    # layer norm for critic
    config.critic_layer_norm = True
    config.use_rnd = False
    config.beta_rnd = 0.1

    return config
