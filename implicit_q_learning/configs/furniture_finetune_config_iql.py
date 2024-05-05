import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4
    config.value_lr = 3e-4
    config.critic_lr = 3e-4

    config.hidden_dims = (512, 256, 256)

    config.discount = 0.996

    config.expectile = 0.8  # The actual tau for expectiles.
    # config.A_scaling = 10.0

    config.dropout_rate = None
    # config.detach_actor = True

    config.tau = 0.005  # For soft target updates.
    config.opt_decay_schedule = None  # Don't decay optimizer lr

    # config.expl_noise = 0.1
    config.expl_noise = 1.0


    # config.encoder_type = "transformer"

    # transformer setup
    config.emb_dim = 512
    config.depth = 3
    config.num_heads = 8

    # critic options
    # config.num_qs = 2
    # config.num_min_qs = 2
    # config.critic_max_grad_norm = None
    config.critic_layer_norm = True

    return config
