import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.actor_optim_kwargs = ml_collections.ConfigDict()
    config.actor_optim_kwargs.learning_rate = 3e-4
    config.actor_optim_kwargs.weight_decay = 1e-4

    config.value_lr = 3e-4
    config.critic_lr = 3e-4

    config.hidden_dims = (512, 256, 256, 256)

    config.discount = 0.99
    config.target_update_period = 1

    config.dropout_rate = None
    config.detach_actor = False

    config.tau = 0.005  # For soft target updates.

    config.beta = 2.0

    config.num_samples = 1
    config.expl_noise = 0.2

    # transformer setup
    config.emb_dim = 256
    config.depth = 3
    config.num_heads = 8

    # critic options
    config.num_qs = 2
    config.num_min_qs = 2
    config.critic_max_grad_norm = None
    config.critic_layer_norm = True

    return config
