import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.actor_optim_kwargs = ml_collections.ConfigDict()
    config.actor_optim_kwargs.learning_rate = 3e-4
    config.actor_optim_kwargs.weight_decay = 1e-4

    config.hidden_dims = (512, 256, 256, 256)

    config.detach_actor = True

    config.expl_noise_init = 0.2
    config.expl_noise_last = 0.02
    config.expl_noise_clip = 0.06

    config.dropout_rate = None

    # transformer setup
    config.emb_dim = 256
    config.depth = 3
    config.num_heads = 8

    return config
