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
    config.policy_delay = 2
    config.alpha = 2.5
    config.use_td3_bc = True
    config.bc_weight = 1.0
    config.detach_actor = True

    config.expl_noise_init = 0.1
    config.expl_noise_last = 0.01
    config.expl_noise_clip = 0.03

    config.dropout_rate = None

    config.tau = 0.005  # For soft target updates.

    # transformer setup
    config.emb_dim = 256
    config.depth = 3
    config.num_heads = 8

    # layer norm for critic
    config.critic_layer_norm = True

    return config
