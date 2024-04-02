import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr = 3e-4

    config.hidden_dims = (512, 256, 256, 256)

    config.discount = 0.996
    config.use_bc = True
    config.expl_noise = 1.0
    config.bc_weight = 1.0

    config.dropout_rate = None

    config.tau = 0.005  # For soft target updates.
    config.backup_entropy = False
    config.fixed_alpha = False
    config.init_alpha = 1.0
    config.init_temperature = 1.0

    config.encoder_type = "transformer"
    # transformer setup
    config.emb_dim = 256
    config.depth = 3
    config.num_heads = 8

    # critic options
    config.num_qs = 10
    config.num_min_qs = 2
    config.critic_max_grad_norm = None
    config.critic_layer_norm = True

    return config
