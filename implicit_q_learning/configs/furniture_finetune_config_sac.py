import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr = 3e-4

    config.hidden_dims = (512, 256, 256, 256)

    config.discount = 0.99
    config.use_bc = True
    config.expl_noise = 0.1
    config.bc_weight = 1.0
    config.detach_actor = True

    config.dropout_rate = None

    config.tau = 0.005  # For soft target updates.
    config.backup_entropy = False

    config.encoder_type = "transformer"
    # transformer setup
    config.emb_dim = 1024
    config.depth = 3
    config.num_heads = 8

    # critic options
    config.num_qs = 10
    config.num_min_qs = 1
    config.critic_max_grad_norm = None
    config.critic_layer_norm = True

    return config
