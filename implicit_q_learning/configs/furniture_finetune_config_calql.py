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
    config.emb_dim = 128
    config.depth = 2
    config.num_heads = 2

    # critic options
    config.num_qs = 2
    config.num_min_qs = 2
    config.critic_max_grad_norm = None
    config.critic_layer_norm = True

    # CQL options
    config.enable_calql = True
    config.cql_n_actions = 4
    config.cql_importance_sample = True
    config.cql_temp = 1.0
    config.cql_lagrange = False
    config.cql_max_target_backup = True
    config.cql_min_q_weight = 5.0
    config.cql_min_q_weight_online = 0.5

    return config
