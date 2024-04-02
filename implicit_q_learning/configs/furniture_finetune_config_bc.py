import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4

    config.hidden_dims = (512, 256, 256, 256)

    config.expl_noise = 1.0

    config.dropout_rate = None

    config.encoder_type = "transformer"
    # transformer setup
    config.emb_dim = 256
    config.depth = 3
    config.num_heads = 8

    return config
