import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4

    config.hidden_dims = (512, 256, 256, 256)

    config.detach_actor = False
    config.expl_noise = 0.1

    config.dropout_rate = None

    config.encoder_type = "concat"
    # transformer setup
    config.emb_dim = 1024
    config.depth = 3
    config.num_heads = 8

    return config
