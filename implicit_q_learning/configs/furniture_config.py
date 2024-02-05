import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 1e-4
    config.value_lr = 1e-4
    config.critic_lr = 1e-4

    config.hidden_dims = (512, 256, 256)

    config.discount = 0.99

    config.expectile = 0.9  # The actual tau for expectiles.
    config.temperature = 1.0
    config.dropout_rate = None

    config.tau = 0.01  # For soft target updates.

    # transformer setup
    config.emb_dim = 512
    config.depth = 6
    config.num_heads = 8

    return config
