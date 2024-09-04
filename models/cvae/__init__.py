from .cvae import CondiontalVAE


def build_cvae(img_dim):
    return CondiontalVAE(img_dim, latent_dim=8)
