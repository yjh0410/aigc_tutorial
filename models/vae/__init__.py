from .vae import VAE


def build_vae(img_dim):
    return VAE(img_dim, latent_dim=8)
