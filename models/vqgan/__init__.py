from .vqgan import VQGAN


def build_vqgan(img_dim):
    return VQGAN(img_dim, hidden_dim=128, latent_dim=64, num_embeddings=512)
