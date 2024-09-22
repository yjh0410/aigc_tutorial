from .vqvae import VQVAE

def build_vqvae(img_dim):
    return VQVAE(img_dim, hidden_dim=128, latent_dim=64, num_embeddings=512)
