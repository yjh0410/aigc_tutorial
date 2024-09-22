from .vqgan import VQGAN
from .lpips import LPIPS
from .discriminator import PatchDiscriminator


def build_vqgan(img_dim):
    return VQGAN(img_dim, num_embeddings=512, hidden_dim=128, latent_dim=64)
