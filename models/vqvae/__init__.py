from .vqvae import VqVAE
from .sampler import VqVaeSampler


def build_vqvae(img_dim):
    return VqVAE(img_dim, hidden_dim=128, latent_dim=64, num_embeddings=512)

def build_vqvae_sampler(scale='tiny', num_vq_embeds=1024):
    if scale == 'base':
        gpt_config = {
            'num_layers': 24,
            'num_heads': 16,
            'embed_dim': 1024,
            'max_seq_len': 2048,
            'rope_theta': 50000,
            'sos_token_id': 0,
            }
    elif scale == 'tiny':
        gpt_config = {
            'num_layers': 12,
            'num_heads': 3,
            'embed_dim': 192,
            'max_seq_len': 2048,
            'rope_theta': 50000,
            'sos_token_id': 0,
            }
    elif scale == 'nano':
        gpt_config = {
            'num_layers': 6,
            'num_heads': 2,
            'embed_dim': 128,
            'max_seq_len': 2048,
            'rope_theta': 50000,
            'sos_token_id': 0,
            }
    else:
        raise NotImplementedError(f"Unknown scale for VqVaeSampler: {scale}")

    return VqVaeSampler(gpt_config, num_vq_embeds)
