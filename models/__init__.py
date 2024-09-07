# -------------- Build VAE models --------------
from .vae   import build_vae
from .cvae  import build_cvae
from .vqvae import build_vqvae

def build_model(args):
    if args.model == 'vae':
        return build_vae(args.img_dim)
    if args.model == 'cvae':
        return build_cvae(args.img_dim)
    if args.model == 'vqvae':
        return build_vqvae(args.img_dim)
    
    raise NotImplementedError(f"Unknown model: {args.model}")


# -------------- Build VAE sampler --------------
from .vqvae import build_vqvae_sampler

def build_sampler(args, num_vq_embeds):
    if 'vqvae' in args.model:
        return build_vqvae_sampler(args.sampler_scale, num_vq_embeds)
    
    raise NotImplementedError(f"We could not to build sampler for model - {args.model}")
