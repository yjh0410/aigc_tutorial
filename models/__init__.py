# -------------- Build VAE models --------------
from .vae   import build_vae
from .cvae  import build_cvae
from .vqvae import build_vqvae
from .vqgan import build_vqgan


def build_model(args):
    if args.model == 'vae':
        return build_vae(args.img_dim)
    
    if args.model == 'cvae':
        return build_cvae(args.img_dim)
    
    if args.model == 'vqvae':
        return build_vqvae(args.img_dim)
    
    if args.model == 'vqgan':
        return build_vqgan(args.img_dim)
    
    raise NotImplementedError(f"Unknown model: {args.model}")


# -------------- Build VAE sampler --------------
from .sampler import build_gpt_sampler

def build_sampler(args, num_vq_embeds):
    if 'gpt' in args.sampler:
        return build_gpt_sampler(args.sampler, num_vq_embeds)
    
    raise NotImplementedError(f"We could not to build sampler for model - {args.model}")
