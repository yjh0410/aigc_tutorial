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
    