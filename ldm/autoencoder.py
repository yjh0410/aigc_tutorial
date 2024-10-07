import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

try:
    from .modules import ConvModule, DeConvModule, ResStage
except:
    from  modules import ConvModule, DeConvModule, ResStage


# ------------ VAE Modules ------------
class VaeEncoder(nn.Module):
    def __init__(self,
                 img_dim: int = 3,
                 hidden_dims: List = [32, 64, 128, 256],
                 latent_dim: int = 8,
                 use_attn: bool = False,
                 ):
        super().__init__()
        # Input projection
        layers = []
        layers.append(ConvModule(img_dim, hidden_dims[0], kernel_size=3, padding=1, stride=1))
        layers.append(ResStage(hidden_dims[0], hidden_dims[0], num_blocks=2, use_attn=False))

        # Inter stages
        for i in range(1, len(hidden_dims)):
            _use_attn = use_attn if 2 ** i >= 4 else False
            layers.append(ConvModule(hidden_dims[i-1], hidden_dims[i], kernel_size=3, padding=1, stride=2))
            layers.append(ResStage(hidden_dims[i], hidden_dims[i], num_blocks=2, use_attn=_use_attn))   

        # Output projection
        layers.append(nn.Conv2d(hidden_dims[-1], latent_dim * 2, kernel_size=3, padding=1, stride=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):        
        return self.layers(x)

class VaeDecoder(nn.Module):
    def __init__(self,
                 img_dim: int = 3,
                 hidden_dims: List = [32, 64, 128, 256],
                 latent_dim: int = 8,
                 use_attn: bool = False,
                 ):
        super().__init__()
        # Input projection
        layers = []
        layers.append(ConvModule(latent_dim, hidden_dims[-1], kernel_size=3, padding=1, stride=1))

        # Inter stages
        for i in range(len(hidden_dims)-1, 0, -1):
            _use_attn = use_attn if 2 ** i >= 4 else False
            layers.append(ResStage(hidden_dims[i], hidden_dims[i], num_blocks=2, use_attn=_use_attn))
            layers.append(DeConvModule(hidden_dims[i], hidden_dims[i-1], kernel_size=4, padding=1, stride=2))

        # Input projection
        layers.append(nn.Conv2d(hidden_dims[0], img_dim, kernel_size=3, padding=1, stride=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# ------------ VariationalAE ------------
class VariationalAE(nn.Module):
    def __init__(self,
                 img_dim: int = 3,
                 hidden_dims: List = [64, 128, 256],
                 latent_dim: int = 8,
                 use_attn: bool = False,
                 ):
        super().__init__()
        self.img_dim = img_dim
        self.hidden_dims = hidden_dims
        self.latent_dim  = latent_dim
        
        self.encoder = VaeEncoder(img_dim, hidden_dims, latent_dim, use_attn)
        self.decoder = VaeDecoder(img_dim, hidden_dims, latent_dim, use_attn)

    def gaussian_reparam(self, mu, log_var):
        # sample from N(0, 1) distribution
        z = torch.randn_like(log_var)

        # Reparam: rep_z = \mu + \sigma * z
        rep_z = mu + z * torch.sqrt(log_var.exp())

        return rep_z
    
    def forward_encode_before_reparam(self, x):
        z = self.encoder(x)

        return z
    
    def forward_encode_after_reparam(self, x):
        z = self.encoder(x)
        mu, log_var = torch.chunk(z, chunks=2, dim=1)
        rep_z = self.gaussian_reparam(mu, log_var)

        return rep_z
    
    def forward_decode(self, x):
        return self.decoder(x)
    
    def forward(self, x):
        # Encode
        z = self.encoder(x)
        mu, log_var = torch.chunk(z, chunks=2, dim=1)

        log_var = torch.clamp(log_var, -30.0, 20.0)

        # Decode
        rep_z = self.gaussian_reparam(mu, log_var)
        x_rec = self.forward_decode(rep_z)

        output = {
            "x_pred": x_rec,
            "mu": mu,
            "log_var": log_var,
        }

        return output


if __name__ == '__main__':
    import torch
    from thop import profile

    print(' \n=========== VAE Encoder =========== ')
    # Prepare an image as the input
    bs, img_dim, img_size = 2, 3, 128
    hidden_dims = [64, 128, 256, 512]
    latent_dim = 4
    x = torch.randn(bs, img_dim, img_size, img_size)

    # Build model
    model = VaeEncoder(img_dim, hidden_dims, latent_dim)

    # Inference
    output = model(x)
    print(output.shape)

    # Compute FLOPs & Params
    model.eval()
    x = torch.randn(1, img_dim, img_size, img_size)
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('Encoder FLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Encoder Params : {:.2f} M'.format(params / 1e6))


    print(' \n=========== VAE Decoder =========== ')
    # Prepare an image as the input
    x = torch.randn(bs, latent_dim, img_size // 8, img_size // 8)

    # Build model
    model = VaeDecoder(img_dim, hidden_dims, latent_dim)

    # Inference
    output = model(x)
    print(output.shape)

    # Compute FLOPs & Params
    model.eval()
    x = torch.randn(1, latent_dim, img_size // 8, img_size // 8)
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('Decoder FLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Decoder Params : {:.2f} M'.format(params / 1e6))


    print(' \n=========== VQ-VAE =========== ')
    # Prepare an image as the input
    bs, img_dim, img_size = 2, 3, 128
    x = torch.randn(bs, img_dim, img_size, img_size)

    # Build model
    model = VariationalAE(img_dim, hidden_dims, latent_dim, use_attn=True)
    print(model)

    # Inference
    output = model(x)

    # Compute FLOPs & Params
    model.eval()
    x = torch.randn(1, img_dim, img_size, img_size)
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('VAE GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('VAE Params : {:.2f} M'.format(params / 1e6))
