import torch
import torch.nn as nn

try:
    from .modules import ResStage, DeConvModule
except:
    from  modules import ResStage, DeConvModule


# ------------ VQ-VAE Decoder ------------
class VqGanDecoder(nn.Module):
    def __init__(self, img_dim=3, hidden_dim=256, latent_dim=128):
        super().__init__()
        # 2x upsampling
        self.layer_1 = ResStage(latent_dim, hidden_dim, num_blocks=3)
        self.layer_2 = DeConvModule(hidden_dim, hidden_dim // 2, kernel_size=4, padding=1, stride=2)
        # 4x upsampling
        self.layer_3 = ResStage(hidden_dim // 2, hidden_dim // 2, num_blocks=3)
        self.layer_4 = DeConvModule(hidden_dim // 2, hidden_dim // 4, kernel_size=4, padding=1, stride=2)
        self.last_conv = nn.Conv2d(hidden_dim // 4, img_dim, kernel_size=3, padding=1, stride=1)

        # Initialize all layers
        self.init_weights()

    def init_weights(self):
        """Initialize the parameters."""
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.reset_parameters()

    def forward(self, x):
        # From latent space into image space: [B, Z] -> [B, N] -> [B, C, H, W]
        x = self.layer_1(x)
        x = self.layer_2(x)

        x = self.layer_3(x)
        x = self.layer_4(x)

        x = self.last_conv(x)

        return x

    

if __name__ == '__main__':
    import torch
    from thop import profile

    # Prepare an image as the input
    bs, hidden_dim, img_size = 2, 256, 32
    latent_dim = 128
    x = torch.randn(bs, latent_dim, img_size, img_size)

    # Build model
    model = VqGanDecoder(3, hidden_dim, latent_dim)
    print(model)

    # Inference
    y = model(x)
    print(y.shape)

    # Compute FLOPs & Params
    print('==============================')
    model.eval()
    x = torch.randn(1, latent_dim, img_size, img_size)
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))
    