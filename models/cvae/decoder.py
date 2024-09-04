import torch
import torch.nn as nn

try:
    from .modules import ResBlock
except:
    from  modules import ResBlock


# ------------------ VAE Modules ------------------
class CVaeDecoder(nn.Module):
    def __init__(self, img_dim=3, latent_dim=8):
        super().__init__()
        # ---------- Basic parameters ----------
        self.latent_dim = latent_dim
        # ---------- Model parameters ----------
        self.layer_1 = nn.Sequential(
            nn.Conv2d(latent_dim + 1, 128, kernel_size=1, padding=0, stride=1),
            nn.ReLU(inplace=True),
        )
        self.layer_2 = ResBlock(128, 128, shortcut=True)
        self.layer_3 = ResBlock(128, 128, shortcut=True)
        self.layer_4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, padding=0, stride=2),
            nn.ReLU(inplace=True),
        )
        self.layer_5 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, padding=0, stride=2),
            nn.ReLU(inplace=True),
        )
        self.layer_6 = nn.Conv2d(64, img_dim, kernel_size=1, padding=0, stride=1)

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
        x = self.layer_5(x)
        x = self.layer_6(x)

        return x
    

if __name__ == '__main__':
    import torch
    from thop import profile

    # Prepare an image as the input
    bs, latent_dim, img_size = 2, 8, 32
    x = torch.randn(bs, latent_dim, img_size, img_size)

    # Build model
    model = CVaeDecoder(latent_dim=8)

    # Inference
    outputs = model(x)
    print(outputs.shape)

    # Compute FLOPs & Params
    print('==============================')
    model.eval()
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))

