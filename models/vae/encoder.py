import torch
import torch.nn as nn

try:
    from .modules import ResBlock
except:
    from  modules import ResBlock


# ------------------ VAE Modules ------------------
class VaeEncoder(nn.Module):
    def __init__(self, img_dim=3, latent_dim=8):
        super().__init__()
        self.latent_dim = latent_dim
        # ---------- Model parameters ----------
        self.layer_1 = nn.Sequential(
            nn.Conv2d(img_dim, 64, kernel_size=2, padding=0, stride=2),
            nn.ReLU(inplace=True),
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=2, padding=0, stride=2),
            nn.ReLU(inplace=True),
        )
        self.layer_3 = ResBlock(128, 128, shortcut=True)
        self.layer_4 = ResBlock(128, 128, shortcut=True)
        self.layer_5 = nn.Conv2d(128, 2 * latent_dim, kernel_size=1, padding=0, stride=1)
        
        # Initialize all layers
        self.init_weights()

    def init_weights(self):
        """Initialize the parameters."""
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.reset_parameters()

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        
        mu, log_var = torch.chunk(x, 2, dim=1)

        return mu, log_var
    

if __name__ == '__main__':
    import torch
    from thop import profile

    # Prepare an image as the input
    bs, img_dim, img_size = 2, 3, 128
    x = torch.randn(bs, img_dim, img_size, img_size)

    # Build model
    model = VaeEncoder(latent_dim=8)

    # Inference
    z = model(x)

    # Compute FLOPs & Params
    print('==============================')
    model.eval()
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))

