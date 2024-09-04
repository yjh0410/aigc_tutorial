import torch
import torch.nn as nn

try:
    from .modules import ResStage
except:
    from  modules import ResStage


# ------------------ VAE Modules ------------------
class VqVaeEncoder(nn.Module):
    def __init__(self, img_dim=3, hidden_dim=256):
        super().__init__()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(img_dim, hidden_dim // 2, kernel_size=4, padding=1, stride=2),
            nn.ReLU(inplace=True),
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=4, padding=1, stride=2),
            nn.ReLU(inplace=True),
        )

        self.layer_3 = ResStage(hidden_dim, hidden_dim, num_blocks=2)
        self.layer_4 = ResStage(hidden_dim, hidden_dim, num_blocks=2)
        
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
        
        return x
    

if __name__ == '__main__':
    import torch
    from thop import profile

    # Prepare an image as the input
    bs, img_dim, img_size = 2, 3, 128
    x = torch.randn(bs, img_dim, img_size, img_size)

    # Build model
    model = VqVaeEncoder(img_dim, hidden_dim=256)

    # Inference
    z = model(x)
    print(z.shape)

    # Compute FLOPs & Params
    print('==============================')
    model.eval()
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))

