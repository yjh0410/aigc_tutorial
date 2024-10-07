import torch
import torch.nn as nn

try:
    from .modules import ResStage
except:
    from  modules import ResStage


class PatchDiscriminator(nn.Module):
    def __init__(self, in_dim=3, ndf=64, n_layers=3):
        super(PatchDiscriminator, self).__init__()
        self.train_iters = 0
        # ------------ Model parameters ------------
        layers = [nn.Conv2d(in_dim, ndf, kernel_size=4, padding=1, stride=2),
                  nn.LeakyReLU(0.2, inplace=True)]

        in_channels = ndf
        out_channels = ndf * 2
        for i in range(1, n_layers + 1):
            stride = 2 if i < n_layers else 1
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, padding=1, stride=stride, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

            in_channels = out_channels
            out_channels = out_channels * (2 if i < 3 else 1)

        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=4, padding=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv2d(out_channels, 1, kernel_size=4, padding=1, stride=1))
        self.layers = nn.Sequential(*layers)

        self.weights_init()
        
    def weights_init(self,):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
        
            if isinstance(m, torch.nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
        
    def forward(self, x):
        return self.layers(x)
    

if __name__ == '__main__':
    import torch
    from thop import profile

    # Prepare an image as the input
    bs, img_dim, img_size = 4, 3, 128
    x = torch.randn(bs, img_dim, img_size, img_size, requires_grad=True)

    # Build model
    model = PatchDiscriminator(in_dim=3, ndf=64, n_layers=3)
    model.train()
    print(model)

    # Inference
    y = model(x)
    print("Disc: ", y)

    # Compute FLOPs & Params
    print('==============================')
    model.eval()
    x = torch.randn(1, img_dim, img_size, img_size, requires_grad=True)
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))
    