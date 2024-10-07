import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResBlock, self).__init__()
        inter_dim = out_dim // 2
        # ----------------- Network setting -----------------
        self.res_layer = nn.Sequential(
            nn.Conv2d(in_dim,  inter_dim, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(inter_dim, inter_dim, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(inter_dim, out_dim, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        if in_dim != out_dim:
            self.identity = nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0, stride=1)
        else:
            self.identity = nn.Identity()

    def forward(self, x):
        return self.res_layer(x) + self.identity(x)


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
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, padding=1, stride=stride))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(ResBlock(out_channels, out_channels))

            in_channels = out_channels
            out_channels = out_channels * (2 if i < 3 else 1)

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
    print(y.shape)

    # Compute FLOPs & Params
    print('==============================')
    model.eval()
    x = torch.randn(1, img_dim, img_size, img_size, requires_grad=True)
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))
    