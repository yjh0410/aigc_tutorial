import torch
import torch.nn as nn

try:
    from .modules import ConvModule
except:
    from  modules import ConvModule


class PatchDiscriminator(nn.Module):
    def __init__(self, in_dim=3, ndf=64, n_layers=3):
        super(PatchDiscriminator, self).__init__()
        self.train_iters = 0
        # ------------ Model parameters ------------
        layers = [ConvModule(in_dim, ndf, kernel_size=4, padding=1, stride=2)]

        num_filters_mult = 1
        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2 ** i, 8)

            in_dim_tmp  = ndf * num_filters_mult_last
            out_dim_tmp = ndf * num_filters_mult
            stride = 2 if i < n_layers else 1
            layers.append(ConvModule(in_dim_tmp, out_dim_tmp, kernel_size=4, padding=1, stride=stride))

        layers.append(nn.Conv2d(out_dim_tmp, 1, kernel_size=4, padding=1, stride=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        self.train_iters += 1
        return self.layers(x)


if __name__ == '__main__':
    import torch
    from thop import profile

    # Prepare an image as the input
    bs, img_dim, img_size = 4, 3, 256
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
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))
    