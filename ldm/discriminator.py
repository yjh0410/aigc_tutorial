import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# --------- Spectral norm based modules ---------
class SpectralConv(nn.Module):
    def __init__(self,
                 in_dim      :int,
                 out_dim     :int,
                 kernel_size :int = 1,
                 padding     :int = 0,
                 stride      :int = 1,
                ):
        super(SpectralConv, self).__init__()
        # ----------- Model parameters -----------
        self.conv = spectral_norm(nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, padding=padding, stride=stride))
        self.act  = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)

        return x

class SpectralResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SpectralResBlock, self).__init__()
        # ----------------- Network setting -----------------
        self.res_layer = nn.Sequential(
            SpectralConv(in_dim,  out_dim, kernel_size=3, padding=1, stride=1),
            SpectralConv(out_dim, out_dim, kernel_size=3, padding=1, stride=1),
        )

    def forward(self, x):
        return self.res_layer(x) + x


# --------- Patch Discriminator ---------
class PatchDiscriminator(nn.Module):
    def __init__(self, in_dim=3, ndf=64):
        super(PatchDiscriminator, self).__init__()
        self.train_iters = 0
        # ------------ Model parameters ------------
        self.layer_1 = SpectralConv(in_dim, ndf, kernel_size=4, padding=1, stride=2)
        self.layer_2 = SpectralResBlock(ndf, ndf)          # 2x downsample

        self.layer_3 = SpectralConv(ndf, ndf * 2, kernel_size=4, padding=1, stride=2)
        self.layer_4 = SpectralResBlock(ndf * 2, ndf * 2)  # 4x downsample

        self.layer_5 = SpectralConv(ndf * 2, ndf * 4, kernel_size=4, padding=1, stride=2)
        self.layer_6 = SpectralResBlock(ndf * 4, ndf * 4)  # 8x downsample

        self.layer_7 = SpectralConv(ndf * 4, ndf * 8, kernel_size=4, padding=1, stride=1)
        self.layer_8 = SpectralResBlock(ndf * 8, ndf * 8)  # 8x downsample
        
        self.layer_9 = nn.Conv2d(ndf * 8, 1, kernel_size=4, padding=1, stride=1)

        self.weights_init()
        
    def weights_init(self,):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
        
            if isinstance(m, torch.nn.GroupNorm):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
        
    def _forward_impl(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.layer_6(x)
        x = self.layer_7(x)
        x = self.layer_8(x)
        x = self.layer_9(x)
        
        return x
    
    def calc_disc_loss(self, real_x, fake_x):
        disc_real = self._forward_impl(real_x.detach())
        disc_fake = self._forward_impl(fake_x.detach())
        
        bs, _, h, w = disc_real.shape

        label_real = torch.ones([bs, 1, h, w], dtype=disc_real.dtype, device=disc_real.device)
        d_loss_real = F.binary_cross_entropy_with_logits(disc_real, label_real, reduction='mean')

        label_fake = torch.zeros([bs, 1, h, w], dtype=disc_fake.dtype, device=disc_fake.device)
        d_loss_fake = F.binary_cross_entropy_with_logits(disc_fake, label_fake, reduction='mean')

        return d_loss_real + d_loss_fake

    def calc_adv_loss(self, fake_x):
        disc_fake = self._forward_impl(fake_x)
        
        bs, _, h, w = disc_fake.shape

        label_fake = torch.ones([bs, 1, h, w], dtype=disc_fake.dtype, device=disc_fake.device)
        d_loss_fake = F.binary_cross_entropy_with_logits(disc_fake, label_fake, reduction='mean')

        return d_loss_fake

    def forward(self, x):
        return self._forward_impl(x)
    
if __name__ == '__main__':
    import torch
    from thop import profile

    # Prepare an image as the input
    bs, img_dim, img_size = 4, 3, 128
    x = torch.randn(bs, img_dim, img_size, img_size)

    # Build model
    model = PatchDiscriminator(in_dim=3, ndf=64)
    model.train()
    print(model)

    # Inference
    y = model(x)
    print(y.shape)

    # Compute FLOPs & Params
    print('==============================')
    model.eval()
    x = torch.randn(1, img_dim, img_size, img_size)
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))
    