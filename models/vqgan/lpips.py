import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .resnet import ResNet
except:
    from  resnet import ResNet


# Frozen BatchNormazlizarion
class FrozenBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        # 使用固定的统计信息进行归一化
        x = (x - self.running_mean[None, :, None, None]) / torch.sqrt(self.running_var[None, :, None, None] + self.eps)
        # 应用仿射变换
        return x * self.gamma[None, :, None, None] + self.beta[None, :, None, None]

    def extra_repr(self):
        return '{num_features}, eps={eps}'.format(**self.__dict__)

def replace_bn_with_frozenbn(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            # Frozen BN
            frozen_bn = FrozenBatchNorm2d(child.num_features, eps=child.eps)

            # Copy the BN weight and bias
            frozen_bn.gamma.data.copy_(child.weight.data)
            frozen_bn.beta.data.copy_(child.bias.data)

            # Copy the BN statistics (mean and var)
            frozen_bn.running_mean.copy_(child.running_mean)
            frozen_bn.running_var.copy_(child.running_var)
            setattr(module, name, frozen_bn)
        else:
            replace_bn_with_frozenbn(child)

class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """
    def __init__(self, in_dim, out_dim=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        self.model = nn.Sequential(
            nn.Dropout() if use_dropout else nn.Identity(),
            nn.Conv2d(in_dim, out_dim, 1, stride=1, padding=0, bias=False),
        )

    def forward(self, x):
        return self.model(x)


# ----------- GAN loss modules -----------
class LPIPS(nn.Module):
    def __init__(self,):
        super().__init__()
        print(" - Use ResNet-18 as the feature extractor in LPIPS.")
        self.feat_model = ResNet(name="resnet18", use_pretrained=True).eval()
        self.feat_dims  = [64, 128, 256, 512]
        self.num_feats  = 4

        replace_bn_with_frozenbn(self.feat_model)

        self.lins = nn.ModuleList()
        for i in range(self.num_feats):
            self.lins.append(NetLinLayer(in_dim=self.feat_dims[i], out_dim=1, use_dropout=True))

        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, real_x, fake_x):
        # Preprocess inputs
        real_x_norm = self.preprocess_image(real_x)
        fake_x_norm = self.preprocess_image(fake_x)
        
        # Extract features
        real_x_feats, fake_x_feats = self.resnet_feats(real_x_norm, fake_x_norm)

        # Layer-wise feature losses
        ploss_list = []
        for i, (r_feat, f_feat) in enumerate(zip(real_x_feats, fake_x_feats)):
            feat_real_norm = self.norm_tensor(r_feat)
            feat_fake_norm = self.norm_tensor(f_feat)

            # Diff between each level feature:
            diff_lvl = (feat_real_norm - feat_fake_norm) ** 2
            diff_lvl = torch.mean(self.lins[i](diff_lvl))
            ploss_list.append(diff_lvl)

        ploss = sum(ploss_list)

        return ploss

    def preprocess_image(self, x):
        pixel_mean = torch.as_tensor([0.485, 0.456, 0.406])[None, :, None, None]  # [B, C, H, W]
        pixel_std  = torch.as_tensor([0.229, 0.224, 0.225])[None, :, None, None]  # [B, C, H, W]

        pixel_mean = pixel_mean.to(x.device)
        pixel_std  = pixel_std.to(x.device)

        x = (x - pixel_mean) / pixel_std

        return x

    def norm_tensor(self, x):
        norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
        return x / (norm_factor + 1e-10)  # [B, 1, H, W]

    def resnet_feats(self, real_x, fake_x):
        real_x_feats = self.feat_model(real_x)
        fake_x_feats = self.feat_model(fake_x)


        return real_x_feats, fake_x_feats


if __name__ == '__main__':
    import torch
    from thop import profile

    # Prepare an image as the input
    bs, img_dim, img_size = 4, 3, 256
    real_x = torch.ones(bs, img_dim, img_size, img_size, requires_grad=True)
    fake_x = torch.ones(bs, img_dim, img_size, img_size, requires_grad=True) * 2

    # Build model
    model = LPIPS().eval()

    # Inference
    ploss = model(real_x, fake_x)
    print("PLoss: ", ploss)

    # Compute FLOPs & Params
    print('==============================')
    model.eval()
    real_x = torch.ones(1, img_dim, img_size, img_size, requires_grad=True)
    fake_x = torch.ones(1, img_dim, img_size, img_size, requires_grad=True) * 2
    flops, params = profile(model, inputs=(real_x, fake_x, ), verbose=False)
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))

    for name, param in model.named_parameters():
        print(f"Requires Grad: {param.requires_grad} | Name: {name}, Shape: {param.shape}")
        