import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------- CNN ops -----------
class ConvModule(nn.Module):
    def __init__(self,
                 # ------ Basic conv parameters ------
                 in_dim      :int,
                 out_dim     :int,
                 kernel_size :int = 1,
                 padding     :int = 0,
                 stride      :int = 1,
                ):
        super(ConvModule, self).__init__()
        # ----------- Model parameters -----------
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, padding=padding, stride=stride)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)

        return x

class DeConvModule(nn.Module):
    def __init__(self,
                 # ------ Basic conv parameters ------
                 in_dim      :int,
                 out_dim     :int,
                 kernel_size :int = 1,
                 padding     :int = 0,
                 stride      :int = 1,
                ):
        super(DeConvModule, self).__init__()
        # ----------- Model parameters -----------
        self.conv = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=kernel_size, padding=padding, stride=stride)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)

        return x


# ----------- ResNet modules -----------
class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, shortcut=True):
        super(ResBlock, self).__init__()
        self.shortcut = shortcut and (in_dim == out_dim)
        inter_dim = out_dim // 2
        # ----------------- Network setting -----------------
        self.res_layer = nn.Sequential(
            ConvModule(in_dim,    inter_dim, kernel_size=1, padding=0, stride=1),
            ConvModule(inter_dim, inter_dim, kernel_size=3, padding=1, stride=1),
            ConvModule(inter_dim, out_dim,   kernel_size=1, padding=0, stride=1),
        )

    def forward(self, x):
        h = self.res_layer(x)
        return x + h if self.shortcut else h

class ResStage(nn.Module):
    def __init__(self, in_dim, out_dim, num_blocks=1) -> None:
        super(ResStage, self).__init__()
        # ----------------- Network setting -----------------
        res_blocks = []
        for i in range(num_blocks):
            if i == 0:
                shortcut = (in_dim == out_dim)
                res_blocks.append(ResBlock(in_dim, out_dim, shortcut))
            else:
                res_blocks.append(ResBlock(out_dim, out_dim, True))
        self.res_blocks = nn.Sequential(*res_blocks)

    def forward(self, x):
        return self.res_blocks(x)
